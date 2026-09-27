"""
Jerry – autonomer Day-Trading-Bot (Paper-Trading, kein echtes Geld).

Läuft NICHT innerhalb der Streamlit-App, sondern als eigenständiges Skript,
das per GitHub Actions (siehe .github/workflows/jerry_bot.yml) alle 15 Minuten
gestartet wird – unabhängig davon, ob irgendjemand die Streamlit-App geöffnet hat.

Regeln:
- Samstag & Sonntag (UTC-Wochentag): nur Krypto (Aktienmärkte sind ohnehin zu).
- Montag bis Freitag: nur Aktien.
- Rein regelbasiert (EMA/RSI/ATR), long-only, ein Trade gleichzeitig.
- Jede Order wird mit Zeitstempel in Supabase (scanner_paper_orders) protokolliert.

Voraussetzung: Umgebungsvariablen SUPABASE_URL und SUPABASE_SECRET_KEY sind gesetzt
(in GitHub Actions als Repository-Secrets hinterlegt, siehe Setup-Anleitung).
"""
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client

ACCOUNT_KEY = "jerry_bot_v1"
STARTING_CASH = 10_000.0
RISK_PERCENT = 2.0
SCAN_LIMIT = 80  # wie viele Assets pro Lauf geprüft werden (Zeitbudget in GitHub Actions ist begrenzt)

DEFAULT_PARAMS = {
    "ema_fast": 9, "ema_slow": 21, "rsi_period": 14, "atr_period": 14,
    "atr_mult": 1.5, "reward_risk": 2.0, "rsi_bull": (50, 70), "rsi_bear": (30, 50),
}


# ------------------------------------------------------------
# Universum aus app.py extrahieren (kein Import von app.py selbst,
# da app.py Streamlit-UI-Code enthält, der außerhalb von `streamlit run` crasht)
# ------------------------------------------------------------
def load_universe(app_py_path: str = "app.py") -> tuple[dict, dict]:
    with open(app_py_path, "r", encoding="utf-8") as f:
        text = f.read()
    pairs = re.findall(r'SCANNER_UNIVERSE\["([^"]+)"\]\s*=\s*"([^"]+)"', text)
    crypto = {label: ticker for label, ticker in pairs if ticker.endswith("-USD")}
    stocks = {label: ticker for label, ticker in pairs if not ticker.endswith("-USD")}
    return stocks, crypto


# ------------------------------------------------------------
# Indikatoren & Setup-Logik (identisch zur App, damit Ergebnisse konsistent sind)
# ------------------------------------------------------------
def _trade_indicators(df: pd.DataFrame, ema_fast=9, ema_slow=21, rsi_period=14, atr_period=14) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"]
    data["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    data["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, 1e-9)
    data["rsi"] = 100 - (100 / (1 + rs))

    high, low, prev_close = data["High"], data["Low"], close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    data["atr"] = tr.rolling(atr_period).mean()
    data["vol_avg"] = data["Volume"].rolling(20).mean() if "Volume" in data.columns else np.nan
    return data


def calculate_trade_setup(df: pd.DataFrame, params: dict | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    data = _trade_indicators(df, p["ema_fast"], p["ema_slow"], p["rsi_period"], p["atr_period"])
    row = data.iloc[-1]
    price = float(row["Close"])
    atr = float(row["atr"]) if pd.notna(row["atr"]) else price * 0.01
    rsi = float(row["rsi"]) if pd.notna(row["rsi"]) else 50.0
    trend_up = row["ema_fast"] > row["ema_slow"]
    trend_down = row["ema_fast"] < row["ema_slow"]
    vol_ratio = float(row["Volume"] / row["vol_avg"]) if pd.notna(row.get("vol_avg")) and row.get("vol_avg") else 1.0

    direction = "flat"
    signal = "HALTEN"
    if trend_up and p["rsi_bull"][0] <= rsi <= p["rsi_bull"][1]:
        direction, signal = "long", "KAUFEN (Trend + RSI bullisch)"
    elif trend_down and p["rsi_bear"][0] <= rsi <= p["rsi_bear"][1]:
        direction, signal = "short", "VERKAUFEN (Trend + RSI bärisch)"

    stop = price - atr * p["atr_mult"] if direction == "long" else None
    target = price + atr * p["atr_mult"] * p["reward_risk"] if direction == "long" else None

    return {
        "entry": price, "stop": stop, "target": target, "direction": direction,
        "signal": signal, "rsi": rsi, "volume_ratio": vol_ratio,
    }


def load_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna().tail(1000).reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    return df


# ------------------------------------------------------------
# Supabase: Account laden/speichern (gleiche Tabellen wie die App)
# ------------------------------------------------------------
def get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SECRET_KEY"]
    return create_client(url, key)


def fetch_one(query):
    rows = query.limit(1).execute().data or []
    return rows[0] if rows else None


def load_account(client) -> dict:
    row = fetch_one(client.table("scanner_paper_accounts").select("*").eq("account_key", ACCOUNT_KEY))
    if not row:
        client.table("scanner_paper_accounts").insert({
            "account_key": ACCOUNT_KEY, "cash": STARTING_CASH,
            "equity": STARTING_CASH, "risk_percent": RISK_PERCENT,
        }).execute()
        row = fetch_one(client.table("scanner_paper_accounts").select("*").eq("account_key", ACCOUNT_KEY))
    return {
        "cash": float(row["cash"]), "equity": float(row["equity"]),
        "risk_percent": float(row["risk_percent"]), "position": row.get("position"),
        "last_candle": row.get("last_candle"),
    }


def save_account(client, account: dict, event: dict | None) -> None:
    client.table("scanner_paper_accounts").update({
        "cash": account["cash"], "equity": account["equity"],
        "risk_percent": account["risk_percent"], "position": account["position"],
        "last_candle": account.get("last_candle"),
        "updated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
    }).eq("account_key", ACCOUNT_KEY).execute()
    if event:
        client.table("scanner_paper_orders").insert({
            "account_key": ACCOUNT_KEY,
            "action": "BUY" if event["Aktion"] == "KAUF" else "SELL",
            "ticker": event["Asset"], "price": event["Preis"], "units": event["Menge"],
            "pnl": event["Ergebnis"], "reason": event["Warum"],
            "signal": {"source": "jerry_daytrader"},
        }).execute()


def run_check(df: pd.DataFrame, ticker: str, account: dict) -> tuple[dict, dict | None]:
    setup = calculate_trade_setup(df)
    candle_time = str(df["Date"].iloc[-1])
    price = float(setup["entry"])
    position = account.get("position")
    event = None

    if position and position.get("ticker") == ticker:
        close_reason = None
        if price <= position["stop"]:
            close_reason = "Stop-Loss erreicht"
        elif price >= position["target"]:
            close_reason = "Take-Profit erreicht"
        elif setup["direction"] == "short":
            close_reason = "Trendwechsel: EMA/RSI geben ein Verkaufssignal"
        if close_reason:
            proceeds = position["units"] * price
            account["cash"] += proceeds
            pnl = proceeds - position["cost"]
            event = {
                "Zeit": candle_time, "Aktion": "VERKAUF", "Asset": ticker,
                "Preis": round(price, 4), "Menge": round(position["units"], 6),
                "Ergebnis": round(pnl, 2), "Warum": close_reason,
            }
            account["position"] = None
    elif not position and account.get("last_candle") != candle_time and setup["direction"] == "long" and setup["stop"] is not None:
        risk_per_unit = max(price - float(setup["stop"]), 1e-9)
        risk_budget = account["cash"] * account["risk_percent"] / 100
        units = min(risk_budget / risk_per_unit, account["cash"] / price)
        if units > 0:
            cost = units * price
            account["cash"] -= cost
            account["position"] = {
                "ticker": ticker, "units": units, "cost": cost,
                "stop": float(setup["stop"]), "target": float(setup["target"]),
            }
            event = {
                "Zeit": candle_time, "Aktion": "KAUF", "Asset": ticker,
                "Preis": round(price, 4), "Menge": round(units, 6), "Ergebnis": 0.0,
                "Warum": "Long-Signal: Kurs über EMA 9/21 und RSI im bullischen Bereich",
            }

    account["last_candle"] = candle_time
    position_value = (account["position"] or {}).get("units", 0) * price
    account["equity"] = account["cash"] + position_value
    return account, event


def main():
    weekday = datetime.now(ZoneInfo("UTC")).weekday()  # 0=Mo ... 5=Sa, 6=So
    is_weekend = weekday >= 5
    stocks, crypto = load_universe()
    universe = crypto if is_weekend else stocks
    mode = "Krypto (Wochenende)" if is_weekend else "Aktien (Werktag)"
    print(f"[Jerry] Modus: {mode} · {len(universe)} Assets im Universum")

    client = get_client()
    account = load_account(client)

    active_ticker = (account.get("position") or {}).get("ticker")
    if active_ticker:
        items = [(label, t) for label, t in universe.items() if t == active_ticker]
        if not items:
            # Position gehört zur anderen Asset-Klasse (z.B. Aktie über's Wochenende offen) -> nur beobachten, nicht neu kaufen
            items = [(active_ticker, active_ticker)]
    else:
        items = list(universe.items())[:SCAN_LIMIT]

    candidates = []
    for label, ticker in items:
        try:
            period, interval = ("60d", "1h") if not is_weekend else ("60d", "1h")
            df = load_data(ticker, period=period, interval=interval)
            if df.empty or len(df) < 50:
                continue
            setup = calculate_trade_setup(df)
            vol_score = min(float(setup.get("volume_ratio") or 0), 3.0)
            rsi = float(setup.get("rsi") or 0)
            score = vol_score * 10 + (70 - abs(60 - rsi))
            candidates.append({"label": label, "ticker": ticker, "df": df, "score": score, "signal": setup["signal"]})
        except Exception as exc:
            print(f"  [skip] {ticker}: {exc}")
            continue

    if not candidates:
        print("[Jerry] Keine verwertbaren Marktdaten in diesem Lauf.")
        return

    candidates.sort(key=lambda c: c["score"], reverse=True)
    tradable = [c for c in candidates if c["signal"].startswith("KAUFEN")]
    chosen = tradable[0] if (tradable and not active_ticker) else (
        next((c for c in candidates if c["ticker"] == active_ticker), candidates[0])
    )

    account, event = run_check(chosen["df"], chosen["ticker"], account)
    save_account(client, account, event)

    if event:
        print(f"[Jerry] {event['Aktion']} {event['Asset']} @ {event['Preis']} · {event['Zeit']} · {event['Warum']}")
    else:
        print(f"[Jerry] Kein neuer Trade. Beobachtet: {chosen['ticker']} ({chosen['signal']})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Jerry] FEHLER: {exc}", file=sys.stderr)
        sys.exit(1)
