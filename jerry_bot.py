"""
Gemeinsame Engine für Jerry / Jan / Joseph (Paper-Trading, kein echtes Geld).

Wichtigster Unterschied zur alten Version (jerry_bot.py v1):
- Es wird NIE mehr das gesamte Kapital in einen einzigen Trade gesteckt.
  Jede Position ist auf `max_position_pct` des aktuellen Kontowerts gedeckelt,
  zusätzlich zur bisherigen ATR-Risiko-Berechnung -- es gilt jeweils das
  strengere (kleinere) der beiden Limits.
- Es können mehrere Positionen GLEICHZEITIG offen sein (bis zu `max_positions`),
  wodurch das Kapital automatisch über mehrere Assets diversifiziert wird,
  statt "all-in" auf ein einzelnes Signal zu gehen.
- Ein fester Cash-Puffer (`min_cash_reserve_pct`) bleibt immer unangetastet,
  damit auch bei mehreren offenen Positionen nie 100 % des Kapitals gebunden ist.

Bug-Fixes gegenüber der alten Version:
- Die alte `risk_budget`-Formel konnte theoretisch nahezu 100 % des Cash für
  EINEN Trade verplanen, wenn der Stop sehr eng am Kurs lag (kleiner
  `risk_per_unit`). Jetzt gibt es zusätzlich das harte `max_position_pct`-Limit.
- Rundungsfehler bei sehr kleinen `units` (z.B. Krypto mit hohem Preis) werden
  abgefangen, statt eine Position mit `units == 0` oder negativem Cash zu
  erzeugen.
- `last_candle` wurde pro Account, nicht pro Ticker gespeichert -- das
  verhinderte fälschlich neue Käufe bei anderen Assets im selben Lauf.
  Jetzt wird `last_candle` pro Ticker in jeder Position/im Order-Log geführt.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client

DEFAULT_PARAMS = {
    "ema_fast": 9, "ema_slow": 21, "rsi_period": 14, "atr_period": 14,
    "atr_mult": 1.5, "reward_risk": 2.0, "rsi_bull": (50, 70), "rsi_bear": (30, 50),
}


@dataclass
class BotConfig:
    account_key: str          # eindeutiger Schlüssel in Supabase, z.B. "jerry_bot_v1"
    name: str                 # Anzeigename, z.B. "Jerry"
    starting_cash: float = 10_000.0
    risk_percent: float = 1.0        # % des Kontowerts, das je Trade riskiert wird (Stop-Distanz)
    max_position_pct: float = 0.20   # harte Obergrenze: max. Anteil des Kontowerts je Einzelposition
    max_positions: int = 4           # wie viele Positionen gleichzeitig offen sein dürfen
    min_cash_reserve_pct: float = 0.10  # dieser Anteil des Kontowerts bleibt immer als Cash-Puffer
    interval: str = "1d"             # Kerzen-Intervall für die Datenabfrage
    period: str = "6mo"              # wie viel Historie geladen wird
    scan_limit: int = 80             # wie viele Assets pro Lauf geprüft werden

    # --- Hebel & selbstbestimmter Cash-out ---
    max_leverage: float = 1.0        # Obergrenze, die der Bot selbst wählen darf (1.0 = kein Hebel)
    trailing_activate_r: float = 1.0 # ab diesem Vielfachen des ursprünglichen Risikos ("R") im Plus
                                      # beginnt der Bot, den Stop nachzuziehen (Gewinn sichern)
    trailing_distance_r: float = 0.5 # wie eng der nachgezogene Stop hinter dem besten Kurs bleibt (in "R")
    max_hold_cycles: int | None = None  # nach so vielen Läufen wird notfalls zum Marktpreis "cashed out",
                                         # auch wenn weder Stop noch Ziel erreicht wurde (None = kein Limit)


# ------------------------------------------------------------
# Universum aus app.py extrahieren (kein Import von app.py selbst,
# da app.py Streamlit-UI-Code enthält, der außerhalb von `streamlit run` crasht)
# ------------------------------------------------------------
def load_universe(app_py_path: str = "app.py") -> tuple[dict, dict]:
    with open(app_py_path, "r", encoding="utf-8") as f:
        text = f.read()
    pairs = re.findall(r'SCANNER_UNIVERSE\["([^"]+)"\]\s*=\s*"([^"]+)"', text)
    if not pairs:
        raise RuntimeError(
            f"Konnte kein SCANNER_UNIVERSE in {app_py_path} finden -- Datei umbenannt oder Format geändert?"
        )
    crypto = {label: ticker for label, ticker in pairs if ticker.endswith("-USD")}
    stocks = {label: ticker for label, ticker in pairs if not ticker.endswith("-USD")}
    return stocks, crypto


# ------------------------------------------------------------
# Indikatoren & Setup-Logik
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
# Supabase: Account laden/speichern
# ------------------------------------------------------------
def get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SECRET_KEY"]
    return create_client(url, key)


def fetch_one(query):
    rows = query.limit(1).execute().data or []
    return rows[0] if rows else None


def load_account(client, cfg: BotConfig) -> dict:
    row = fetch_one(client.table("scanner_paper_accounts").select("*").eq("account_key", cfg.account_key))
    if not row:
        client.table("scanner_paper_accounts").insert({
            "account_key": cfg.account_key, "cash": cfg.starting_cash,
            "equity": cfg.starting_cash, "risk_percent": cfg.risk_percent,
            "position": [],
        }).execute()
        row = fetch_one(client.table("scanner_paper_accounts").select("*").eq("account_key", cfg.account_key))

    positions_raw = row.get("position")
    if isinstance(positions_raw, dict):  # Altbestand: einzelne Position -> Liste
        positions = [positions_raw] if positions_raw else []
    elif isinstance(positions_raw, list):
        positions = positions_raw
    else:
        positions = []

    raw_seen = row.get("last_candle")
    if isinstance(raw_seen, str):
        # Alt-Bug: last_candle wurde als Text-String (z.B. "{}") statt als
        # echtes JSON-Objekt gespeichert -- robust auflösen statt crashen.
        import json as _json
        try:
            raw_seen = _json.loads(raw_seen)
        except (ValueError, TypeError):
            raw_seen = {}
    seen_candles = raw_seen if isinstance(raw_seen, dict) else {}

    return {
        "cash": float(row["cash"]), "equity": float(row["equity"]),
        "risk_percent": float(row.get("risk_percent") or cfg.risk_percent),
        "positions": positions,
        "seen_candles": seen_candles,  # {ticker: candle_time}
    }


def save_account(client, cfg: BotConfig, account: dict, events: list[dict]) -> None:
    seen_candles = account.get("seen_candles") or {}
    # Abwärtskompatibel: last_candle könnte in der DB noch als reiner String
    # aus der alten Single-Position-Version vorliegen -- dann verwerfen wir ihn.
    if not isinstance(seen_candles, dict):
        seen_candles = {}
    client.table("scanner_paper_accounts").update({
        "cash": account["cash"], "equity": account["equity"],
        "risk_percent": account["risk_percent"], "position": account["positions"],
        "last_candle": seen_candles,
        "updated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
    }).eq("account_key", cfg.account_key).execute()
    for event in events:
        client.table("scanner_paper_orders").insert({
            "account_key": cfg.account_key,
            "action": "BUY" if event["Aktion"] == "KAUF" else "SELL",
            "ticker": event["Asset"], "price": event["Preis"], "units": event["Menge"],
            "pnl": event["Ergebnis"], "reason": event["Warum"],
            "signal": {"source": cfg.account_key},
        }).execute()


def _mark_to_market_equity(account: dict, price_by_ticker: dict[str, float]) -> float:
    """Bei gehebelten Positionen gehört dem Bot nicht der volle Positionswert,
    sondern nur die hinterlegte Margin + der bisherige (unrealisierte)
    Gewinn/Verlust der Position."""
    equity = account["cash"]
    for pos in account["positions"]:
        price = price_by_ticker.get(pos["ticker"], pos.get("entry", 0))
        margin = pos.get("margin", pos.get("cost", 0))
        unrealized_pnl = pos["units"] * (price - pos["entry"])
        equity += margin + unrealized_pnl
    return equity


def _choose_leverage(cfg: BotConfig, setup: dict, score: float) -> float:
    """Der Bot bestimmt den Hebel selbst -- je stärker/sauberer das Signal
    (hohes Volumen, RSI nahe am Sweet-Spot), desto mehr von seinem
    persönlichen Hebel-Limit (max_leverage) schöpft er aus. Ein schwaches
    Signal wird nicht gehebelt (Hebel = 1x)."""
    if cfg.max_leverage <= 1.0:
        return 1.0
    # score liegt typischerweise zwischen ca. 0 und 100 (siehe scan_candidates)
    confidence = max(0.0, min(score / 85.0, 1.0))
    leverage = 1.0 + confidence * (cfg.max_leverage - 1.0)
    return round(leverage, 2)


def _try_close_position(cfg: BotConfig, account: dict, pos: dict, ticker: str, price: float, direction: str, candle_time: str) -> dict | None:
    """Prüft, ob der Bot diese Position schließt. Vier mögliche Gründe:
    1) fester Stop-Loss erreicht, 2) festes Take-Profit-Ziel erreicht,
    3) nachgezogener Trailing-Stop ausgelöst (der Bot sichert selbstständig
       Gewinn, sobald er im Plus liegt -- "cash out, wann er will"),
    4) Trendwechsel (EMA/RSI drehen) -- der Bot verlässt die Position auch
       dann, wenn weder Stop noch Ziel erreicht wurde, weil die Logik hinter
       dem Trade nicht mehr gilt,
    5) maximale Haltedauer überschritten (nur falls cfg das vorsieht) --
       verhindert, dass eine Position endlos offen bleibt."""
    pos["best_price"] = max(pos.get("best_price", pos["entry"]), price)
    pos["runs_held"] = pos.get("runs_held", 0) + 1

    risk = max(pos["entry"] - pos["stop"], 1e-9)  # "1R" in Preis-Einheiten
    # Wichtig: die Aktivierung hängt am BESTEN je erreichten Kurs, nicht am
    # aktuellen -- sonst würde sich der Gewinnschutz bei jedem Rücksetzer
    # wieder deaktivieren, obwohl der Trade zwischenzeitlich im Plus war.
    best_r = (pos["best_price"] - pos["entry"]) / risk

    effective_stop = pos["stop"]
    if best_r >= cfg.trailing_activate_r:
        trailing_stop = pos["best_price"] - cfg.trailing_distance_r * risk
        effective_stop = max(effective_stop, trailing_stop)

    close_reason = None
    if price <= effective_stop:
        close_reason = (
            "Trailing-Stop ausgelöst -- Gewinn selbst gesichert"
            if effective_stop > pos["stop"] else "Stop-Loss erreicht"
        )
    elif price >= pos["target"]:
        close_reason = "Take-Profit erreicht"
    elif direction != "long":
        close_reason = "Trendwechsel: EMA/RSI drehen -- Position selbst aufgelöst"
    elif cfg.max_hold_cycles and pos["runs_held"] >= cfg.max_hold_cycles:
        close_reason = "Maximale Haltedauer erreicht -- Position vorsorglich ausgecasht"
    if not close_reason:
        return None

    margin = pos.get("margin", pos.get("cost", 0))
    pnl = pos["units"] * (price - pos["entry"])  # gehebelter Gewinn/Verlust in $
    account["cash"] += margin + pnl
    account["positions"] = [p for p in account["positions"] if p is not pos]
    return {
        "Zeit": candle_time, "Aktion": "VERKAUF", "Asset": ticker,
        "Preis": round(price, 4), "Menge": round(pos["units"], 6),
        "Ergebnis": round(pnl, 2), "Warum": close_reason,
    }


def _try_open_position(cfg: BotConfig, account: dict, ticker: str, setup: dict, score: float, candle_time: str) -> dict | None:
    """Öffnet eine neue, diversifizierte Position -- die als MARGIN (Cash,
    das wirklich abgezogen wird) niemals mehr als max_position_pct des
    Kontowerts und niemals mehr, als der Cash-Puffer zulässt. Das verhindert,
    dass der Bot sein gesamtes Geld in einen einzelnen Trade steckt.
    Der Bot darf diese Margin zusätzlich selbst hebeln (bis max_leverage) --
    das vergrößert die Positionsgröße (und damit Gewinn-/Verlust-Chance),
    OHNE den gebundenen Cash-Einsatz zu erhöhen."""
    if len(account["positions"]) >= cfg.max_positions:
        return None
    if any(p["ticker"] == ticker for p in account["positions"]):
        return None  # schon investiert -> nicht nachkaufen, lieber diversifizieren

    price = float(setup["entry"])
    stop = setup["stop"]
    if stop is None or price <= 0:
        return None

    equity = account["equity"]
    reserve = equity * cfg.min_cash_reserve_pct
    available_cash = max(account["cash"] - reserve, 0.0)
    if available_cash <= 0:
        return None

    risk_per_unit = max(price - float(stop), 1e-9)
    risk_budget = equity * account["risk_percent"] / 100
    max_position_value = equity * cfg.max_position_pct

    # Drei unabhängige Obergrenzen für die MARGIN (der tatsächliche Cash-Einsatz),
    # es gilt die strengste (kleinste):
    # 1) wie viel laut Risiko-Budget (Stop-Distanz) investiert werden darf
    # 2) die harte Positionsgrößen-Grenze (Diversifikations-Schutz)
    # 3) wie viel Cash nach Abzug des Reserve-Puffers überhaupt noch frei ist
    margin = min(risk_budget / risk_per_unit * price, max_position_value, available_cash)
    if margin <= 0 or margin < 1.0:
        return None

    leverage = _choose_leverage(cfg, setup, score)
    position_value = margin * leverage  # Hebel vergrößert die Positionsgröße, nicht die Margin
    units = position_value / price
    if units <= 0:
        return None

    account["cash"] -= margin
    account["positions"].append({
        "ticker": ticker, "units": units, "cost": position_value, "margin": margin,
        "leverage": leverage, "entry": price, "best_price": price, "runs_held": 0,
        "stop": float(stop), "target": float(setup["target"]),
    })
    hebel_hinweis = f" mit {leverage:.1f}x Hebel" if leverage > 1.0 else " ohne Hebel"
    return {
        "Zeit": candle_time, "Aktion": "KAUF", "Asset": ticker,
        "Preis": round(price, 4), "Menge": round(units, 6), "Ergebnis": 0.0,
        "Warum": f"Long-Signal: Kurs über EMA {DEFAULT_PARAMS['ema_fast']}/{DEFAULT_PARAMS['ema_slow']} "
                 f"und RSI im bullischen Bereich ({margin / equity * 100:.1f} % Margin des Kontowerts"
                 f"{hebel_hinweis})",
    }


def run_cycle(cfg: BotConfig, candidates: list[dict], account: dict) -> tuple[dict, list[dict]]:
    """Ein Durchlauf über alle gescannten Kandidaten: zuerst bestehende
    Positionen auf Exit prüfen, danach -- solange Platz und Kapital übrig ist
    -- neue, diversifizierte Positionen unter den besten Kauf-Signalen
    eröffnen (bis max_positions erreicht ist)."""
    events: list[dict] = []
    price_by_ticker = {c["ticker"]: float(c["setup"]["entry"]) for c in candidates}

    # 1) Exits zuerst -- bestehende Positionen bei JEDEM Lauf mit aktuellen
    #    Kursen prüfen (nicht nur bei neuer Kerze), damit Trailing-Stop und
    #    Haltedauer-Zählung den Bot wirklich jederzeit selbst entscheiden
    #    lassen, auszucashen -- nicht erst, wenn eine neue Kerze entsteht.
    for pos in list(account["positions"]):
        match = next((c for c in candidates if c["ticker"] == pos["ticker"]), None)
        if not match:
            continue
        candle_time = match["candle_time"]
        event = _try_close_position(
            cfg, account, pos, pos["ticker"], price_by_ticker[pos["ticker"]],
            match["setup"]["direction"], candle_time,
        )
        account["seen_candles"][pos["ticker"]] = candle_time
        if event:
            events.append(event)

    # 2) Neue Einstiege -- nur klare KAUFEN-Signale, beste Kandidaten zuerst,
    #    bis das Positionslimit erreicht ist. So verteilt sich das Kapital
    #    automatisch über mehrere Assets statt "alles auf eine Karte". Der
    #    Hebel wird je Trade vom Bot selbst anhand der Signalstärke gewählt.
    buy_candidates = [c for c in candidates if c["setup"]["signal"].startswith("KAUFEN")]
    buy_candidates.sort(key=lambda c: c["score"], reverse=True)
    for cand in buy_candidates:
        if len(account["positions"]) >= cfg.max_positions:
            break
        ticker, candle_time = cand["ticker"], cand["candle_time"]
        if account["seen_candles"].get(ticker) == candle_time:
            continue  # diese Kerze für dieses Asset wurde schon verarbeitet -> nicht doppelt kaufen
        event = _try_open_position(cfg, account, ticker, cand["setup"], cand["score"], candle_time)
        account["seen_candles"][ticker] = candle_time
        if event:
            events.append(event)

    account["equity"] = _mark_to_market_equity(account, price_by_ticker)
    return account, events


def scan_candidates(cfg: BotConfig, universe: dict[str, str]) -> list[dict]:
    items = list(universe.items())[:cfg.scan_limit]
    candidates = []
    for label, ticker in items:
        try:
            df = load_data(ticker, period=cfg.period, interval=cfg.interval)
            if df.empty or len(df) < 50:
                continue
            setup = calculate_trade_setup(df)
            vol_score = min(float(setup.get("volume_ratio") or 0), 3.0)
            rsi = float(setup.get("rsi") or 0)
            score = vol_score * 10 + (70 - abs(60 - rsi))
            candidates.append({
                "label": label, "ticker": ticker, "setup": setup, "score": score,
                "candle_time": str(df["Date"].iloc[-1]),
            })
        except Exception as exc:
            print(f"  [skip] {ticker}: {exc}")
            continue
    return candidates


def main(cfg: BotConfig, app_py_path: str = "app.py") -> None:
    weekday = datetime.now(ZoneInfo("UTC")).weekday()  # 0=Mo ... 5=Sa, 6=So
    is_weekend = weekday >= 5
    stocks, crypto = load_universe(app_py_path)
    universe = crypto if is_weekend else stocks
    mode = "Krypto (Wochenende)" if is_weekend else "Aktien (Werktag)"
    print(f"[{cfg.name}] Modus: {mode} · {len(universe)} Assets im Universum")

    client = get_client()
    account = load_account(client, cfg)

    candidates = scan_candidates(cfg, universe)
    if not candidates:
        print(f"[{cfg.name}] Keine verwertbaren Marktdaten in diesem Lauf.")
        return

    account, events = run_cycle(cfg, candidates, account)
    save_account(client, cfg, account, events)

    if events:
        for event in events:
            print(f"[{cfg.name}] {event['Aktion']} {event['Asset']} @ {event['Preis']} · {event['Zeit']} · {event['Warum']}")
    else:
        open_tickers = ", ".join(p["ticker"] for p in account["positions"]) or "keine"
        print(f"[{cfg.name}] Kein neuer Trade. Offene Positionen: {open_tickers}")


def run_bot(cfg: BotConfig) -> None:
    try:
        main(cfg)
    except Exception as exc:
        print(f"[{cfg.name}] FEHLER: {exc}", file=sys.stderr)
        sys.exit(1)
