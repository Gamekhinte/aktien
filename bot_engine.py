"""
Gemeinsame Engine für Jerry / Jan / Joseph (Paper-Trading, kein echtes Geld).

Jeder Bot kauft UND verkauft komplett selbstständig nach seinen eigenen
Regeln (siehe jerry_bot.py / jan_bot.py / joseph_bot.py). Pro Lauf:

1. Universum bauen: offene Positionen + Wichtig-Watchlist (IMMER) + ein
   rotierendes Fenster über den Rest des Universums -- dadurch sieht jeder
   Bot nach ein paar Läufen ALLE Assets, nicht nur die ersten 60-80 der Liste.
2. Kursdaten aller Assets in wenigen Batch-Downloads laden (parallel, statt
   einzeln nacheinander -> deutlich schneller).
3. Offene Positionen verwalten: Stop-Loss / Take-Profit werden über die
   Hochs/Tiefs ALLER Kerzen seit dem letzten Lauf geprüft (nicht nur über den
   aktuellen Schlusskurs), Break-even-Stop, Trailing-Stop, Trendwechsel,
   maximale Haltedauer.
4. Neue Einstiege -- Long UND (wenn erlaubt) Short. Ein Signal braucht:
   EMA 9/21-Trend + übergeordneter Trend (EMA 50) + RSI im passenden Bereich
   + MACD-Bestätigung + ADX (Trendstärke). Zusätzlich wird das Setup auf der
   Historie DIESES Assets nachgetestet ("Edge"): nur wenn es dort
   historisch positiv war, gibt es einen hohen Score.
5. Lernen aus eigenen Trades: Gewinnquote je Asset fließt in den Score ein,
   nach einem Verlust gibt es eine Abkühlphase für dieses Asset, und bei
   einem Drawdown halbiert der Bot automatisch sein Risiko.

Risiko-Grundsätze (unverändert): nie alles in einen Trade, harte
Positionsgrenze je Asset, fester Cash-Puffer, Hebel erhöht NICHT das
Risiko pro Trade (der Stop-Abstand bestimmt das Risiko), und eine gehebelte
Position kann nie mehr als ihre Margin verlieren (Knock-out-Logik).

Lokal testen ohne Supabase:  python jerry_bot.py --dry-run
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from watchlist_wichtig import load_wichtig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_KEY = "__memory__"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "90m": 90, "1d": 1440, "1wk": 10080}


@dataclass
class BotConfig:
    account_key: str          # eindeutiger Schlüssel in Supabase, z.B. "jerry_bot_v1"
    name: str                 # Anzeigename, z.B. "Jerry"
    starting_cash: float = 10_000.0
    risk_percent: float = 1.0        # % des Kontowerts, das je Trade riskiert wird (Stop-Distanz)
    max_position_pct: float = 0.20   # harte Obergrenze: max. Margin-Anteil des Kontowerts je Einzelposition
    max_positions: int = 4           # wie viele Positionen gleichzeitig offen sein dürfen
    min_cash_reserve_pct: float = 0.10  # dieser Anteil des Kontowerts bleibt immer als Cash-Puffer
    interval: str = "1d"             # Kerzen-Intervall für die Datenabfrage
    period: str = "1y"               # wie viel Historie geladen wird
    scan_limit: int = 80             # Größe des rotierenden Fensters (zusätzlich zu Wichtig + offenen Positionen)

    # --- Hebel & selbstbestimmter Cash-out ---
    max_leverage: float = 1.0        # Obergrenze, die der Bot selbst wählen darf (1.0 = kein Hebel)
    trailing_activate_r: float = 1.0 # ab so viel "R" im Plus wird der Stop nachgezogen
    trailing_distance_r: float = 0.5 # Abstand des Trailing-Stops zum besten Kurs (in "R")
    breakeven_r: float = 1.0         # ab so viel "R" im Plus wandert der Stop auf den Einstieg
    max_hold_cycles: int | None = None  # max. Haltedauer in KERZEN des Bot-Intervalls (None = kein Limit)
    exit_on_ema_cross: bool = True   # raus, sobald der Kurs klar auf der falschen Seite der EMA 21 schließt

    # --- Signal-Qualität ---
    allow_short: bool = True         # darf der Bot auch auf fallende Kurse setzen?
    min_score: float = 55.0          # Mindest-Score (0-100) für einen Einstieg
    min_adx: float = 18.0            # Mindest-Trendstärke
    min_edge: float = -0.05          # Mindest-Erwartungswert (in R) aus dem Asset-eigenen Backtest
    atr_mult: float = 1.5            # Stop-Abstand = atr_mult * ATR
    reward_risk: float = 2.0         # Ziel = reward_risk * Stop-Abstand
    trend_ema: int = 50              # übergeordneter Trendfilter
    edge_horizon: int = 10           # so viele Kerzen hat ein historischer Test-Trade Zeit

    # --- Ausführung / Lernen ---
    fee_per_trade: float = 1.0       # Ordergebühr je Kauf/Verkauf (Trade Republic: 1 €)
    max_new_trades_per_run: int = 3  # nicht alles in einem einzigen Lauf kaufen
    cooldown_runs: int = 4           # nach einem Verlust so viele Läufe Pause für dieses Asset
    max_drawdown_pct: float = 12.0   # ab diesem Drawdown vom Höchststand: halbes Risiko, kein Hebel
    watch_alert_pct: float = 3.0     # Wichtig-Assets mit Tagesbewegung >= x % werden hervorgehoben


# ------------------------------------------------------------
# Universum
# ------------------------------------------------------------
def load_universe(app_py_path: str | None = None) -> tuple[dict, dict]:
    """Liest SCANNER_UNIVERSE aus app.py (ohne app.py zu importieren -- das
    würde außerhalb von `streamlit run` crashen). Doppelte Ticker werden entfernt."""
    path = app_py_path or os.path.join(BASE_DIR, "app.py")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    pairs = re.findall(r'SCANNER_UNIVERSE\["([^"]+)"\]\s*=\s*"([^"]+)"', text)
    if not pairs:
        raise RuntimeError(f"Konnte kein SCANNER_UNIVERSE in {path} finden -- Datei umbenannt oder Format geändert?")
    seen: set[str] = set()
    crypto: dict[str, str] = {}
    stocks: dict[str, str] = {}
    for label, ticker in pairs:
        if ticker in seen:
            continue
        seen.add(ticker)
        (crypto if ticker.endswith("-USD") else stocks)[label] = ticker
    return stocks, crypto


def build_universe(cfg: BotConfig, base: dict[str, str], wichtig: dict[str, str],
                   held: list[str], memory: dict, rotation_key: str) -> dict[str, tuple[str, bool]]:
    """Reihenfolge: offene Positionen -> Wichtig -> rotierendes Fenster.
    Rückgabe: {ticker: (label, ist_wichtig)}"""
    wichtig_tickers = set(wichtig.values())
    meta: dict[str, tuple[str, bool]] = {}
    label_by_ticker = {t: l for l, t in base.items()}
    label_by_ticker.update({t: l for l, t in wichtig.items()})

    for t in held:
        meta[t] = (label_by_ticker.get(t, t), t in wichtig_tickers)
    for label, t in wichtig.items():
        meta.setdefault(t, (label, True))

    rest = [(l, t) for l, t in base.items() if t not in meta]
    if rest:
        rotation = memory.setdefault("rotation", {})
        offset = int(rotation.get(rotation_key, 0)) % len(rest)
        take = min(cfg.scan_limit, len(rest))
        window = (rest[offset:] + rest[:offset])[:take]
        rotation[rotation_key] = (offset + take) % len(rest)
        for l, t in window:
            meta.setdefault(t, (l, False))
    return meta


# ------------------------------------------------------------
# Daten (Batch-Download)
# ------------------------------------------------------------
def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "Date" in df.columns:
        df = df.set_index("Date")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        return pd.DataFrame()
    df = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]
    df["Volume"] = df["Volume"].fillna(0.0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.tail(1000)


def _extract(raw: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    level0 = raw.columns.get_level_values(0)
    if ticker in level0:
        return raw[ticker]
    if single:
        sub = raw.copy()
        sub.columns = [next((x for x in col if x in OHLCV), col[0]) for col in sub.columns]
        return sub
    return pd.DataFrame()


def download_many(tickers: list[str], period: str, interval: str, chunk: int = 80) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    unique = list(dict.fromkeys(tickers))
    for start in range(0, len(unique), chunk):
        part = unique[start:start + chunk]
        try:
            raw = yf.download(
                part, period=period, interval=interval, group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as exc:
            print(f"  [download] Fehler bei {len(part)} Tickern: {exc}")
            continue
        if raw is None or raw.empty:
            continue
        for ticker in part:
            try:
                df = _normalize_frame(_extract(raw, ticker, single=len(part) == 1).dropna(how="all"))
                if len(df) >= 60:
                    result[ticker] = df
            except Exception:
                continue
    return result


# ------------------------------------------------------------
# Indikatoren, Signale, Backtest, Score
# ------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    d = _normalize_frame(df)
    if d.empty:
        return d
    close, high, low = d["Close"], d["High"], d["Low"]
    d["ema_f"] = close.ewm(span=9, adjust=False).mean()
    d["ema_s"] = close.ewm(span=21, adjust=False).mean()
    d["ema_t"] = close.ewm(span=cfg.trend_ema, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    d["macd_h"] = macd - macd.ewm(span=9, adjust=False).mean()

    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=d.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=d.index)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    vol = d["Volume"].replace(0, np.nan)
    d["vol_ratio"] = vol / vol.rolling(20).mean()
    # Warm-up: die ersten Kerzen haben noch keine sinnvollen Indikatoren
    d.loc[d.index[:30], ["adx", "atr"]] = np.nan
    return d


def signal_masks(d: pd.DataFrame, cfg: BotConfig) -> tuple[pd.Series, pd.Series]:
    c = d["Close"]
    strong = d["adx"] >= cfg.min_adx
    long_m = (
        (c > d["ema_f"]) & (d["ema_f"] > d["ema_s"]) & (c > d["ema_t"])
        & d["rsi"].between(50, 70) & (d["macd_h"] > 0) & strong
    )
    short_m = (
        (c < d["ema_f"]) & (d["ema_f"] < d["ema_s"]) & (c < d["ema_t"])
        & d["rsi"].between(30, 50) & (d["macd_h"] < 0) & strong
    )
    return long_m.fillna(False), short_m.fillna(False)


def backtest_edge(d: pd.DataFrame, mask: pd.Series, side: int, cfg: BotConfig) -> dict:
    """Testet exakt dieses Setup auf der Historie des Assets (ohne die aktuelle
    Kerze, ohne überlappende Trades). Ergebnis in R (1R = Stop-Abstand)."""
    close, high, low, atr = (d[c].to_numpy(dtype=float) for c in ("Close", "High", "Low", "atr"))
    n, horizon = len(d), cfg.edge_horizon
    total_r, wins, trades, next_free = 0.0, 0, 0, 0
    for i in np.flatnonzero(mask.to_numpy()[: n - 1]):
        if i < next_free:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        risk = a * cfg.atr_mult
        entry = close[i]
        stop = entry - side * risk
        target = entry + side * risk * cfg.reward_risk
        end = min(i + horizon, n - 1)
        r, exit_idx = None, end
        for j in range(i + 1, end + 1):
            hit_stop = low[j] <= stop if side == 1 else high[j] >= stop
            hit_target = high[j] >= target if side == 1 else low[j] <= target
            if hit_stop:  # konservativ: Stop zuerst, falls beides in einer Kerze
                r, exit_idx = -1.0, j
                break
            if hit_target:
                r, exit_idx = cfg.reward_risk, j
                break
        if r is None:
            if i + horizon > n - 1:
                continue  # Testfenster noch nicht vollständig
            r = side * (close[end] - entry) / risk
        total_r += r
        trades += 1
        wins += int(r > 0)
        next_free = exit_idx + 1
    return {
        "trades": trades,
        "win_rate": wins / trades if trades else None,
        "exp_r": total_r / trades if trades else 0.0,
        # Bayes-artig zur 0 geschrumpft: 2 Glückstreffer sind noch keine Edge
        "exp_shrunk": total_r / (trades + 5),
    }


def _day_change(d: pd.DataFrame) -> float | None:
    days = d.index.normalize()
    prev = d["Close"][days < days[-1]]
    if prev.empty:
        return None
    return float(d["Close"].iloc[-1] / prev.iloc[-1] - 1) * 100


def analyze(df: pd.DataFrame, cfg: BotConfig | None = None, ticker: str = "", label: str = "",
            wichtig: bool = False, memory: dict | None = None) -> dict | None:
    """Bewertet ein Asset. Wird von den Bots UND von der Wichtig-Ansicht in app.py genutzt."""
    cfg = cfg or BotConfig(account_key="view", name="view")
    d = compute_indicators(df, cfg)
    if len(d) < 60:
        return None
    last = d.iloc[-1]
    price, atr = float(last["Close"]), float(last["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    long_m, short_m = signal_masks(d, cfg)
    direction = "long" if long_m.iloc[-1] else "short" if short_m.iloc[-1] else "flat"
    rsi = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
    adx = float(last["adx"]) if pd.notna(last["adx"]) else 0.0
    vol_ratio = float(last["vol_ratio"]) if pd.notna(last["vol_ratio"]) else 1.0
    out = {
        "ticker": ticker, "label": label or ticker, "wichtig": wichtig,
        "price": price, "atr": atr, "rsi": rsi, "adx": adx, "volume_ratio": vol_ratio,
        "day_change": _day_change(d), "candle_time": d.index[-1].isoformat(),
        "direction": direction, "signal": "ABWARTEN", "score": 0.0,
        "stop": None, "target": None, "edge": None, "tradable": False,
        "ema_s": float(last["ema_s"]), "_frame": d,
    }
    if direction == "flat":
        return out

    side = 1 if direction == "long" else -1
    mask = long_m if side == 1 else short_m
    edge = backtest_edge(d, mask, side, cfg)
    fresh = not bool(mask.iloc[-4:-1].all())  # Signal erst in den letzten 3 Kerzen entstanden
    extension = side * (price - float(last["ema_s"])) / atr

    score = 0.0
    score += min(adx, 50.0) / 50.0 * 25                                   # Trendstärke
    score += float(np.clip(side * float(last["macd_h"]) / atr / 0.5, 0, 1)) * 15  # Momentum
    score += max(0.0, 1 - abs(rsi - (60 if side == 1 else 40)) / 10) * 15  # RSI-Sweet-Spot
    score += float(np.clip(vol_ratio, 0, 3)) / 3 * 10                     # Volumen
    score += float(np.clip((edge["exp_shrunk"] + 0.5) / 1.5, 0, 1)) * 25  # eigene historische Edge
    score += 5 if fresh else 0
    score += 5 if wichtig else 0
    if extension > 3:
        score -= 15  # zu weit gelaufen -> nicht hinterherhecheln
    elif extension > 2:
        score -= 5
    stats = ((memory or {}).get("ticker_stats") or {}).get(ticker)
    if stats and stats.get("trades", 0) >= 3:
        score += float(np.clip((stats["wins"] / stats["trades"] - 0.5) * 20, -10, 10))

    stop_dist = atr * cfg.atr_mult
    out.update({
        "signal": "KAUFEN (Long-Setup)" if side == 1 else "VERKAUFEN (Short-Setup)",
        "score": round(max(score, 0.0), 1),
        "stop": price - side * stop_dist,
        "target": price + side * stop_dist * cfg.reward_risk,
        "edge": edge, "fresh": fresh, "extension": extension,
        "tradable": side == 1 or cfg.allow_short,
    })
    return out


# ------------------------------------------------------------
# Supabase: Account laden/speichern
# ------------------------------------------------------------
def get_client():
    from supabase import create_client  # lazy: app.py importiert dieses Modul auch ohne Bot-Lauf
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])


def fetch_one(query):
    rows = query.limit(1).execute().data or []
    return rows[0] if rows else None


def _parse_account_row(row: dict, cfg: BotConfig) -> dict:
    positions_raw = row.get("position")
    if isinstance(positions_raw, dict):  # Altbestand: einzelne Position / {"positions": [...]}
        positions = positions_raw.get("positions") if "positions" in positions_raw else [positions_raw]
    elif isinstance(positions_raw, list):
        positions = positions_raw
    else:
        positions = []
    raw_seen = row.get("last_candle")
    if isinstance(raw_seen, str):
        try:
            raw_seen = json.loads(raw_seen)
        except (ValueError, TypeError):
            raw_seen = {}
    seen = dict(raw_seen) if isinstance(raw_seen, dict) else {}
    memory = seen.pop(MEMORY_KEY, None)
    return {
        "cash": float(row["cash"]), "equity": float(row["equity"]),
        "risk_percent": float(row.get("risk_percent") or cfg.risk_percent),
        "positions": [p for p in (positions or []) if isinstance(p, dict) and p.get("ticker")],
        "seen_candles": seen,
        "memory": memory if isinstance(memory, dict) else {},
    }


def load_account(client, cfg: BotConfig) -> dict:
    row = fetch_one(client.table("scanner_paper_accounts").select("*").eq("account_key", cfg.account_key))
    if not row:
        client.table("scanner_paper_accounts").insert({
            "account_key": cfg.account_key, "cash": cfg.starting_cash,
            "equity": cfg.starting_cash, "risk_percent": cfg.risk_percent, "position": [],
        }).execute()
        row = fetch_one(client.table("scanner_paper_accounts").select("*").eq("account_key", cfg.account_key))
    return _parse_account_row(row, cfg)


def _orders_payload(cfg: BotConfig, events: list[dict]) -> list[dict]:
    return [{
        "account_key": cfg.account_key,
        "action": "BUY" if event["Aktion"] == "KAUF" else "SELL",
        "ticker": event["Asset"], "price": event["Preis"], "units": event["Menge"],
        "pnl": event["Ergebnis"], "reason": event["Warum"],
        "signal": {"source": cfg.account_key, **event.get("_signal", {})},
    } for event in events]


def _account_payload(account: dict) -> dict:
    seen = {**(account.get("seen_candles") or {}), MEMORY_KEY: account.get("memory") or {}}
    return {
        "cash": max(account["cash"], 0.0), "equity": max(account["equity"], 0.0),
        "risk_percent": account["risk_percent"], "position": account["positions"],
        "last_candle": json.dumps(seen, separators=(",", ":"), default=str),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def save_account(client, cfg: BotConfig, account: dict, events: list[dict]) -> None:
    client.table("scanner_paper_accounts").update(_account_payload(account)).eq("account_key", cfg.account_key).execute()
    payload = _orders_payload(cfg, events)
    if payload:
        client.table("scanner_paper_orders").insert(payload).execute()  # ein Request statt einer pro Order


def _dry_run_path(cfg: BotConfig) -> str:
    return os.path.join(BASE_DIR, f".dryrun_{cfg.account_key}.json")


def load_account_dry(cfg: BotConfig) -> dict:
    path = _dry_run_path(cfg)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return _parse_account_row(json.load(f), cfg)
    return _parse_account_row({"cash": cfg.starting_cash, "equity": cfg.starting_cash,
                               "risk_percent": cfg.risk_percent, "position": []}, cfg)


def save_account_dry(cfg: BotConfig, account: dict, events: list[dict]) -> None:
    with open(_dry_run_path(cfg), "w", encoding="utf-8") as f:
        json.dump(_account_payload(account), f, indent=1, default=str)


# ------------------------------------------------------------
# Positionen verwalten
# ------------------------------------------------------------
def _pos_side(pos: dict) -> int:
    return -1 if pos.get("side") == "short" else 1


def _unrealized(pos: dict, price: float) -> float:
    margin = float(pos.get("margin", pos.get("cost", 0)))
    pnl = _pos_side(pos) * pos["units"] * (price - pos["entry"])
    return max(pnl, -margin)  # Knock-out: nie mehr als die Margin verlieren


def _mark_to_market_equity(account: dict, price_by_ticker: dict[str, float]) -> float:
    equity = account["cash"]
    for pos in account["positions"]:
        price = price_by_ticker.get(pos["ticker"], pos["entry"])
        equity += float(pos.get("margin", pos.get("cost", 0))) + _unrealized(pos, price)
    return equity


def _close_position(cfg: BotConfig, account: dict, pos: dict, price: float, reason: str, when: str) -> dict:
    margin = float(pos.get("margin", pos.get("cost", 0)))
    gross = _unrealized(pos, price)
    account["cash"] += margin + gross - cfg.fee_per_trade
    account["positions"] = [p for p in account["positions"] if p is not pos]
    net = gross - cfg.fee_per_trade - float(pos.get("fee_paid", 0.0))

    memory = account["memory"]
    stats = memory.setdefault("ticker_stats", {}).setdefault(pos["ticker"], {"trades": 0, "wins": 0, "pnl": 0.0})
    stats["trades"] += 1
    stats["wins"] += int(net > 0)
    stats["pnl"] = round(stats["pnl"] + net, 2)
    totals = memory.setdefault("totals", {"trades": 0, "wins": 0, "pnl": 0.0})
    totals["trades"] += 1
    totals["wins"] += int(net > 0)
    totals["pnl"] = round(totals["pnl"] + net, 2)
    if net < 0:
        memory.setdefault("cooldown", {})[pos["ticker"]] = memory.get("runs", 0) + cfg.cooldown_runs

    richtung = "Short" if _pos_side(pos) == -1 else "Long"
    return {
        "Zeit": when, "Aktion": "VERKAUF", "Asset": pos["ticker"],
        "Preis": round(price, 4), "Menge": round(pos["units"], 6),
        "Ergebnis": round(net, 2), "Warum": f"{richtung} geschlossen: {reason}",
        "_signal": {"side": richtung.lower(), "leverage": pos.get("leverage", 1.0)},
    }


def _better(side: int, a: float, b: float) -> float:
    """Der für die Position günstigere (engere) Stop."""
    return max(a, b) if side == 1 else min(a, b)


def manage_position(cfg: BotConfig, account: dict, pos: dict, analysis: dict) -> dict | None:
    d: pd.DataFrame = analysis["_frame"]
    side = _pos_side(pos)
    entry = float(pos["entry"])
    initial_stop = float(pos["stop"])
    risk = float(pos.get("risk") or abs(entry - initial_stop)) or 1e-9
    target = float(pos["target"])
    cur_stop = float(pos.get("cur_stop", initial_stop))
    best = float(pos.get("best_price", entry))

    entry_bar = pd.Timestamp(pos["entry_bar"]) if pos.get("entry_bar") else None
    last_bar = pd.Timestamp(pos["last_bar"]) if pos.get("last_bar") else None
    if entry_bar is None and last_bar is None:
        bars = d.iloc[-1:]  # Altbestand ohne Kerzen-Historie
    else:
        mask = pd.Series(True, index=d.index)
        if entry_bar is not None:
            mask &= d.index > entry_bar
        if last_bar is not None:
            mask &= d.index >= last_bar  # letzte (evtl. damals unfertige) Kerze erneut prüfen
        bars = d[mask]

    exit_price = reason = None
    when = analysis["candle_time"]
    for ts, row in bars.iterrows():
        o, h, l = float(row["Open"]), float(row["High"]), float(row["Low"])
        hit_stop = l <= cur_stop if side == 1 else h >= cur_stop
        hit_target = h >= target if side == 1 else l <= target
        if hit_stop:
            gapped = o < cur_stop if side == 1 else o > cur_stop
            exit_price = o if gapped else cur_stop
            locked_r = side * (cur_stop - entry) / risk
            reason = ("Trailing-Stop ausgelöst -- Gewinn selbst gesichert" if locked_r > 0.05
                      else "Break-even-Stop -- ohne Verlust raus" if locked_r > -0.05
                      else "Stop-Loss erreicht")
            when = ts.isoformat()
            break
        if hit_target:
            gapped = o > target if side == 1 else o < target
            exit_price = o if gapped else target
            reason = "Take-Profit erreicht"
            when = ts.isoformat()
            break
        if last_bar is None or ts > last_bar:
            pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
        best = max(best, h) if side == 1 else min(best, l)
        best_r = side * (best - entry) / risk
        if best_r >= cfg.breakeven_r:
            cur_stop = _better(side, cur_stop, entry)
        if best_r >= cfg.trailing_activate_r:
            cur_stop = _better(side, cur_stop, best - side * cfg.trailing_distance_r * risk)

    pos["best_price"], pos["cur_stop"] = best, cur_stop
    pos["last_bar"] = d.index[-1].isoformat()
    pos["runs_held"] = int(pos.get("runs_held", 0)) + 1

    if exit_price is None:
        price = analysis["price"]
        opposite = analysis["direction"] == ("short" if side == 1 else "long")
        wrong_side_of_ema = side * (price - analysis["ema_s"]) < -0.25 * analysis["atr"]
        if opposite:
            exit_price, reason = price, "Trendwechsel: Gegensignal (EMA/RSI/MACD drehen)"
        elif cfg.exit_on_ema_cross and wrong_side_of_ema and int(pos.get("bars_held", 0)) >= 2:
            exit_price, reason = price, "Momentum weg: Kurs auf der falschen Seite der EMA 21"
        elif cfg.max_hold_cycles and int(pos.get("bars_held", 0)) >= cfg.max_hold_cycles:
            exit_price, reason = price, "Maximale Haltedauer erreicht -- vorsorglich ausgecasht"
    if exit_price is None:
        return None
    return _close_position(cfg, account, pos, float(exit_price), reason, when)


def _is_stale(candle_time: str, interval: str, now: datetime) -> bool:
    minutes = INTERVAL_MINUTES.get(interval, 1440)
    allowed = 4 * 1440 if minutes >= 1440 else 3 * minutes + 10
    age = (now - pd.Timestamp(candle_time).to_pydatetime()).total_seconds() / 60
    return age > allowed


def open_position(cfg: BotConfig, account: dict, cand: dict, risk_scale: float) -> dict | None:
    side = 1 if cand["direction"] == "long" else -1
    price, stop = float(cand["price"]), float(cand["stop"])
    risk_per_unit = abs(price - stop)
    if price <= 0 or risk_per_unit <= 0:
        return None

    equity = account["equity"]
    available_cash = account["cash"] - equity * cfg.min_cash_reserve_pct - cfg.fee_per_trade
    if available_cash <= 0:
        return None

    confidence = float(np.clip((cand["score"] - cfg.min_score) / max(100 - cfg.min_score, 1), 0, 1))
    risk_budget = equity * account["risk_percent"] / 100 * risk_scale * (0.6 + 0.6 * confidence)
    leverage = 1.0 if risk_scale < 1 else round(1.0 + confidence * (max(cfg.max_leverage, 1.0) - 1.0), 2)
    margin_cap = min(equity * cfg.max_position_pct, available_cash)
    # Hebel vergrößert nur dann die Position, wenn die Margin-Grenze bindet --
    # das Risiko je Trade bleibt durch den Stop-Abstand gedeckelt.
    units = min(risk_budget / risk_per_unit, margin_cap * leverage / price)
    margin = units * price / leverage
    if units <= 0 or margin < 10:
        return None

    account["cash"] -= margin + cfg.fee_per_trade
    account["positions"].append({
        "ticker": cand["ticker"], "side": "long" if side == 1 else "short",
        "units": units, "cost": units * price, "margin": margin, "leverage": leverage,
        "entry": price, "stop": stop, "cur_stop": stop, "risk": risk_per_unit,
        "target": float(cand["target"]), "best_price": price,
        "entry_bar": cand["candle_time"], "last_bar": cand["candle_time"],
        "bars_held": 0, "runs_held": 0, "score": cand["score"],
        "fee_paid": cfg.fee_per_trade, "wichtig": bool(cand.get("wichtig")),
        "opened_at": datetime.now(timezone.utc).isoformat(),
    })
    edge = cand.get("edge") or {}
    wr = edge.get("win_rate")
    richtung = "Long" if side == 1 else "Short"
    hebel = f"{leverage:.1f}x Hebel" if leverage > 1.0 else "ohne Hebel"
    star = "⭐ " if cand.get("wichtig") else ""
    backtest = f", Backtest {wr * 100:.0f} % Treffer in {edge.get('trades', 0)} Trades" if wr is not None else ""
    return {
        "Zeit": cand["candle_time"], "Aktion": "KAUF", "Asset": cand["ticker"],
        "Preis": round(price, 4), "Menge": round(units, 6), "Ergebnis": 0.0,
        "Warum": (
            f"{star}{richtung}-Einstieg, Score {cand['score']:.0f}: "
            f"EMA-Trend + RSI {cand['rsi']:.0f} + ADX {cand['adx']:.0f}{backtest} "
            f"({margin / equity * 100:.1f} % Margin, {hebel})"
        ),
        "_signal": {"side": richtung.lower(), "score": cand["score"], "leverage": leverage,
                    "edge_r": round(edge.get("exp_r", 0.0), 3), "edge_trades": edge.get("trades", 0)},
    }


def run_cycle(cfg: BotConfig, analyses: dict[str, dict], account: dict,
              now: datetime | None = None) -> tuple[dict, list[dict]]:
    now = now or datetime.now(timezone.utc)
    memory = account["memory"]
    memory["runs"] = int(memory.get("runs", 0)) + 1
    events: list[dict] = []
    price_by_ticker = {t: a["price"] for t, a in analyses.items()}

    # 1) Exits zuerst -- bei JEDEM Lauf
    closed_now: set[str] = set()
    for pos in list(account["positions"]):
        analysis = analyses.get(pos["ticker"])
        if not analysis:
            continue
        event = manage_position(cfg, account, pos, analysis)
        if event:
            events.append(event)
            closed_now.add(pos["ticker"])

    # 2) Risiko an die eigene Performance anpassen (Drawdown-Bremse)
    account["equity"] = _mark_to_market_equity(account, price_by_ticker)
    peak = max(float(memory.get("peak_equity", account["equity"])), account["equity"])
    memory["peak_equity"] = peak
    drawdown_pct = (1 - account["equity"] / peak) * 100 if peak > 0 else 0.0
    risk_scale = 0.5 if drawdown_pct >= cfg.max_drawdown_pct else 1.0
    memory["drawdown_pct"] = round(drawdown_pct, 2)

    # 3) Neue Einstiege: beste Scores zuerst
    cooldown = {t: r for t, r in (memory.get("cooldown") or {}).items() if r > memory["runs"]}
    memory["cooldown"] = cooldown
    held = {p["ticker"] for p in account["positions"]}
    candidates = [
        a for a in analyses.values()
        if a["tradable"] and a["score"] >= cfg.min_score
        and (a.get("edge") or {}).get("exp_shrunk", 0.0) >= cfg.min_edge
        and a["ticker"] not in held and a["ticker"] not in closed_now
        and a["ticker"] not in cooldown
        and account["seen_candles"].get(a["ticker"]) != a["candle_time"]
        and not _is_stale(a["candle_time"], cfg.interval, now)
    ]
    candidates.sort(key=lambda a: a["score"], reverse=True)
    opened = 0
    for cand in candidates:
        if len(account["positions"]) >= cfg.max_positions or opened >= cfg.max_new_trades_per_run:
            break
        account["seen_candles"][cand["ticker"]] = cand["candle_time"]
        event = open_position(cfg, account, cand, risk_scale)
        if event:
            events.append(event)
            opened += 1

    # seen_candles klein halten: nur Assets dieses Laufs
    account["seen_candles"] = {t: v for t, v in account["seen_candles"].items() if t in analyses}
    account["equity"] = _mark_to_market_equity(account, price_by_ticker)
    memory["last_run"] = now.isoformat()
    return account, events


# ------------------------------------------------------------
# Lauf
# ------------------------------------------------------------
def _print_wichtig_report(cfg: BotConfig, analyses: dict[str, dict]) -> None:
    rows = [a for a in analyses.values() if a.get("wichtig")]
    if not rows:
        return
    rows.sort(key=lambda a: a["day_change"] if a["day_change"] is not None else -999, reverse=True)
    print(f"[{cfg.name}] ⭐ Wichtig-Watchlist ({len(rows)} Assets):")
    for a in rows:
        chg = f"{a['day_change']:+6.2f} %" if a["day_change"] is not None else "   –   "
        alert = "  ⚠️ starke Bewegung" if a["day_change"] is not None and abs(a["day_change"]) >= cfg.watch_alert_pct else ""
        sig = f"{a['signal']} · Score {a['score']:.0f}" if a["direction"] != "flat" else "ABWARTEN"
        print(f"    {a['ticker']:<9} {chg}  RSI {a['rsi']:5.1f}  {sig}{alert}")


def main(cfg: BotConfig, app_py_path: str | None = None, dry_run: bool = False) -> None:
    t0 = time.time()
    now = datetime.now(timezone.utc)
    is_weekend = now.weekday() >= 5

    client = None if dry_run else get_client()
    account = load_account_dry(cfg) if dry_run else load_account(client, cfg)
    memory = account["memory"]

    stocks, crypto = load_universe(app_py_path)
    wichtig_all = load_wichtig(client)
    # Wochenende: Aktienbörsen zu -> nur Krypto (+ offene Positionen). Werktags: Aktien.
    wichtig = {l: t for l, t in wichtig_all.items() if t.endswith("-USD") == is_weekend}
    base = crypto if is_weekend else stocks
    held = [p["ticker"] for p in account["positions"]]
    meta = build_universe(cfg, base, wichtig, held, memory, "crypto" if is_weekend else "stocks")
    mode = "Krypto (Wochenende)" if is_weekend else "Aktien (Werktag)"
    print(f"[{cfg.name}] Modus: {mode} · {len(meta)} Assets in diesem Lauf "
          f"({len(held)} offen, {len(wichtig)} ⭐ Wichtig, Rest rotierend){' · DRY-RUN' if dry_run else ''}")

    data = download_many(list(meta), cfg.period, cfg.interval)
    print(f"[{cfg.name}] Kursdaten für {len(data)}/{len(meta)} Assets in {time.time() - t0:.1f}s geladen")
    if not data:
        print(f"[{cfg.name}] Keine verwertbaren Marktdaten in diesem Lauf.")
        return

    analyses: dict[str, dict] = {}
    for ticker, df in data.items():
        label, is_wichtig = meta.get(ticker, (ticker, False))
        try:
            a = analyze(df, cfg, ticker, label, is_wichtig, memory)
            if a:
                analyses[ticker] = a
        except Exception as exc:
            print(f"  [skip] {ticker}: {exc}")

    account, events = run_cycle(cfg, analyses, account, now)
    if dry_run:
        save_account_dry(cfg, account, events)
    else:
        save_account(client, cfg, account, events)

    for event in events:
        print(f"[{cfg.name}] {event['Aktion']} {event['Asset']} @ {event['Preis']} · "
              f"Ergebnis {event['Ergebnis']:+.2f} · {event['Warum']}")
    if not events:
        print(f"[{cfg.name}] Kein neuer Trade in diesem Lauf.")
    _print_wichtig_report(cfg, analyses)

    signals = sorted((a for a in analyses.values() if a["direction"] != "flat"), key=lambda a: a["score"], reverse=True)
    if signals:
        print(f"[{cfg.name}] Top-Signale: " + ", ".join(
            f"{a['ticker']} {a['direction']} {a['score']:.0f}" for a in signals[:8]))
    totals = memory.get("totals") or {}
    wr = f"{totals['wins'] / totals['trades'] * 100:.0f} %" if totals.get("trades") else "–"
    open_txt = ", ".join(f"{p['ticker']}({p.get('side', 'long')})" for p in account["positions"]) or "keine"
    print(f"[{cfg.name}] Kontowert {account['equity']:,.2f} · Cash {account['cash']:,.2f} · "
          f"Drawdown {memory.get('drawdown_pct', 0)} % · Gewinnquote {wr} · Offen: {open_txt} · "
          f"Laufzeit {time.time() - t0:.1f}s")


def run_bot(cfg: BotConfig) -> None:
    dry_run = "--dry-run" in sys.argv or os.environ.get("BOT_DRY_RUN") == "1"
    try:
        main(cfg, dry_run=dry_run)
    except Exception as exc:
        print(f"[{cfg.name}] FEHLER: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
