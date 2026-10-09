"""
 "Wichtig"-Watchlist -- Assets, auf die IMMER ein Auge geworfen wird.

Wird von app.py (Scanner, Watchlist-Ansicht) UND von den Bots (bot_engine.py)
benutzt, deshalb bewusst ohne Streamlit-Import.

- WICHTIG_DEFAULT ist die Grundliste (aus den Trade-Republic-Screenshots).
- Wird die Liste in der App bearbeitet, landet sie in Supabase
  (Tabelle learning_state, key = "wichtig_watchlist") -- die Bots lesen
  dann automatisch die bearbeitete Version.

Ticker sind Yahoo-Finance-Symbole (deutsche Werte über XETRA = .DE usw.).
"""
from __future__ import annotations

WICHTIG_STATE_KEY = "wichtig_watchlist"

WICHTIG_DEFAULT: dict[str, str] = {
    "Palantir Technologies": "PLTR",
    "TotalEnergies": "TTE.PA",
    "SAP": "SAP.DE",
    "Deutsche Börse": "DB1.DE",
    "Rheinmetall": "RHM.DE",
    "Apple": "AAPL",
    "SpaceX": "SPCX",
    "RENK Group": "R3NK.DE",
    "UiPath": "PATH",
    "Edison International": "EIX",
    "Coinbase Global (A)": "COIN",
    "Alphabet (A)": "GOOGL",
    "Amazon.com": "AMZN",
    "Tesla": "TSLA",
    "Strategy A": "MSTR",
    "NVIDIA": "NVDA",
    "BioNTech (ADR)": "BNTX",
    "Hensoldt": "HAG.DE",
    "Oracle": "ORCL",
    "Micron Technology": "MU",
    "MP Materials": "MP",
    "Siemens Energy": "ENR.DE",
    "D-Wave Quantum": "QBTS",
    "ASML": "ASML.AS",
    "Bayer": "BAYN.DE",
    "Rocket Lab": "RKLB",
    "Xiaomi": "1810.HK",
    "Bloom Energy": "BE",
    "Marvell Technology": "MRVL",
    "Lumentum Holdings": "LITE",
    "ARM (ADR)": "ARM",
    "Hubspot": "HUBS",
    "DroneShield": "DRO.AX",
    "Redwood AI": "RDWCF",
    "Infineon Technologies": "IFX.DE",
    "Zhejiang Sanhua Intelligent": "2050.HK",
}


def _clean(raw) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for label, ticker in raw.items():
        if isinstance(label, str) and isinstance(ticker, str) and ticker.strip():
            out[label.strip()] = ticker.strip().upper()
    return out


def load_wichtig(client=None) -> dict[str, str]:
    """Liest die (ggf. in der App bearbeitete) Liste aus Supabase, sonst Default."""
    if client is not None:
        try:
            rows = (
                client.table("learning_state").select("value")
                .eq("key", WICHTIG_STATE_KEY).limit(1).execute().data or []
            )
            if rows:
                stored = _clean((rows[0].get("value") or {}).get("assets"))
                if stored:
                    return stored
        except Exception:
            pass
    return dict(WICHTIG_DEFAULT)


def save_wichtig(client, assets: dict[str, str]) -> bool:
    if client is None:
        return False
    try:
        client.table("learning_state").upsert({
            "key": WICHTIG_STATE_KEY,
            "value": {"assets": _clean(assets)},
        }).execute()
        return True
    except Exception:
        return False
