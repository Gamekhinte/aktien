"""
Jerry (grün) - Minuten-Trader.

Handelt auf 15-Minuten-Kerzen, läuft per GitHub Actions alle paar Minuten
(siehe jerry_bot.yml). Höchste Frequenz der drei Bots -- gerade deshalb mit
den KLEINSTEN Positionsgrößen und den MEISTEN gleichzeitigen Positionen, damit
ein einzelner schneller Fehltrade nie ins Gewicht fällt.

Hinweis: Yahoo Finance liefert Intraday-Daten (<1h) nur für einen begrenzten
Zeitraum zurück (im Regelfall die letzten ~60 Tage bei 15m-Kerzen). Für
Krypto (Wochenende) funktioniert das rund um die Uhr, bei Aktien nur während
der Handelszeiten der jeweiligen Börse.
"""
from bot_engine import BotConfig, run_bot

CONFIG = BotConfig(
    account_key="jerry_bot_v1",
    name="Jerry",
    starting_cash=10_000.0,
    risk_percent=0.5,          # kleinstes Risiko pro Trade -- hohe Frequenz braucht kleine Häppchen
    max_position_pct=0.10,     # nie mehr als 10 % des Kontos in einer Position
    max_positions=6,           # bis zu 6 Positionen gleichzeitig -> stark diversifiziert
    min_cash_reserve_pct=0.15, # größerer Cash-Puffer wegen der hohen Handelsfrequenz
    interval="15m",
    period="60d",
    scan_limit=60,             # etwas kleineres Universum, damit ein Lauf im Zeitbudget bleibt

    # Jerry ist der schnellste der drei: er darf bis zu 3x hebeln (selbst
    # gewählt, je nach Signalstärke) und sichert Gewinne sehr früh und eng --
    # dadurch verkauft er deutlich schneller als vorher.
    max_leverage=3.0,
    trailing_activate_r=0.5,   # zieht den Stop schon nach, sobald er die halbe Risikodistanz im Plus ist
    trailing_distance_r=0.3,   # und lässt den Kurs dabei nur wenig Spielraum
    max_hold_cycles=12,        # cashed notfalls nach 12 Läufen (~3h bei 15-Min-Kerzen) zwangsweise aus
)

if __name__ == "__main__":
    run_bot(CONFIG)
