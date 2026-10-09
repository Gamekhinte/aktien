"""
Jerry (grün) - Minuten-Trader.

Handelt auf 15-Minuten-Kerzen, läuft per GitHub Actions alle 15 Minuten
(siehe jerry_bot.yml). Höchste Frequenz der drei Bots -- gerade deshalb mit
den KLEINSTEN Positionsgrößen und den MEISTEN gleichzeitigen Positionen, damit
ein einzelner schneller Fehltrade nie ins Gewicht fällt.

Jerry kauft UND verkauft selbstständig: Long bei Aufwärtstrend, Short bei
Abwärtstrend, sichert Gewinne sehr früh (Break-even + enger Trailing-Stop)
und schaut bei jedem Lauf auf die ⭐ Wichtig-Watchlist.

Hinweis: Yahoo Finance liefert 15m-Kerzen nur für die letzten ~60 Tage. Aktien
bewegen sich nur während der Börsenzeiten -- außerhalb davon eröffnet Jerry
keine neuen Positionen auf veralteten Kursen.

Lokal testen: python jerry_bot.py --dry-run
"""
from bot_engine import BotConfig, run_bot

CONFIG = BotConfig(
    account_key="jerry_bot_v1",
    name="Jerry",
    starting_cash=10_000.0,
    risk_percent=0.5,          # kleinstes Risiko pro Trade -- hohe Frequenz braucht kleine Häppchen
    max_position_pct=0.10,     # nie mehr als 10 % des Kontos als Margin in einer Position
    max_positions=6,           # bis zu 6 Positionen gleichzeitig -> stark diversifiziert
    min_cash_reserve_pct=0.15, # größerer Cash-Puffer wegen der hohen Handelsfrequenz
    interval="15m",
    period="59d",              # Yahoo erlaubt 15m nur innerhalb der letzten 60 Tage
    scan_limit=100,            # + Wichtig + offene Positionen; rotiert -> alle Assets in ~2 h

    # Jerry ist der schnellste: bis zu 3x Hebel (selbst gewählt je nach Score),
    # sehr früher Break-even und enger Trailing-Stop.
    max_leverage=3.0,
    breakeven_r=0.7,
    trailing_activate_r=0.8,
    trailing_distance_r=0.4,
    max_hold_cycles=16,        # max. 16 Kerzen (~4 h) -- dann wird ausgecasht

    allow_short=True,
    min_score=55.0,
    min_adx=20.0,
    atr_mult=1.5,
    reward_risk=2.0,
    edge_horizon=16,
    max_new_trades_per_run=2,
    cooldown_runs=8,           # nach Verlust 2 h Pause für dieses Asset
)

if __name__ == "__main__":
    run_bot(CONFIG)
