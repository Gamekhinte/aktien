"""
Jan (gelb) - Stunden-Trader.

Handelt auf 1h-Kerzen, läuft per GitHub Actions jede Stunde (siehe
jan_bot.yml). Mittlere Frequenz zwischen Jerry und Joseph: moderater Hebel,
mittleres Trailing-Tempo, Long UND Short.

Lokal testen: python jan_bot.py --dry-run
"""
from bot_engine import BotConfig, run_bot

CONFIG = BotConfig(
    account_key="jan_bot_v1",
    name="Jan",
    starting_cash=10_000.0,
    risk_percent=0.75,         # etwas mehr Risiko als Jerry, weniger als Joseph
    max_position_pct=0.15,     # nie mehr als 15 % des Kontos als Margin in einer Position
    max_positions=5,           # bis zu 5 Positionen gleichzeitig
    min_cash_reserve_pct=0.10,
    interval="1h",
    period="180d",             # genug Historie für EMA 50 + Backtest der eigenen Edge
    scan_limit=120,

    max_leverage=2.0,
    breakeven_r=0.8,
    trailing_activate_r=1.0,
    trailing_distance_r=0.5,
    max_hold_cycles=35,        # ~5 Handelstage bei 1h-Kerzen

    allow_short=True,
    min_score=57.0,
    min_adx=20.0,
    atr_mult=1.5,
    reward_risk=2.0,
    edge_horizon=14,
    max_new_trades_per_run=2,
    cooldown_runs=6,
)

if __name__ == "__main__":
    run_bot(CONFIG)
