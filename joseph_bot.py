"""
Joseph (rot) - ruhiger Tages-Trader.

Handelt auf Tages-Kerzen, läuft per GitHub Actions 2x täglich (siehe
joseph_bot.yml). Als "alter Hase" der drei Bots handelt er am seltensten und
am überlegtesten: nur Long, höchste Mindest-Qualität, weitere Stops und er
lässt Gewinne geduldig laufen.

Lokal testen: python joseph_bot.py --dry-run
"""
from bot_engine import BotConfig, run_bot

CONFIG = BotConfig(
    account_key="joseph_bot_v1",
    name="Joseph",
    starting_cash=10_000.0,
    risk_percent=1.0,          # 1 % Risiko je Trade (Stop-Distanz)
    max_position_pct=0.20,     # nie mehr als 20 % des Kontos als Margin in einer Position
    max_positions=4,           # bis zu 4 Positionen gleichzeitig -> Diversifikation
    min_cash_reserve_pct=0.10, # 10 % Cash-Puffer bleibt immer unangetastet
    interval="1d",
    period="2y",
    scan_limit=150,

    max_leverage=1.5,          # vorsichtig, und nur bei richtig starkem Signal
    breakeven_r=1.0,
    trailing_activate_r=1.5,
    trailing_distance_r=0.8,
    max_hold_cycles=25,        # ~5 Wochen

    allow_short=False,         # Joseph setzt nicht auf fallende Kurse
    min_score=60.0,
    min_adx=18.0,
    atr_mult=2.0,
    reward_risk=2.5,
    edge_horizon=15,
    max_new_trades_per_run=2,
    cooldown_runs=6,           # ~3 Tage Pause nach einem Verlust
)

if __name__ == "__main__":
    run_bot(CONFIG)
