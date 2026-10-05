"""
Joseph (rot) - ruhiger Tages-Trader.

Handelt auf Tages-Kerzen, läuft per GitHub Actions ca. 2x täglich (siehe
joseph_bot.yml). Verteilt sein Kapital auf bis zu 4 Positionen gleichzeitig
und riskiert je Trade nur einen kleinen Bruchteil des Kontos -- er setzt
nie alles auf eine Karte. Als "alter Hase" der drei Bots handelt er am
seltensten und am überlegtesten.
"""
from bot_engine import BotConfig, run_bot

CONFIG = BotConfig(
    account_key="joseph_bot_v1",
    name="Joseph",
    starting_cash=10_000.0,
    risk_percent=1.0,          # 1 % Risiko je Trade (Stop-Distanz)
    max_position_pct=0.20,     # nie mehr als 20 % des Kontos in einer Position
    max_positions=4,           # bis zu 4 Positionen gleichzeitig -> Diversifikation
    min_cash_reserve_pct=0.10, # 10 % Cash-Puffer bleibt immer unangetastet
    interval="1d",
    period="1y",
    scan_limit=80,

    # Joseph hebelt nur vorsichtig (max. 1.5x, und auch das nur bei richtig
    # starkem Signal) und lässt Gewinne länger laufen, bevor er nachzieht --
    # passt zu seinem geduldigen "alter Hase"-Charakter.
    max_leverage=1.5,
    trailing_activate_r=1.0,
    trailing_distance_r=0.6,
    max_hold_cycles=20,        # ~20 Handelstage, bevor notfalls zwangsweise ausgecasht wird
)

if __name__ == "__main__":
    run_bot(CONFIG)
