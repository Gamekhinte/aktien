"""
Jan (gelb) - Stunden-Trader.

Handelt auf 1h-Kerzen, läuft per GitHub Actions alle paar Stunden (siehe
jan_bot.yml). Etwas höhere Frequenz als Jerry, dafür kleinere Positionen
und mehr gleichzeitige Trades, um das Risiko pro Position weiter zu senken.
"""
from bot_engine import BotConfig, run_bot

CONFIG = BotConfig(
    account_key="jan_bot_v1",
    name="Jan",
    starting_cash=10_000.0,
    risk_percent=0.75,         # etwas vorsichtiger pro Trade als Jerry
    max_position_pct=0.15,     # nie mehr als 15 % des Kontos in einer Position
    max_positions=5,           # bis zu 5 Positionen gleichzeitig
    min_cash_reserve_pct=0.10,
    interval="1h",
    period="60d",
    scan_limit=80,
)

if __name__ == "__main__":
    run_bot(CONFIG)
