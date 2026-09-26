"""
Background learner.

Run this in a second terminal:
    uv run python train_worker.py

It periodically downloads fresh market history, updates the shared SQLite
learning store and retrains only when enough new labeled samples exist.
"""
from __future__ import annotations

import time
from learning_engine import SharedLearningEngine, acquire_training_lock, release_training_lock

# Keep these in sync with the ASSETS/INTERVAL_CONFIG of app.py.
ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
}

INTERVALS = ["1d"]


def main() -> None:
    engine = SharedLearningEngine(
        root="learning_store",
        horizon=5,
        min_samples=250,
        retrain_after_new_samples=25,
    )

    print("Gemeinsamer Lern-Worker gestartet.")
    print("Strg+C beendet den Worker.")

    while True:
        for asset, ticker in ASSETS.items():
            for interval in INTERVALS:
                try:
                    added = engine.sync_from_yfinance(
                        asset=asset,
                        interval=interval,
                        ticker=ticker,
                        period="2y",
                    )
                    result = engine.train_if_needed(asset, interval)

                    print(
                        f"[{asset} | {interval}] "
                        f"+{added} Samples | {result.status} | {result.message}"
                    )
                except Exception as exc:
                    print(f"[{asset} | {interval}] FEHLER: {exc}")

        # Yahoo Finance data is not tick-level. One hour is plenty for this
        # educational/paper-learning loop and avoids hammering the endpoint.
        time.sleep(3600)


if __name__ == "__main__":
    if acquire_training_lock():
        try:
            main()
        finally:
            release_training_lock()
    else:
        raise SystemExit("Ein anderer Lern-Worker läuft bereits.")
