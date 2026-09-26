"""
Shared, persistent learning engine for the Streamlit trading app.

Design goals:
- One shared model state for every user of the same server.
- SQLite stores the learned samples and model registry.
- Models are versioned on disk with joblib.
- Training uses time-ordered data and a final holdout; no random shuffle.
- A candidate model is only activated when it passes validation gates.
- The engine never places real trades.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


FEATURES = [
    "rsi",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "bb_pos",
    "ema_gap",
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_ratio",
    "body_ratio",
    "range_norm",
]

MODEL_VERSION = 1


@dataclass
class TrainResult:
    asset: str
    interval: str
    status: str
    message: str
    version: int | None = None
    samples: int = 0
    accuracy: float | None = None
    roc_auc: float | None = None
    log_loss_value: float | None = None


class SharedLearningEngine:
    def __init__(
        self,
        root: str | Path = "learning_store",
        horizon: int = 5,
        min_samples: int = 250,
        retrain_after_new_samples: int = 25,
        holdout_fraction: float = 0.20,
    ) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.db_path = self.root / "learning.sqlite3"
        self.lock_dir = self.root / ".train_lock"
        self.horizon = horizon
        self.min_samples = min_samples
        self.retrain_after_new_samples = retrain_after_new_samples
        self.holdout_fraction = holdout_fraction

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS samples (
                    asset TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    target INTEGER NOT NULL,
                    close REAL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(asset, interval, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_samples_asset_interval_ts
                ON samples(asset, interval, timestamp);

                CREATE TABLE IF NOT EXISTS model_registry (
                    asset TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    model_path TEXT NOT NULL,
                    trained_at TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    accuracy REAL,
                    roc_auc REAL,
                    log_loss REAL,
                    status TEXT NOT NULL,
                    PRIMARY KEY(asset, interval, version)
                );

                CREATE TABLE IF NOT EXISTS state (
                    asset TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    last_seen_timestamp TEXT,
                    last_trained_sample_count INTEGER DEFAULT 0,
                    active_version INTEGER,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(asset, interval)
                );
                """
            )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        data.columns = [str(c).strip().title() for c in data.columns]
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Fehlende Kursdaten: {missing}")

        if "Datetime" in data.columns:
            data["Date"] = pd.to_datetime(data["Datetime"], utc=True, errors="coerce")
        elif "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], utc=True, errors="coerce")
        else:
            data["Date"] = pd.to_datetime(data.index, utc=True, errors="coerce")

        data = data.dropna(subset=["Date", "Open", "High", "Low", "Close"]).copy()
        data = data.sort_values("Date").drop_duplicates("Date")
        return data.reset_index(drop=True)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = self._normalize_ohlcv(df)
        close = data["Close"].astype(float)
        high = data["High"].astype(float)
        low = data["Low"].astype(float)
        open_ = data["Open"].astype(float)
        volume = data["Volume"].astype(float).replace(0, np.nan)

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        data["rsi"] = 100 - 100 / (1 + rs)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        # Normalize price-dependent MACD values so one model can be used
        # across assets with very different nominal prices.
        data["macd_norm"] = macd / close.replace(0, np.nan)
        data["macd_signal_norm"] = macd_signal / close.replace(0, np.nan)
        data["macd_hist_norm"] = (macd - macd_signal) / close.replace(0, np.nan)

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        data["bb_pos"] = (close - sma20) / std20.replace(0, np.nan)

        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        data["ema_gap"] = (ema9 - ema21) / close.replace(0, np.nan)

        data["ret_1"] = close.pct_change(1)
        data["ret_3"] = close.pct_change(3)
        data["ret_5"] = close.pct_change(5)
        data["ret_10"] = close.pct_change(10)

        vol_mean = volume.rolling(20).mean()
        data["vol_ratio"] = volume / vol_mean.replace(0, np.nan)

        candle_range = (high - low).replace(0, np.nan)
        data["body_ratio"] = (close - open_).abs() / candle_range
        data["range_norm"] = candle_range / close.replace(0, np.nan)

        # The target is deliberately created AFTER all current-candle features.
        # A sample at t predicts the direction at t+horizon.
        future_return = close.shift(-self.horizon) / close - 1
        volatility = data["ret_1"].rolling(50).std()
        threshold = volatility.fillna(data["ret_1"].std()).clip(lower=0.0005)

        data["target"] = np.where(
            future_return > threshold,
            1,
            np.where(future_return < -threshold, 0, np.nan),
        )

        return data

    def sync_from_yfinance(
        self,
        asset: str,
        interval: str,
        ticker: str,
        period: str = "2y",
    ) -> int:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return 0

        data = self.build_features(raw)
        usable = data.dropna(subset=FEATURES + ["target", "Date"]).copy()

        rows: list[tuple[Any, ...]] = []
        now = self._utc_now()

        for _, row in usable.iterrows():
            feature_values = {
                f: float(row[f])
                for f in FEATURES
                if pd.notna(row[f]) and math.isfinite(float(row[f]))
            }
            if len(feature_values) != len(FEATURES):
                continue

            rows.append(
                (
                    asset,
                    interval,
                    pd.Timestamp(row["Date"]).isoformat(),
                    json.dumps(feature_values, separators=(",", ":")),
                    int(row["target"]),
                    float(row["Close"]),
                    now,
                )
            )

        if not rows:
            return 0

        with self._connect() as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO samples
                (asset, interval, timestamp, features_json, target, close, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            con.execute(
                """
                INSERT INTO state(asset, interval, last_seen_timestamp,
                                   last_trained_sample_count, active_version, updated_at)
                VALUES (?, ?, ?, COALESCE(
                    (SELECT last_trained_sample_count FROM state
                     WHERE asset=? AND interval=?), 0),
                    COALESCE(
                    (SELECT active_version FROM state
                     WHERE asset=? AND interval=?), NULL), ?)
                ON CONFLICT(asset, interval) DO UPDATE SET
                    last_seen_timestamp=excluded.last_seen_timestamp,
                    updated_at=excluded.updated_at
                """,
                (
                    asset,
                    interval,
                    rows[-1][2],
                    asset,
                    interval,
                    asset,
                    interval,
                    now,
                ),
            )
        return len(rows)

    def _load_samples(self, asset: str, interval: str) -> pd.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT timestamp, features_json, target
                FROM samples
                WHERE asset=? AND interval=?
                ORDER BY timestamp ASC
                """,
                (asset, interval),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            features = json.loads(r["features_json"])
            features["timestamp"] = r["timestamp"]
            features["target"] = int(r["target"])
            records.append(features)

        return pd.DataFrame(records)

    @staticmethod
    def _make_model() -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=220,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=8,
            subsample=0.85,
            random_state=42,
        )

    def _walk_forward_metrics(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[FEATURES]
        y = data["target"].astype(int)

        # Five chronological folds. No shuffle.
        n = len(data)
        fold_size = max(20, n // 6)
        metrics: list[tuple[float, float, float]] = []

        for fold in range(1, 6):
            train_end = fold_size * fold
            test_end = min(fold_size * (fold + 1), n)
            if train_end < 100 or test_end <= train_end:
                continue

            train_x = X.iloc[:train_end]
            train_y = y.iloc[:train_end]
            test_x = X.iloc[train_end:test_end]
            test_y = y.iloc[train_end:test_end]

            if train_y.nunique() < 2 or test_y.nunique() < 2:
                continue

            model = self._make_model()
            model.fit(train_x, train_y)
            proba = model.predict_proba(test_x)[:, 1]
            pred = (proba >= 0.5).astype(int)

            acc = accuracy_score(test_y, pred)
            auc = roc_auc_score(test_y, proba)
            ll = log_loss(test_y, np.clip(proba, 1e-6, 1 - 1e-6))
            metrics.append((float(acc), float(auc), float(ll)))

        if not metrics:
            raise ValueError("Zu wenig unterschiedliche Daten für Walk-Forward-Validierung.")

        arr = np.array(metrics)
        return {
            "accuracy": float(arr[:, 0].mean()),
            "roc_auc": float(arr[:, 1].mean()),
            "log_loss": float(arr[:, 2].mean()),
        }

    def _next_version(self, asset: str, interval: str) -> int:
        with self._connect() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM model_registry WHERE asset=? AND interval=?",
                (asset, interval),
            ).fetchone()
        return int(row["v"]) + 1

    def train_if_needed(
        self,
        asset: str,
        interval: str,
        force: bool = False,
    ) -> TrainResult:
        data = self._load_samples(asset, interval)
        if len(data) < self.min_samples:
            return TrainResult(
                asset, interval, "not_ready",
                f"Nur {len(data)} Lernbeispiele; benötigt werden mindestens {self.min_samples}.",
                samples=len(data),
            )

        with self._connect() as con:
            state = con.execute(
                "SELECT last_trained_sample_count, active_version FROM state WHERE asset=? AND interval=?",
                (asset, interval),
            ).fetchone()

        last_count = int(state["last_trained_sample_count"]) if state else 0
        if not force and len(data) - last_count < self.retrain_after_new_samples:
            return TrainResult(
                asset, interval, "waiting",
                f"Noch {self.retrain_after_new_samples - (len(data) - last_count)} neue Beispiele bis zum nächsten Training.",
                version=int(state["active_version"]) if state and state["active_version"] else None,
                samples=len(data),
            )

        # Keep the newest holdout completely out of fitting.
        holdout_n = max(40, int(len(data) * self.holdout_fraction))
        train = data.iloc[:-holdout_n].copy()
        holdout = data.iloc[-holdout_n:].copy()

        if train["target"].nunique() < 2 or holdout["target"].nunique() < 2:
            return TrainResult(asset, interval, "not_ready",
                               "Holdout enthält nicht beide Klassen; mehr Daten abwarten.",
                               samples=len(data))

        wf = self._walk_forward_metrics(data)

        candidate = self._make_model()
        candidate.fit(train[FEATURES], train["target"].astype(int))

        proba = candidate.predict_proba(holdout[FEATURES])[:, 1]
        pred = (proba >= 0.5).astype(int)
        holdout_acc = float(accuracy_score(holdout["target"], pred))
        holdout_auc = float(roc_auc_score(holdout["target"], proba))
        holdout_ll = float(log_loss(holdout["target"], np.clip(proba, 1e-6, 1 - 1e-6)))

        # Conservative gate: the candidate must be at least plausible on
        # the newest unseen data. This avoids blindly replacing a model.
        baseline = max(float(holdout["target"].mean()), 1 - float(holdout["target"].mean()))
        passes = (
            wf["accuracy"] >= 0.50
            and holdout_acc >= max(0.50, baseline - 0.02)
            and holdout_auc >= 0.50
            and holdout_ll < 0.72
        )

        if not passes:
            return TrainResult(
                asset, interval, "rejected",
                "Neues Modell hat den Validierungs-Gate nicht bestanden; aktives Modell bleibt unverändert.",
                samples=len(data),
                accuracy=holdout_acc,
                roc_auc=holdout_auc,
                log_loss_value=holdout_ll,
            )

        version = self._next_version(asset, interval)
        model_path = self.models_dir / (
            f"{self._safe(asset)}__{self._safe(interval)}__v{version}.joblib"
        )
        payload = {
            "model": candidate,
            "features": FEATURES,
            "asset": asset,
            "interval": interval,
            "horizon": self.horizon,
            "trained_at": self._utc_now(),
            "training_samples": len(train),
            "walk_forward": wf,
        }
        joblib.dump(payload, model_path)

        now = self._utc_now()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO model_registry
                (asset, interval, version, model_path, trained_at, sample_count,
                 accuracy, roc_auc, log_loss, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    asset, interval, version, str(model_path), now, len(train),
                    holdout_acc, holdout_auc, holdout_ll,
                ),
            )
            con.execute(
                """
                UPDATE model_registry
                SET status='archived'
                WHERE asset=? AND interval=? AND version<>?
                """,
                (asset, interval, version),
            )
            con.execute(
                """
                INSERT INTO state(asset, interval, last_seen_timestamp,
                                   last_trained_sample_count, active_version, updated_at)
                VALUES (?, ?, NULL, ?, ?, ?)
                ON CONFLICT(asset, interval) DO UPDATE SET
                    last_trained_sample_count=excluded.last_trained_sample_count,
                    active_version=excluded.active_version,
                    updated_at=excluded.updated_at
                """,
                (asset, interval, len(data), version, now),
            )

        return TrainResult(
            asset, interval, "activated",
            f"Modell v{version} wurde als gemeinsames Modell aktiviert.",
            version=version,
            samples=len(data),
            accuracy=holdout_acc,
            roc_auc=holdout_auc,
            log_loss_value=holdout_ll,
        )

    def load_active_model(self, asset: str, interval: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT model_path, version, trained_at
                FROM model_registry
                WHERE asset=? AND interval=? AND status='active'
                ORDER BY version DESC LIMIT 1
                """,
                (asset, interval),
            ).fetchone()

        if not row:
            return None
        path = Path(row["model_path"])
        if not path.exists():
            return None

        payload = joblib.load(path)
        payload["version"] = int(row["version"])
        payload["trained_at"] = row["trained_at"]
        return payload

    def predict_latest(self, df: pd.DataFrame, asset: str, interval: str) -> dict[str, Any] | None:
        payload = self.load_active_model(asset, interval)
        if payload is None:
            return None

        data = self.build_features(df)
        latest = data.iloc[[-1]].copy()
        if latest[FEATURES].isna().any(axis=1).iloc[0]:
            return None

        model = payload["model"]
        probability_up = float(model.predict_proba(latest[FEATURES])[0, 1])
        return {
            "probability_up": probability_up,
            "probability_down": 1 - probability_up,
            "version": payload["version"],
            "trained_at": payload["trained_at"],
            "samples": payload.get("training_samples"),
        }

    @staticmethod
    def _safe(value: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def acquire_training_lock(lock_dir: str | Path = "learning_store/.train_lock", stale_after_seconds: int = 3600) -> bool:
    path = Path(lock_dir)
    try:
        path.mkdir(parents=True)
        (path / "meta.json").write_text(
            json.dumps({"pid": os.getpid(), "created_at": time.time()}),
            encoding="utf-8",
        )
        return True
    except FileExistsError:
        meta = path / "meta.json"
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            if time.time() - float(info.get("created_at", 0)) > stale_after_seconds:
                for child in path.iterdir():
                    child.unlink(missing_ok=True)
                path.rmdir()
                return acquire_training_lock(lock_dir, stale_after_seconds)
        except Exception:
            pass
        return False


def release_training_lock(lock_dir: str | Path = "learning_store/.train_lock") -> None:
    path = Path(lock_dir)
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()
