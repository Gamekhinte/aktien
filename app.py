# ============================================================
# 📊 Crypto & Stock Pattern Analyzer – app.py
# ============================================================
# requirements.txt (Streamlit Cloud):
#   streamlit
#   yfinance
#   pandas
#   numpy
#   google-genai
#   plotly
#   scikit-learn
#   supabase
#
# Start lokal: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import io
import os
import json
import itertools
import base64
import pickle
import hashlib
import urllib.request
import urllib.error
from datetime import datetime, time
from zoneinfo import ZoneInfo

try:
    from broker_feed import IBKRFeed  # type: ignore[import-not-found]
except ImportError:
    IBKRFeed = None

try:
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from supabase import create_client, Client
    SUPABASE_PACKAGE_AVAILABLE = True
except ImportError:
    create_client = None
    Client = None
    SUPABASE_PACKAGE_AVAILABLE = False

# ------------------------------------------------------------
# Persistentes gemeinsames Lernsystem (Supabase)
# ------------------------------------------------------------
LEARNING_FEATURE_VERSION = "v3"
AUTO_TRAIN_MIN_NEW_EXAMPLES = 100
MODEL_ALGORITHM = "GradientBoostingClassifier"

# Single source of truth for which features the model sees and how the
# label is constructed. Both build_ml_features() (used when saving examples)
# and train_and_maybe_promote_shared_model() (used when training) import
# these from here, so they can no longer silently drift apart like before.
ML_FEATURE_COLS = [
    "rsi", "macd_hist_norm", "bb_pos", "ema_gap",
    "ret_1", "ret_5", "ret_10", "vol_ratio", "body_ratio",
    "atr_norm", "trend_strength",
]
ML_LABEL_ATR_MULT = 1.5       # Stop-Distanz = ATR * dieser Faktor
ML_LABEL_REWARD_RISK = 2.0    # Zielgewinn = Stop-Distanz * dieser Faktor
ML_LABEL_MAX_HORIZON = 10     # Max. Kerzen, die auf ein Stop/Ziel-Ereignis gewartet wird
ML_COST_R_PER_TRADE = 0.10    # Angenommene Kosten (Fees+Slippage) in Risiko-Einheiten (R) pro Trade

def asset_class_for_symbol(symbol: str) -> str:
    """Grobe, aber wirksame Trennung: Krypto- und Aktienkurse verhalten sich
    fundamental unterschiedlich (24/7 vs. Handelszeiten, Volatilitätsregime).
    Ein einziges gemeinsames Modell für beide verwischt reale Muster."""
    return "crypto" if str(symbol).upper().endswith("-USD") else "stock"

@st.cache_resource(show_spinner=False)
def get_supabase_client():
    if not SUPABASE_PACKAGE_AVAILABLE:
        return None
    try:
        url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    except Exception:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

def learning_db_ready() -> bool:
    return get_supabase_client() is not None

def get_supabase_diagnosis() -> dict:
    """Explains WHY the Supabase connection is or isn't working, instead of
    just returning None like get_supabase_client() does."""
    if not SUPABASE_PACKAGE_AVAILABLE:
        return {"ok": False, "reason": "Das 'supabase'-Paket ist nicht installiert. Prüfe requirements.txt."}
    try:
        url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    except Exception:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url:
        return {"ok": False, "reason": "SUPABASE_URL fehlt in den Streamlit-Secrets."}
    if not key:
        return {"ok": False, "reason": "SUPABASE_SERVICE_ROLE_KEY fehlt in den Streamlit-Secrets."}
    try:
        client = create_client(url, key)
    except Exception as exc:
        return {"ok": False, "reason": f"create_client() ist fehlgeschlagen: {exc}"}
    try:
        client.table("learning_examples").select("id").limit(1).execute()
    except Exception as exc:
        return {"ok": False, "reason": f"Verbindung steht, aber Zugriff auf 'learning_examples' schlägt fehl: {exc}"}
    return {"ok": True, "reason": "Verbindung und Tabellenzugriff funktionieren."}

def _event_key(symbol: str, interval_key: str, timestamp) -> str:
    raw = f"{symbol}|{interval_key}|{LEARNING_FEATURE_VERSION}|{timestamp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def save_learning_examples(symbol: str, interval_key: str, feature_df: pd.DataFrame, feature_cols: list[str]) -> tuple[int, list[str]]:
    """Returns (inserted_count, error_messages). Never silently hides a
    failed insert anymore - if 0 rows land in Supabase, you'll see why."""
    client = get_supabase_client()
    if client is None:
        return 0, ["Kein Supabase-Client verfügbar (Secrets fehlen oder Verbindung schlägt fehl)."]
    usable = feature_df.dropna(subset=feature_cols + ["target"]).copy()
    if usable.empty:
        return 0, ["Nach Feature-Berechnung blieben 0 verwertbare Zeilen übrig (zu wenig Historie oder zu viele NaNs)."]
    asset_class = asset_class_for_symbol(symbol)
    rows = []
    for idx, row in usable.iterrows():
        timestamp = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        features = {name: float(row[name]) for name in feature_cols}
        future_return = row.get("future_return")
        rows.append({
            "symbol": symbol,
            "interval_key": interval_key,
            "asset_class": asset_class,
            "feature_version": LEARNING_FEATURE_VERSION,
            "features": features,
            "target": int(row["target"]),
            "future_return": float(future_return) if pd.notna(future_return) else None,
            "label_time": timestamp,
            "source": "paper",
            "event_key": _event_key(symbol, interval_key, timestamp),
        })
    inserted = 0
    errors: list[str] = []
    for start in range(0, len(rows), 500):
        batch = rows[start:start + 500]
        try:
            result = client.table("learning_examples").upsert(batch, on_conflict="event_key").execute()
            inserted += len(result.data or [])
        except Exception as exc:
            # Ein einzelner fehlerhafter Batch soll das Training nicht komplett
            # blockieren - aber der Fehler wird jetzt gesammelt statt versteckt.
            errors.append(f"Batch {start}-{start + len(batch)}: {exc}")
            continue
    if inserted == 0 and not errors:
        errors.append(
            "0 neue Zeilen gespeichert - vermutlich existieren diese Kerzen (event_key) "
            "schon in Supabase (z.B. weil du dasselbe Asset/Intervall vorher schon geklickt hast)."
        )
    return inserted, errors

def load_learning_examples(asset_class: str | None = None) -> pd.DataFrame:
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()
    rows = []
    try:
        offset = 0
        while True:
            query = (
                client.table("learning_examples")
                .select("features,target,symbol,interval_key,asset_class,created_at")
                .eq("feature_version", LEARNING_FEATURE_VERSION)
            )
            if asset_class:
                query = query.eq("asset_class", asset_class)
            result = query.order("created_at", desc=False).range(offset, offset + 999).execute()
            batch = result.data or []
            rows.extend(batch)
            if len(batch) < 1000:
                break
            offset += 1000
            if offset >= 50000:
                break
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    records = []
    for item in rows:
        record = dict(item.get("features") or {})
        record["target"] = int(item["target"])
        record["symbol"] = item.get("symbol")
        record["interval_key"] = item.get("interval_key")
        record["asset_class"] = item.get("asset_class")
        records.append(record)
    return pd.DataFrame(records)

def _serialize_model(model) -> str:
    return base64.b64encode(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")

def _deserialize_model(payload: str):
    return pickle.loads(base64.b64decode(payload.encode("ascii")))

def get_learning_status(asset_class: str | None = None) -> dict:
    client = get_supabase_client()
    if client is None:
        return {"ready": False, "examples": 0, "active_model": None, "last_training": None}
    try:
        state_key = f"class:{asset_class}" if asset_class else "global"
        state = client.table("learning_state").select("value").eq("key", state_key).maybe_single().execute().data
        value = (state or {}).get("value") or {}
        count_query = client.table("learning_examples").select("id", count="exact").eq("feature_version", LEARNING_FEATURE_VERSION)
        if asset_class:
            count_query = count_query.eq("asset_class", asset_class)
        count = count_query.execute().count or 0
        return {
            "ready": True,
            "examples": int(count),
            "active_model": value.get("active_model_id"),
            "last_training": value.get("last_training_at"),
        }
    except Exception:
        return {"ready": True, "examples": 0, "active_model": None, "last_training": None}

def _set_learning_state(asset_class: str | None = None, **updates):
    client = get_supabase_client()
    if client is None:
        return
    state_key = f"class:{asset_class}" if asset_class else "global"
    try:
        current = client.table("learning_state").select("value").eq("key", state_key).maybe_single().execute().data
        value = dict((current or {}).get("value") or {})
        value.update(updates)
        client.table("learning_state").upsert({"key": state_key, "value": value}).execute()
    except Exception:
        pass

def _latest_promoted_model(asset_class: str):
    client = get_supabase_client()
    if client is None:
        return None
    try:
        result = (
            client.table("model_versions")
            .select("*")
            .eq("promoted", True)
            .eq("asset_class", asset_class)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return (result.data or [None])[0]
    except Exception:
        return None

def _expectancy_r(win_rate: float, reward_risk: float = ML_LABEL_REWARD_RISK, cost_r: float = ML_COST_R_PER_TRADE) -> float:
    """Erwartungswert pro Trade in R (Vielfachen des Stop-Risikos), nach
    Abzug angenommener Kosten. win_rate * reward_risk ist der erwartete
    Gewinn, (1-win_rate) der erwartete Verlust (1R), minus Kosten pro Trade.
    Nur wenn das > 0 ist, verdient die Strategie im Mittel Geld - eine reine
    Trefferquote über der Basisrate reicht dafür nicht automatisch aus."""
    return win_rate * reward_risk - (1.0 - win_rate) - cost_r

def train_and_maybe_promote_shared_model(asset_class: str) -> dict:
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn fehlt."}
    client = get_supabase_client()
    if client is None:
        return {"error": "Supabase ist noch nicht verbunden. Hinterlege SUPABASE_URL und SUPABASE_SERVICE_ROLE_KEY in den Streamlit-Secrets."}

    data = load_learning_examples(asset_class=asset_class)
    feature_cols = ML_FEATURE_COLS
    usable = data.dropna(subset=feature_cols + ["target"]).copy() if not data.empty else pd.DataFrame()
    if len(usable) < 200:
        return {"error": f"Noch zu wenig Trainingsdaten für '{asset_class}': {len(usable)}/200."}

    run = client.table("training_runs").insert({"status": "running", "sample_count": int(len(usable))}).execute()
    run_id = (run.data or [{}])[0].get("id")
    try:
        n_folds = 5
        fold_size = len(usable) // (n_folds + 1)
        fold_accuracies = []
        fold_expectancies = []
        fold_details = []
        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            test_end = fold_size * (fold + 2)
            train_slice = usable.iloc[:train_end]
            test_slice = usable.iloc[train_end:test_end]
            if len(train_slice) < 50 or len(test_slice) < 10:
                continue
            model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(train_slice[feature_cols], train_slice["target"])
            preds = model.predict(test_slice[feature_cols])
            accuracy = float((preds == test_slice["target"].values).mean())
            # Erwartungswert nur über die Kerzen, in denen das Modell tatsächlich
            # "long" gesagt hätte (preds==1) - das entspricht dem echten Handeln,
            # nicht der reinen Klassifikations-Trefferquote über alle Kerzen.
            longs = test_slice[preds == 1]
            long_win_rate = float((longs["target"] == 1).mean()) if len(longs) else None
            expectancy = _expectancy_r(long_win_rate) if long_win_rate is not None else None
            fold_accuracies.append(accuracy)
            if expectancy is not None:
                fold_expectancies.append(expectancy)
            fold_details.append({
                "fold": fold + 1, "train_size": len(train_slice), "test_size": len(test_slice),
                "accuracy": round(accuracy * 100, 1),
                "long_signals": int(len(longs)),
                "long_win_rate": round(long_win_rate * 100, 1) if long_win_rate is not None else None,
                "expectancy_r": round(expectancy, 3) if expectancy is not None else None,
            })
        if not fold_accuracies:
            raise RuntimeError("Keine gültigen Walk-Forward-Folds möglich.")

        new_accuracy = float(np.mean(fold_accuracies))
        mean_expectancy = float(np.mean(fold_expectancies)) if fold_expectancies else None
        baseline = float(usable["target"].mean())
        final_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
        final_model.fit(usable[feature_cols], usable["target"])
        metrics = {
            "mean_accuracy": round(new_accuracy * 100, 2),
            "mean_expectancy_r": round(mean_expectancy, 3) if mean_expectancy is not None else None,
            "baseline_up_rate": round(baseline * 100, 2),
            "fold_details": fold_details,
            "sample_size": int(len(usable)),
            "asset_class": asset_class,
            "feature_importances": dict(zip(feature_cols, final_model.feature_importances_.round(4))),
        }
        previous = _latest_promoted_model(asset_class)
        previous_accuracy = None
        previous_expectancy = None
        if previous:
            prev_metrics = previous.get("validation_metrics") or {}
            previous_accuracy = prev_metrics.get("mean_accuracy")
            previous_expectancy = prev_metrics.get("mean_expectancy_r")

        # Ein neues Modell wird nur aktiviert, wenn es:
        #  1) nach Kosten im Mittel positiv erwartbar ist (expectancy_r > 0), UND
        #  2) das bisher aktive Modell nicht verschlechtert.
        # Reine Trefferquote über der Basisrate reicht NICHT mehr - ein Modell,
        # das zwar "genauer als Münzwurf" ist, aber nach Fees/Slippage trotzdem
        # verliert, wird jetzt bewusst verworfen statt live geschaltet.
        has_positive_expectancy = mean_expectancy is not None and mean_expectancy > 0
        not_worse_than_active = (
            previous is None
            or previous_expectancy is None
            or mean_expectancy is None
            or mean_expectancy >= float(previous_expectancy)
        )
        promote = has_positive_expectancy and not_worse_than_active
        if not has_positive_expectancy:
            rejection_reason = (
                f"Erwartungswert nach Kosten {mean_expectancy:.3f}R <= 0 - "
                f"Modell würde im Schnitt Geld verlieren, egal wie 'genau' es klingt."
                if mean_expectancy is not None else
                "Modell hat nie ein Long-Signal ausgegeben; Erwartungswert nicht bestimmbar."
            )
        else:
            rejection_reason = f"Erwartungswert {mean_expectancy:.3f}R < aktives Modell {float(previous_expectancy):.3f}R"

        model_row = {
            "training_run_id": run_id,
            "algorithm": MODEL_ALGORITHM,
            "feature_version": LEARNING_FEATURE_VERSION,
            "asset_class": asset_class,
            "sample_count": int(len(usable)),
            "validation_metrics": metrics,
            "model_artifact_base64": _serialize_model(final_model),
            "promoted": bool(promote),
            "rejection_reason": None if promote else rejection_reason,
        }
        if promote and previous:
            client.table("model_versions").update({"promoted": False}).eq("promoted", True).eq("asset_class", asset_class).execute()
        model_insert = client.table("model_versions").insert(model_row).execute()
        model_id = (model_insert.data or [{}])[0].get("id")
        now = datetime.now(ZoneInfo("UTC")).isoformat()
        client.table("training_runs").update({"status": "completed" if promote else "rejected", "finished_at": now, "metrics": metrics}).eq("id", run_id).execute()
        if promote:
            _set_learning_state(asset_class, active_model_id=model_id, examples_seen=int(len(usable)), last_training_at=now, feature_version=LEARNING_FEATURE_VERSION)
        return {"ok": True, "promoted": promote, "metrics": metrics, "previous_accuracy": previous_accuracy, "previous_expectancy": previous_expectancy, "model_id": model_id, "rejection_reason": None if promote else rejection_reason}
    except Exception as exc:
        client.table("training_runs").update({"status": "failed", "finished_at": datetime.now(ZoneInfo("UTC")).isoformat(), "error_message": str(exc)}).eq("id", run_id).execute()
        return {"error": f"Training fehlgeschlagen: {exc}"}

# ------------------------------------------------------------
# Seiten-Konfiguration (Handy-optimiert)
# ------------------------------------------------------------
_icon_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
st.set_page_config(
    page_title="Pattern Analyzer",
    page_icon=_icon_full_path if os.path.exists(_icon_full_path) else "📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.6em; padding-bottom: 3em; max-width: 600px; }

    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    }
    body, .stApp { background-color: #171c28 !important; color: #e0e5ef; }

    .app-header { margin-bottom: 1.6em; display: flex; align-items: center; gap: 0.6em; }
    .app-brand { text-align: center; margin: 0 auto 1.7em; }
    .app-title {
        font-size: 1.55em;
        font-weight: 700;
        letter-spacing: 0;
        line-height: 1.15;
        color: #ffffff;
        margin-bottom: 0.15em;
    }
    .app-subtitle { color: #aeb8c9; font-size: 0.9em; line-height: 1.35; }

    .section-label {
        font-size: 0.74em;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #aeb8c9;
        margin: 1.5em 0 0.5em 2px;
    }
    .pattern-book-heading {
        display: flex;
        align-items: center;
        gap: 0.65em;
        margin: 1.25em 0 0.25em;
    }
    .pattern-book-icon { font-size: 1.65em; line-height: 1; }
    .pattern-book-heading-title { color: #ffffff; font-size: 1.08em; font-weight: 750; line-height: 1.25; }
    .pattern-book-heading-subtitle { color: #aeb8c9; font-size: 0.76em; line-height: 1.35; margin-top: 0.18em; }

    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] [role="combobox"] {
        border-radius: 10px !important;
        border: 1px solid #3b4354 !important;
        background: #252b39 !important;
        background-color: #252b39 !important;
        color: #e0e5ef !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"] {
        background: #252b39 !important;
        background-color: #252b39 !important;
        color: #e0e5ef !important;
    }
    div[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"],
    div[data-testid="stSelectbox"] div[role="group"] {
        background: #252b39 !important;
        background-color: #252b39 !important;
        border: 1px solid #4a566c !important;
        border-radius: 10px !important;
    }
    div[data-testid="stSelectbox"] input[role="combobox"] {
        background: transparent !important;
        color: #e0e5ef !important;
        -webkit-text-fill-color: #e0e5ef !important;
    }
    div[data-baseweb="select"] * {
        color: #e0e5ef !important;
    }
    .stTextInput input {
        border-radius: 10px !important;
        border: 1px solid #3b4354 !important;
        background-color: #252b39 !important;
        color: #e0e5ef !important;
    }
    .stTextInput > div > div,
    div[data-baseweb="input"],
    .stNumberInput input,
    .stNumberInput > div > div {
        background-color: #252b39 !important;
        border: 1px solid #4a566c !important;
        color: #e0e5ef !important;
    }
    .stTextInput input::placeholder {
        color: #c8d1df !important;
        opacity: 1 !important;
    }
    .stTextInput label,
    div[data-testid="stExpander"] label {
        color: #dce4f0 !important;
    }
    div[data-testid="stExpander"] {
        border-color: #4a566c !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #252b39 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #222938 !important;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid #3b4354 !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px !important;
        color: #aeb8c9 !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        background-color: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #35405a !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs button[data-baseweb="tab"]:focus { outline: none !important; box-shadow: none !important; }
    button:focus, button:focus-visible { outline: none !important; box-shadow: none !important; }

    .stButton button {
        width: 100%;
        height: 3.3em;
        font-size: 1.02em;
        font-weight: 650;
        border-radius: 10px;
        border: 1px solid #3b4354;
        background: #252b39;
        color: #ffffff;
        margin-top: 0.3em;
        transition: opacity 0.15s ease;
    }
    .stButton button:hover,
    .stButton button:focus,
    .stButton button:active {
        background: #252b39 !important;
        border-color: #5a6880 !important;
        color: #ffffff !important;
    }
    section[data-testid="stFileUploader"] button {
        background: #252b39 !important;
        border: 1px solid #3b4354 !important;
        color: #e0e5ef !important;
    }
    .stButton button:active { opacity: 0.7; }
    .stButton button:disabled { opacity: 0.35; }

    .cta-btn .stButton button {
        background: #2962ff;
        border: none;
        color: #ffffff;
        font-weight: 700;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.4em 1.3em;
        margin-top: 1.2em;
        background: #252b39;
        border: 1px solid #3b4354;
    }
    .pattern-name { font-size: 1em; font-weight: 650; color: #ffffff; margin-bottom: 0.2em; }
    .pattern-meta { font-size: 0.78em; color: #aeb8c9; margin-bottom: 1.1em; }

    .prob-box {
        padding: 0.7em;
        border-radius: 10px;
        text-align: center;
        font-size: 1.7em;
        font-weight: 750;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
    }
    .neutral-box { background-color: #30384a; color: #b8c2d3; font-size: 1.15em; padding: 0.55em; }

    .chart-card {
        border-radius: 14px;
        padding: 0.9em 0.6em 0.3em 0.6em;
        margin-top: 0.9em;
        background: #252b39;
        border: 1px solid #3b4354;
    }
    .green-box { background-color: rgba(61,214,176,0.18); color: #3dd6b0; }
    .red-box   { background-color: rgba(255,107,107,0.18); color: #ff6b6b; }

    .mini-card {
        border-radius: 12px;
        padding: 0.9em 1.1em;
        margin-top: 0.7em;
        background: #252b39;
        border: 1px solid #3b4354;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .mini-ticker { font-weight: 650; font-size: 0.98em; color: #ffffff; }
    .mini-pattern { font-size: 0.78em; color: #aeb8c9; }
    .mini-prob {
        font-size: 1.2em;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        border-radius: 999px;
        padding: 0.3em 0.75em;
    }

    .ai-card {
        border-radius: 12px;
        padding: 1.05em 1.2em;
        margin-top: 0.8em;
        background: #252b39;
        border: 1px solid #3b4354;
        font-size: 0.9em;
        line-height: 1.55em;
        color: #e0e5ef;
    }
    .ai-label { font-size: 0.72em; font-weight: 700; color: #2962ff; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5em; }

    .info-card {
        border-radius: 12px;
        padding: 1em 1.15em;
        margin: 0.8em 0;
        background: #252b39;
        border: 1px solid #3b4354;
        font-size: 0.84em;
        line-height: 1.5em;
        color: #c5ccda;
    }

    .trade-card {
        border-radius: 14px;
        padding: 1.15em 1.2em;
        margin-top: 0.9em;
        background: #252b39;
        border: 1px solid #3b4354;
    }
    .trade-title { font-size: 1em; font-weight: 700; color: #ffffff; margin-bottom: 0.25em; }
    .trade-subtitle { font-size: 0.78em; color: #aeb8c9; margin-bottom: 0.9em; }
    .trade-signal { font-size: 1.35em; font-weight: 750; margin-bottom: 0.7em; }
    .trade-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.55em; }
    .trade-metric { background: #30384a; border-radius: 9px; padding: 0.65em 0.75em; }
    .trade-metric-label { display: block; font-size: 0.72em; color: #b8c2d3; }
    .trade-metric-value { display: block; font-size: 0.98em; font-weight: 650; color: #ffffff; margin-top: 0.15em; }
    .trade-note { font-size: 0.78em; line-height: 1.45em; color: #c5ccda; margin-top: 0.85em; }

    .pattern-book-card {
        border-radius: 12px;
        padding: 0.85em 0.9em;
        margin: 0.55em 0;
        background: #252b39;
        border: 1px solid #3b4354;
    }
    .pattern-book-title { font-size: 0.95em; font-weight: 700; color: #ffffff; }
    .pattern-book-direction { font-size: 0.72em; font-weight: 700; color: #26a69a; text-transform: uppercase; }
    .pattern-book-direction.bearish { color: #ef5350; }
    .pattern-book-direction.neutral { color: #ffd60a; }
    .pattern-book-text { font-size: 0.8em; line-height: 1.45em; color: #c5ccda; margin-top: 0.5em; }
    .pattern-visual { display: flex; align-items: center; justify-content: center; gap: 0.65em; height: 66px; margin: 0.25em 0 0.45em; background: #171a23; border-radius: 8px; }
    .candle { position: relative; width: 18px; height: 54px; }
    .candle-wick { position: absolute; left: 8px; top: 2px; width: 2px; height: 50px; background: #c5ccda; border-radius: 2px; }
    .candle-body { position: absolute; left: 3px; top: 14px; width: 12px; height: 27px; border-radius: 2px; background: #26a69a; border: 1px solid #26a69a; }
    .candle.bear .candle-body { background: #ef5350; border-color: #ef5350; }
    .candle.small .candle-body { top: 24px; height: 11px; }
    .candle.long .candle-body { top: 7px; height: 41px; }
    .candle.doji .candle-body { top: 27px; height: 3px; background: #ffd60a; border-color: #ffd60a; }
    .candle.dragonfly .candle-body { top: 8px; height: 3px; background: #ffd60a; border-color: #ffd60a; }
    .candle.gravestone .candle-body { top: 43px; height: 3px; background: #ffd60a; border-color: #ffd60a; }

    .watch-chip {
        display: inline-block;
        background: #252b39;
        border: 1px solid #3b4354;
        border-radius: 999px;
        padding: 0.4em 0.95em;
        margin: 0.2em 0.3em 0.2em 0;
        font-size: 0.85em;
        color: #ffffff;
        font-variant-numeric: tabular-nums;
    }

    .disclaimer { font-size: 0.74em; color: #8996aa; margin-top: 1.6em; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-brand">'
    '<div class="app-title">Pattern Analyzer</div>'
    '<div class="app-subtitle">Candlestick- &amp; Marktanalyse</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Gemeinsame Konfiguration
# ------------------------------------------------------------
ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
}

INTERVAL_CONFIG = {
    "1m":  {"yf_interval": "1m",  "period": "7d",   "resample": None},
    "5m":  {"yf_interval": "5m",  "period": "60d",  "resample": None},
    "15m": {"yf_interval": "15m", "period": "60d",  "resample": None},
    "30m": {"yf_interval": "30m", "period": "60d",  "resample": None},
    "1h":  {"yf_interval": "1h",  "period": "730d", "resample": None},
    "4h":  {"yf_interval": "1h",  "period": "730d", "resample": "4h"},
    "1d":  {"yf_interval": "1d",  "period": "5y",   "resample": None},
    "1wk": {"yf_interval": "1wk", "period": "10y",  "resample": None},
    "1mo": {"yf_interval": "1mo", "period": "max",  "resample": None},
}

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = ""

if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "single_ai_text" not in st.session_state:
    st.session_state.single_ai_text = None

if "single_ai_key" not in st.session_state:
    st.session_state.single_ai_key = None

if "ibkr_feed" not in st.session_state:
    st.session_state.ibkr_feed = None

if "paper_bot_result" not in st.session_state:
    st.session_state.paper_bot_result = None

if "paper_bot_best_params" not in st.session_state:
    st.session_state.paper_bot_best_params = None

if "ml_result" not in st.session_state:
    st.session_state.ml_result = None

# ------------------------------------------------------------
# API-Key-Eingabe
# ------------------------------------------------------------
has_any_ai_key = any((st.session_state.gemini_api_key, st.session_state.openai_api_key, st.session_state.anthropic_api_key))
key_label = "KI-Anbieter und API-Keys" if not has_any_ai_key else "KI-Anbieter und API-Keys (gesetzt)"
with st.expander(key_label, expanded=not has_any_ai_key):
    ai_provider = st.selectbox(
        "KI-Anbieter",
        ["Keiner", "Gemini", "OpenAI", "Claude"],
        key="ai_provider",
        help="Nur der ausgewählte Anbieter wird bei einem Klick kontaktiert.",
    )
    if ai_provider == "Gemini":
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key", type="password", value=st.session_state.gemini_api_key,
            placeholder="Gemini-Key hier einfügen",
            key="main_api_key",
        )
    elif ai_provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key", type="password", value=st.session_state.openai_api_key,
            placeholder="OpenAI-Key hier einfügen",
            key="openai_api_key_input",
        )
    elif ai_provider == "Claude":
        st.session_state.anthropic_api_key = st.text_input(
            "Claude API Key", type="password", value=st.session_state.anthropic_api_key,
            placeholder="Claude-Key hier einfügen",
            key="anthropic_api_key_input",
        )
    else:
        st.caption("Wähle zuerst einen KI-Anbieter aus.")
    st.caption("Bleibt nur für diese Sitzung gespeichert (Session State), nicht dauerhaft.")

with st.expander("Marktdatenquelle", expanded=False):
    data_source = st.selectbox(
        "Kursquelle",
        ["Yahoo Finance", "Interactive Brokers Paper-Feed"],
        key="data_source",
        help="Der IBKR-Feed liest nur Paper-Marktdaten und platziert keine Orders.",
    )
    if data_source == "Interactive Brokers Paper-Feed":
        ibkr_host = st.text_input("IBKR Host", value="127.0.0.1", key="ibkr_host")
        ibkr_port = st.number_input("IBKR Paper-Port", min_value=1, value=7497, step=1, key="ibkr_port")
        if IBKRFeed is None:
            st.warning("Für den IBKR-Feed fehlt noch 'ib_insync'. Installiere es mit: python -m pip install ib_insync")
        elif st.button("Paper-Feed verbinden", key="ibkr_connect"):
            try:
                st.session_state.ibkr_feed = IBKRFeed(ibkr_host, int(ibkr_port))
                st.session_state.ibkr_feed.connect()
                st.success("IBKR-Paper-Feed verbunden. Es werden keine Orders platziert.")
            except Exception as error:
                st.session_state.ibkr_feed = None
                st.error(f"IBKR-Verbindung fehlgeschlagen: {error}")

# ------------------------------------------------------------
# Daten laden
# ------------------------------------------------------------
def load_data(ticker: str, interval_key: str, source: str = "Yahoo Finance", period_override: str | None = None, max_candles: int = 1000) -> pd.DataFrame:
    if source == "Interactive Brokers Paper-Feed":
        feed = st.session_state.get("ibkr_feed")
        if feed is None or not feed.connected:
            raise RuntimeError("IBKR-Paper-Feed ist nicht verbunden. TWS/IB Gateway starten und Paper-Feed verbinden.")
        return feed.fetch_bars(ticker, interval_key)

    cfg = INTERVAL_CONFIG[interval_key]
    period = period_override or cfg["period"]
    df = yf.download(ticker, period=period, interval=cfg["yf_interval"], progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

    df = df.dropna()
    if max_candles:
        df = df.tail(max_candles)
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    return df

# ------------------------------------------------------------
# Candlestick-Muster-Erkennung
# ------------------------------------------------------------
def _body(row):
    return abs(row["Close"] - row["Open"])

def _range(row):
    return row["High"] - row["Low"]

def is_hammer(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    return (lower_wick > 2 * body) and (upper_wick < body) and (body / range_ < 0.35)

def is_shooting_star(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return (upper_wick > 2 * body) and (lower_wick < body) and (body / range_ < 0.35)

def is_doji(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    return body / range_ < 0.08

def is_bullish_engulfing(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and row["Open"] <= prev["Close"] and row["Close"] >= prev["Open"]
    )

def is_bearish_engulfing(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and row["Open"] >= prev["Close"] and row["Close"] <= prev["Open"]
    )

def is_bullish_harami(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and _body(prev) > _body(row) * 1.5
        and row["Open"] >= prev["Close"] and row["Close"] <= prev["Open"]
    )

def is_bearish_harami(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and _body(prev) > _body(row) * 1.5
        and row["Open"] <= prev["Close"] and row["Close"] >= prev["Open"]
    )

def is_piercing_line(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    midpoint = (prev["Open"] + prev["Close"]) / 2
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and row["Open"] <= prev["Close"] and row["Close"] > midpoint
        and row["Close"] < prev["Open"]
    )

def is_dark_cloud_cover(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    midpoint = (prev["Open"] + prev["Close"]) / 2
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and row["Open"] >= prev["Close"] and row["Close"] < midpoint
        and row["Close"] > prev["Open"]
    )

def is_hanging_man(df, i) -> bool:
    if i < 3:
        return False
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    prior_uptrend = df.iloc[i - 1]["Close"] > df.iloc[i - 3]["Close"]
    return prior_uptrend and lower_wick > 2 * body and upper_wick < body and body / range_ < 0.35

def is_inverted_hammer(df, i) -> bool:
    if i < 3:
        return False
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    prior_downtrend = df.iloc[i - 1]["Close"] < df.iloc[i - 3]["Close"]
    return prior_downtrend and upper_wick > 2 * body and lower_wick < body and body / range_ < 0.35

def is_tweezer_bottom(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    tolerance = max(_range(prev), _range(row), 1e-9) * 0.1
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and abs(prev["Low"] - row["Low"]) <= tolerance
    )

def is_tweezer_top(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    tolerance = max(_range(prev), _range(row), 1e-9) * 0.1
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and abs(prev["High"] - row["High"]) <= tolerance
    )

def is_morning_star(df, i) -> bool:
    if i < 2:
        return False
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    a_bear = a["Close"] < a["Open"] and _body(a) / max(_range(a), 1e-9) > 0.4
    b_small = _body(b) / max(_range(b), 1e-9) < 0.35
    c_bull = c["Close"] > c["Open"] and c["Close"] > (a["Open"] + a["Close"]) / 2
    return bool(a_bear and b_small and c_bull)

def is_evening_star(df, i) -> bool:
    if i < 2:
        return False
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    a_bull = a["Close"] > a["Open"] and _body(a) / max(_range(a), 1e-9) > 0.4
    b_small = _body(b) / max(_range(b), 1e-9) < 0.35
    c_bear = c["Close"] < c["Open"] and c["Close"] < (a["Open"] + a["Close"]) / 2
    return bool(a_bull and b_small and c_bear)

def is_three_white_soldiers(df, i) -> bool:
    if i < 2:
        return False
    window = df.iloc[i - 2:i + 1]
    closes_rising = all(window["Close"].iloc[k] > window["Close"].iloc[k - 1] for k in range(1, 3))
    return bool((window["Close"] > window["Open"]).all() and closes_rising)

def is_three_black_crows(df, i) -> bool:
    if i < 2:
        return False
    window = df.iloc[i - 2:i + 1]
    closes_falling = all(window["Close"].iloc[k] < window["Close"].iloc[k - 1] for k in range(1, 3))
    return bool((window["Close"] < window["Open"]).all() and closes_falling)

def is_bullish_marubozu(df, i) -> bool:
    row = df.iloc[i]
    range_ = _range(row)
    if range_ == 0:
        return False
    return row["Close"] > row["Open"] and _body(row) / range_ > 0.9

def is_bearish_marubozu(df, i) -> bool:
    row = df.iloc[i]
    range_ = _range(row)
    if range_ == 0:
        return False
    return row["Close"] < row["Open"] and _body(row) / range_ > 0.9

def is_spinning_top(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return 0.08 <= body / range_ <= 0.3 and upper_wick > body and lower_wick > body

def is_dragonfly_doji(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return body / range_ < 0.08 and lower_wick > 2 * upper_wick

def is_gravestone_doji(df, i) -> bool:
    row = df.iloc[i]
    body, range_ = _body(row), _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return body / range_ < 0.08 and upper_wick > 2 * lower_wick

def is_bullish_belt_hold(df, i) -> bool:
    row = df.iloc[i]
    range_ = _range(row)
    if range_ == 0:
        return False
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return row["Close"] > row["Open"] and _body(row) / range_ > 0.7 and lower_wick / range_ < 0.1

def is_bearish_belt_hold(df, i) -> bool:
    row = df.iloc[i]
    range_ = _range(row)
    if range_ == 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    return row["Close"] < row["Open"] and _body(row) / range_ > 0.7 and upper_wick / range_ < 0.1

def is_three_inside_up(df, i) -> bool:
    if i < 2:
        return False
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    return (
        a["Close"] < a["Open"] and b["Close"] > b["Open"]
        and _body(b) < _body(a) * 0.6
        and b["Open"] >= a["Close"] and b["Close"] <= a["Open"]
        and c["Close"] > a["Open"]
    )

def is_three_inside_down(df, i) -> bool:
    if i < 2:
        return False
    a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    return (
        a["Close"] > a["Open"] and b["Close"] < b["Open"]
        and _body(b) < _body(a) * 0.6
        and b["Open"] <= a["Close"] and b["Close"] >= a["Open"]
        and c["Close"] < a["Open"]
    )

def is_rising_three_methods(df, i) -> bool:
    if i < 4:
        return False
    window = df.iloc[i - 4:i + 1]
    first, last = window.iloc[0], window.iloc[4]
    middle = window.iloc[1:4]
    return (
        first["Close"] > first["Open"] and last["Close"] > last["Open"]
        and (middle["Close"] < middle["Open"]).all()
        and (middle["High"] < first["High"]).all()
        and (middle["Low"] > first["Low"]).all()
        and last["Close"] > first["High"]
    )

def is_falling_three_methods(df, i) -> bool:
    if i < 4:
        return False
    window = df.iloc[i - 4:i + 1]
    first, last = window.iloc[0], window.iloc[4]
    middle = window.iloc[1:4]
    return (
        first["Close"] < first["Open"] and last["Close"] < last["Open"]
        and (middle["Close"] > middle["Open"]).all()
        and (middle["High"] < first["High"]).all()
        and (middle["Low"] > first["Low"]).all()
        and last["Close"] < first["Low"]
    )

def is_bullish_three_line_strike(df, i) -> bool:
    if i < 3:
        return False
    window = df.iloc[i - 3:i + 1]
    first_three = window.iloc[:3]
    last = window.iloc[3]
    return (
        (first_three["Close"] < first_three["Open"]).all()
        and first_three["Close"].iloc[1] < first_three["Close"].iloc[0]
        and first_three["Close"].iloc[2] < first_three["Close"].iloc[1]
        and last["Close"] > last["Open"]
        and last["Open"] <= first_three["Close"].iloc[2]
        and last["Close"] >= first_three["Open"].iloc[0]
    )

def is_bearish_three_line_strike(df, i) -> bool:
    if i < 3:
        return False
    window = df.iloc[i - 3:i + 1]
    first_three = window.iloc[:3]
    last = window.iloc[3]
    return (
        (first_three["Close"] > first_three["Open"]).all()
        and first_three["Close"].iloc[1] > first_three["Close"].iloc[0]
        and first_three["Close"].iloc[2] > first_three["Close"].iloc[1]
        and last["Close"] < last["Open"]
        and last["Open"] >= first_three["Close"].iloc[2]
        and last["Close"] <= first_three["Open"].iloc[0]
    )

def is_bullish_kicker(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] < prev["Open"] and row["Close"] > row["Open"]
        and _body(row) / max(_range(row), 1e-9) > 0.6
        and row["Open"] > prev["Open"]
    )

def is_bearish_kicker(df, i) -> bool:
    if i < 1:
        return False
    prev, row = df.iloc[i - 1], df.iloc[i]
    return (
        prev["Close"] > prev["Open"] and row["Close"] < row["Open"]
        and _body(row) / max(_range(row), 1e-9) > 0.6
        and row["Open"] < prev["Open"]
    )

PATTERNS = {
    "Bullish Engulfing (bullisch)": is_bullish_engulfing,
    "Bearish Engulfing (bärisch)": is_bearish_engulfing,
    "Bullish Harami (bullisch)": is_bullish_harami,
    "Bearish Harami (bärisch)": is_bearish_harami,
    "Piercing Line (bullisch)": is_piercing_line,
    "Dark Cloud Cover (bärisch)": is_dark_cloud_cover,
    "Morning Star (bullisch)": is_morning_star,
    "Evening Star (bärisch)": is_evening_star,
    "3 weiße Soldaten (bullisch)": is_three_white_soldiers,
    "3 schwarze Krähen (bärisch)": is_three_black_crows,
    "Hammer (bullisch)": is_hammer,
    "Hanging Man (bärisch)": is_hanging_man,
    "Inverted Hammer (bullisch)": is_inverted_hammer,
    "Shooting Star (bärisch)": is_shooting_star,
    "Tweezer Bottom (bullisch)": is_tweezer_bottom,
    "Tweezer Top (bärisch)": is_tweezer_top,
    "Doji (Unentschlossenheit)": is_doji,
    "Bullish Marubozu (bullisch)": is_bullish_marubozu,
    "Bearish Marubozu (bärisch)": is_bearish_marubozu,
    "Spinning Top (Unentschlossenheit)": is_spinning_top,
    "Dragonfly Doji (bullisch)": is_dragonfly_doji,
    "Gravestone Doji (bärisch)": is_gravestone_doji,
    "Bullish Belt Hold (bullisch)": is_bullish_belt_hold,
    "Bearish Belt Hold (bärisch)": is_bearish_belt_hold,
    "Three Inside Up (bullisch)": is_three_inside_up,
    "Three Inside Down (bärisch)": is_three_inside_down,
    "Rising Three Methods (bullisch)": is_rising_three_methods,
    "Falling Three Methods (bärisch)": is_falling_three_methods,
    "Bullish Three Line Strike (bullisch)": is_bullish_three_line_strike,
    "Bearish Three Line Strike (bärisch)": is_bearish_three_line_strike,
    "Bullish Kicker (bullisch)": is_bullish_kicker,
    "Bearish Kicker (bärisch)": is_bearish_kicker,
}

PATTERN_EXPLAIN = {
    "Bullish Engulfing (bullisch)": "Eine grüne Kerze schluckt komplett den Körper der vorherigen roten Kerze – die Käufer haben die Kontrolle übernommen.",
    "Bearish Engulfing (bärisch)": "Eine rote Kerze schluckt komplett den Körper der vorherigen grünen Kerze – die Verkäufer haben die Kontrolle übernommen.",
    "Bullish Harami (bullisch)": "Eine kleine grüne Kerze liegt vollständig im Körper der vorherigen großen roten Kerze – mögliches Nachlassen des Verkaufsdrucks.",
    "Bearish Harami (bärisch)": "Eine kleine rote Kerze liegt vollständig im Körper der vorherigen großen grünen Kerze – mögliches Nachlassen des Kaufdrucks.",
    "Piercing Line (bullisch)": "Eine rote Kerze wird von einer grünen Kerze mehr als bis zur Körpermitte zurückerobert – mögliches bullisches Umkehrsignal.",
    "Dark Cloud Cover (bärisch)": "Eine rote Kerze fällt nach starkem Start unter die Körpermitte der vorherigen grünen Kerze – mögliches bearisches Umkehrsignal.",
    "Morning Star (bullisch)": "Dreier-Formation: große rote Kerze, kleine unentschlossene Kerze, dann eine große grüne Kerze – klassisches Bodenbildungsmuster.",
    "Evening Star (bärisch)": "Dreier-Formation: große grüne Kerze, kleine unentschlossene Kerze, dann eine große rote Kerze – klassisches Topbildungsmuster.",
    "3 weiße Soldaten (bullisch)": "Drei aufeinanderfolgende grüne Kerzen mit steigenden Schlusskursen – starker Aufwärtstrend.",
    "3 schwarze Krähen (bärisch)": "Drei aufeinanderfolgende rote Kerzen mit fallenden Schlusskursen – starker Abwärtstrend.",
    "Hammer (bullisch)": "Langer unterer Docht, kleiner Körper oben – Verkäufer drückten den Kurs runter, Käufer haben ihn zurückgeholt.",
    "Hanging Man (bärisch)": "Langer unterer Docht nach einem Anstieg – mögliches Warnsignal für nachlassende Käufer.",
    "Inverted Hammer (bullisch)": "Langer oberer Docht nach einem Abwärtstrend – mögliches erstes Zeichen für eine Bodenbildung.",
    "Shooting Star (bärisch)": "Langer oberer Docht, kleiner Körper unten – Käufer drückten den Kurs hoch, Verkäufer haben ihn zurückgeholt.",
    "Tweezer Bottom (bullisch)": "Zwei Kerzen testen nahezu dasselbe Tief, danach übernehmen die Käufer – mögliches Umkehrsignal.",
    "Tweezer Top (bärisch)": "Zwei Kerzen testen nahezu dasselbe Hoch, danach übernehmen die Verkäufer – mögliches Umkehrsignal.",
    "Doji (Unentschlossenheit)": "Open und Close liegen fast gleich – der Markt ist unentschlossen, oft ein Wendepunkt-Hinweis.",
    "Bullish Marubozu (bullisch)": "Eine lange grüne Kerze mit kaum Dochten zeigt starken, nahezu ununterbrochenen Kaufdruck.",
    "Bearish Marubozu (bärisch)": "Eine lange rote Kerze mit kaum Dochten zeigt starken, nahezu ununterbrochenen Verkaufsdruck.",
    "Spinning Top (Unentschlossenheit)": "Kleiner Körper und Dochte auf beiden Seiten zeigen ein ausgeglichenes Kräftemessen.",
    "Dragonfly Doji (bullisch)": "Open und Close liegen oben, während ein langer unterer Docht eine kräftige Erholung vom Tagestief zeigt.",
    "Gravestone Doji (bärisch)": "Open und Close liegen unten, während ein langer oberer Docht die Zurückweisung höherer Kurse zeigt.",
    "Bullish Belt Hold (bullisch)": "Eine starke grüne Kerze startet nahe dem Tief und schließt deutlich höher – bullischer Impuls.",
    "Bearish Belt Hold (bärisch)": "Eine starke rote Kerze startet nahe dem Hoch und schließt deutlich tiefer – bärischer Impuls.",
    "Three Inside Up (bullisch)": "Rote große Kerze, kleine Kerze innerhalb ihres Körpers und ein Ausbruch nach oben bilden eine mögliche Bodenwende.",
    "Three Inside Down (bärisch)": "Grüne große Kerze, kleine Kerze innerhalb ihres Körpers und ein Ausbruch nach unten bilden eine mögliche Topwende.",
    "Rising Three Methods (bullisch)": "Auf eine lange grüne Kerze folgt eine kurze Gegenbewegung innerhalb ihrer Spanne, danach setzt der Aufwärtstrend fort.",
    "Falling Three Methods (bärisch)": "Auf eine lange rote Kerze folgt eine kurze Gegenbewegung innerhalb ihrer Spanne, danach setzt der Abwärtstrend fort.",
    "Bullish Three Line Strike (bullisch)": "Drei steigende grüne Kerzen werden von einer großen roten Kerze zurückgenommen – mögliches Fortsetzungssignal im Aufwärtstrend.",
    "Bearish Three Line Strike (bärisch)": "Drei fallende rote Kerzen werden von einer großen grünen Kerze zurückgenommen – mögliches Fortsetzungssignal im Abwärtstrend.",
    "Bullish Kicker (bullisch)": "Ein starker Wechsel von einer roten zu einer grünen Kerze mit höherem Start signalisiert abrupten Kaufdruck.",
    "Bearish Kicker (bärisch)": "Ein starker Wechsel von einer grünen zu einer roten Kerze mit niedrigerem Start signalisiert abrupten Verkaufsdruck.",
}

# ------------------------------------------------------------
# Erweiterte Multi-Signal & Indikator-Erkennung
# ------------------------------------------------------------
def detect_pattern(df: pd.DataFrame) -> list[str]:
    if len(df) < 50:
        return ["Zu wenig Daten"]

    i = len(df) - 1
    found_signals = []

    for name, fn in PATTERNS.items():
        if fn(df, i):
            found_signals.append(name)

    close = df["Close"]
    volume = df["Volume"]

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = sma20 + (std20 * 2)
    lower_bb = sma20 - (std20 * 2)

    if macd.iloc[i-1] < signal.iloc[i-1] and macd.iloc[i] > signal.iloc[i]:
        found_signals.append("MACD Bullish Crossover")
    elif macd.iloc[i-1] > signal.iloc[i-1] and macd.iloc[i] < signal.iloc[i]:
        found_signals.append("MACD Bearish Crossover")

    if rsi.iloc[i] < 30:
        found_signals.append("RSI Überverkauft (< 30)")
    elif rsi.iloc[i] > 70:
        found_signals.append("RSI Überhitzt (> 70)")

    if close.iloc[i] < lower_bb.iloc[i]:
        found_signals.append("Unter Bollinger-Band gefallen")
    elif close.iloc[i] > upper_bb.iloc[i]:
        found_signals.append("Über Bollinger-Band ausgebrochen")

    vol_mean = volume.rolling(20).mean()
    if len(volume) >= 20 and volume.iloc[i] > (1.8 * vol_mean.iloc[i]):
        found_signals.append("Hohes Handelsvolumen (Volumen-Spike)")

    if not found_signals:
        sma50 = close.rolling(50).mean()
        if len(close) >= 50 and close.iloc[i] > sma50.iloc[i]:
            found_signals.append("Neutrale Konsolidierung im Aufwärtstrend")
        else:
            found_signals.append("Neutrale Konsolidierung im Abwärtstrend")

    return found_signals

def historical_probability(df: pd.DataFrame, pattern: str) -> tuple[float, int]:
    if pattern not in PATTERNS:
        return None, 0
    fn = PATTERNS[pattern]
    hits, ups = 0, 0
    for i in range(len(df) - 1):
        if fn(df, i):
            hits += 1
            if df.iloc[i + 1]["Close"] > df.iloc[i]["Close"]:
                ups += 1
    if hits == 0:
        return None, 0
    return round(100 * ups / hits, 1), hits

# ------------------------------------------------------------
# Generalisierte Regel-Indikatoren (parametrisierbar für Grid-Search)
# ------------------------------------------------------------
DEFAULT_PARAMS = {
    "ema_fast": 9, "ema_slow": 21, "rsi_period": 14,
    "atr_period": 14, "atr_mult": 1.0, "reward_risk": 2.0,
    "rsi_bull": (50, 70), "rsi_bear": (30, 50),
}

def _trade_indicators(df: pd.DataFrame, ema_fast=9, ema_slow=21, rsi_period=14, atr_period=14) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, 1e-9)
    data["RSI"] = 100 - (100 / (1 + rs))
    data["EMA_FAST"] = close.ewm(span=ema_fast, adjust=False).mean()
    data["EMA_SLOW"] = close.ewm(span=ema_slow, adjust=False).mean()
    previous_close = close.shift(1)
    true_range = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - previous_close).abs(),
        (data["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    data["ATR"] = true_range.rolling(atr_period).mean()
    data["VolumeRatio"] = data["Volume"] / data["Volume"].rolling(20).mean().replace(0, np.nan)
    return data

def _trade_direction(row: pd.Series, rsi_bull=(50, 70), rsi_bear=(30, 50)) -> str:
    if pd.isna(row["RSI"]) or pd.isna(row["ATR"]):
        return "neutral"
    bullish = row["Close"] > row["EMA_FAST"] > row["EMA_SLOW"] and rsi_bull[0] <= row["RSI"] <= rsi_bull[1]
    bearish = row["Close"] < row["EMA_FAST"] < row["EMA_SLOW"] and rsi_bear[0] <= row["RSI"] <= rsi_bear[1]
    if bullish:
        return "long"
    if bearish:
        return "short"
    return "neutral"

def calculate_trade_setup(df: pd.DataFrame, params: dict = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    data = _trade_indicators(df, p["ema_fast"], p["ema_slow"], p["rsi_period"], p["atr_period"])
    row = data.iloc[-1]
    direction = _trade_direction(row, p["rsi_bull"], p["rsi_bear"])
    entry = float(row["Close"])
    atr = float(row["ATR"]) if not pd.isna(row["ATR"]) else 0.0
    stop_distance = atr * p["atr_mult"]
    if direction == "long":
        stop = entry - stop_distance
        target = entry + (p["reward_risk"] * stop_distance)
        signal = "KAUFEN (Long-Setup)"
    elif direction == "short":
        stop = entry + stop_distance
        target = entry - (p["reward_risk"] * stop_distance)
        signal = "VERKAUFEN (Short-Setup)"
    else:
        stop = target = None
        signal = "ABWARTEN"

    wins = losses = 0
    look_ahead = 5
    for index in range(len(data) - look_ahead):
        historical_row = data.iloc[index]
        historical_direction = _trade_direction(historical_row, p["rsi_bull"], p["rsi_bear"])
        historical_atr = historical_row["ATR"]
        if historical_direction == "neutral" or pd.isna(historical_atr) or historical_atr <= 0:
            continue
        historical_entry = float(historical_row["Close"])
        historical_stop_distance = float(historical_atr) * p["atr_mult"]
        if historical_direction == "long":
            historical_stop = historical_entry - historical_stop_distance
            historical_target = historical_entry + (p["reward_risk"] * historical_stop_distance)
        else:
            historical_stop = historical_entry + historical_stop_distance
            historical_target = historical_entry - (p["reward_risk"] * historical_stop_distance)
        for future_index in range(index + 1, index + look_ahead + 1):
            future_row = data.iloc[future_index]
            if historical_direction == "long":
                hit_stop = future_row["Low"] <= historical_stop
                hit_target = future_row["High"] >= historical_target
            else:
                hit_stop = future_row["High"] >= historical_stop
                hit_target = future_row["Low"] <= historical_target
            if hit_stop and hit_target:
                losses += 1
                break
            if hit_target:
                wins += 1
                break
            if hit_stop:
                losses += 1
                break

    total = wins + losses
    win_rate = round(100 * wins / total, 1) if total else None
    return {
        "data": data,
        "direction": direction,
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "target": target,
        "atr": atr,
        "rsi": float(row["RSI"]) if not pd.isna(row["RSI"]) else None,
        "volume_ratio": float(row["VolumeRatio"]) if not pd.isna(row["VolumeRatio"]) else None,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }

def simulate_paper_bot(df: pd.DataFrame, initial_capital: float, risk_percent: float,
                        params: dict = None) -> dict:
    """Testet eine EMA/RSI/ATR-Regel ohne echte Orders oder Look-ahead."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    data = _trade_indicators(
        df, p["ema_fast"], p["ema_slow"], p["rsi_period"], p["atr_period"]
    ).reset_index(drop=True)
    training_end = max(50, int(len(data) * 0.7))
    equity = float(initial_capital)
    position = 0.0
    entry_price = stop_price = target_price = 0.0
    wins = losses = 0
    trades = []

    for index in range(training_end, len(data) - 1):
        row = data.iloc[index]
        next_row = data.iloc[index + 1]
        if position != 0:
            if position > 0:
                hit_stop = row["Low"] <= stop_price
                hit_target = row["High"] >= target_price
                exit_price = stop_price if hit_stop else target_price if hit_target else None
            else:
                hit_stop = row["High"] >= stop_price
                hit_target = row["Low"] <= target_price
                exit_price = stop_price if hit_stop else target_price if hit_target else None
            if exit_price is not None:
                pnl = position * (exit_price - entry_price)
                equity += pnl
                won = pnl > 0
                wins += int(won)
                losses += int(not won)
                trades.append({
                    "Datum": str(row["Date"]) if "Date" in row else str(index),
                    "Richtung": "Long" if position > 0 else "Short",
                    "Entry": round(entry_price, 4), "Exit": round(exit_price, 4),
                    "Ergebnis": round(pnl, 2),
                })
                position = 0.0
                continue

            direction = _trade_direction(row, p["rsi_bull"], p["rsi_bear"])
            if direction != "neutral" and ((position > 0 and direction == "short") or (position < 0 and direction == "long")):
                exit_price = float(next_row["Open"])
                pnl = position * (exit_price - entry_price)
                equity += pnl
                won = pnl > 0
                wins += int(won)
                losses += int(not won)
                trades.append({
                    "Datum": str(next_row["Date"]) if "Date" in next_row else str(index + 1),
                    "Richtung": "Long" if position > 0 else "Short",
                    "Entry": round(entry_price, 4), "Exit": round(exit_price, 4),
                    "Ergebnis": round(pnl, 2),
                })
                position = 0.0
            continue

        direction = _trade_direction(row, p["rsi_bull"], p["rsi_bear"])
        atr = row["ATR"]
        if direction == "neutral" or pd.isna(atr) or atr <= 0 or equity <= 0:
            continue
        entry_price = float(next_row["Open"])
        stop_distance = float(atr) * p["atr_mult"]
        risk_amount = equity * risk_percent / 100
        units = risk_amount / stop_distance if stop_distance else 0
        if direction == "long":
            position = units
            stop_price = entry_price - stop_distance
            target_price = entry_price + p["reward_risk"] * stop_distance
        else:
            position = -units
            stop_price = entry_price + stop_distance
            target_price = entry_price - p["reward_risk"] * stop_distance

    if position != 0:
        exit_price = float(data.iloc[-1]["Close"])
        pnl = position * (exit_price - entry_price)
        equity += pnl
        wins += int(pnl > 0)
        losses += int(pnl <= 0)
        trades.append({
            "Datum": str(data.iloc[-1]["Date"]) if "Date" in data else str(len(data) - 1),
            "Richtung": "Long" if position > 0 else "Short",
            "Entry": round(entry_price, 4), "Exit": round(exit_price, 4),
            "Ergebnis": round(pnl, 2),
        })

    total = wins + losses
    return {
        "params": p,
        "training_end": training_end,
        "final_equity": equity,
        "return_percent": (equity / initial_capital - 1) * 100,
        "wins": wins, "losses": losses,
        "win_rate": 100 * wins / total if total else None,
        "trade_count": total,
        "trades": trades,
    }

def optimize_paper_bot(df: pd.DataFrame, initial_capital: float, risk_percent: float,
                        min_trades: int = 8, min_holdout_trades: int = 3) -> dict:
    """
    Grid-Search über EMA/RSI/ATR/Chance-Risiko-Kombinationen.
    Bewertet wird ausschließlich auf einem Out-of-Sample-Holdout-Zeitraum,
    damit sich die Auswahl nicht einfach an die Vergangenheit anpasst (Overfitting).
    """
    grid = {
        "ema_fast": [5, 9, 12],
        "ema_slow": [21, 26, 34],
        "rsi_period": [10, 14, 21],
        "atr_mult": [1.0, 1.5, 2.0],
        "reward_risk": [1.5, 2.0, 3.0],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    split = int(len(df) * 0.85)
    fit_df = df.iloc[:split].copy()
    holdout_df = df.iloc[max(0, split - 60):].copy()

    best = None
    for combo in combos:
        params = dict(zip(keys, combo))
        if params["ema_fast"] >= params["ema_slow"]:
            continue
        params["rsi_bull"] = (50, 70)
        params["rsi_bear"] = (30, 50)
        params["atr_period"] = 14

        fit_result = simulate_paper_bot(fit_df, initial_capital, risk_percent, params)
        if fit_result["trade_count"] < min_trades:
            continue

        holdout_result = simulate_paper_bot(holdout_df, initial_capital, risk_percent, params)
        if holdout_result["trade_count"] < min_holdout_trades:
            continue

        score = holdout_result["return_percent"]
        if best is None or score > best["score"]:
            best = {"params": params, "score": score, "fit": fit_result, "holdout": holdout_result}

    return best

# ------------------------------------------------------------
# Lernendes ML-Modell (Gradient Boosting, Walk-Forward-Validierung)
# ------------------------------------------------------------
ML_HORIZON = 5  # Kerzen in die Zukunft, deren Richtung vorhergesagt wird

def _triple_barrier_labels(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, atr: np.ndarray,
    atr_mult: float = ML_LABEL_ATR_MULT, reward_risk: float = ML_LABEL_REWARD_RISK,
    max_horizon: int = ML_LABEL_MAX_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Statt "steigt der Kurs in N Kerzen über einen Schwellwert" simulieren wir
    hier für jede Kerze i einen echten Long-Trade mit ATR-Stop und
    Take-Profit im festen Reward:Risk-Verhältnis (wie im Paper-Bot) und
    schauen, welche Barriere zuerst berührt wird:
      - Take-Profit zuerst erreicht  -> target = 1 (Trade wäre profitabel)
      - Stop zuerst erreicht          -> target = 0 (Trade wäre verloren)
      - keine der beiden Barrieren
        innerhalb von max_horizon
        Kerzen erreicht               -> target = NaN (wird verworfen)
    Das bringt das Lernziel näher an das, was beim echten Handeln zählt
    (Trade-Ausgang), statt an eine willkürliche Richtungs-Schwelle.
    """
    n = len(close)
    labels = np.full(n, np.nan)
    realized_return = np.full(n, np.nan)
    for i in range(n - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(close[i]):
            continue
        entry = close[i]
        stop_dist = a * atr_mult
        target_dist = stop_dist * reward_risk
        upper = entry + target_dist
        lower = entry - stop_dist
        end = min(i + 1 + max_horizon, n)
        for j in range(i + 1, end):
            hit_up = high[j] >= upper
            hit_down = low[j] <= lower
            if hit_up and hit_down:
                # Beides in derselben Kerze berührt: keine Reihenfolge bekannt
                # -> konservativ als Verlust werten, nicht als Zufallstreffer.
                labels[i] = 0.0
                realized_return[i] = -stop_dist / entry
                break
            if hit_up:
                labels[i] = 1.0
                realized_return[i] = target_dist / entry
                break
            if hit_down:
                labels[i] = 0.0
                realized_return[i] = -stop_dist / entry
                break
    return labels, realized_return

def build_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    data["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]
    # Normalized by price so the value is comparable across assets with very
    # different nominal prices (e.g. BTC ~60000 vs AAPL ~200). Without this,
    # a single shared model is skewed toward whichever asset has the largest
    # raw price scale.
    data["macd_hist_norm"] = data["macd_hist"] / close.replace(0, np.nan)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data["bb_pos"] = (close - sma20) / std20.replace(0, np.nan)

    data["ema_fast"] = close.ewm(span=9, adjust=False).mean()
    data["ema_slow"] = close.ewm(span=21, adjust=False).mean()
    data["ema_gap"] = (data["ema_fast"] - data["ema_slow"]) / close

    data["ret_1"] = close.pct_change(1)
    data["ret_5"] = close.pct_change(5)
    data["ret_10"] = close.pct_change(10)

    vol_mean = data["Volume"].rolling(20).mean()
    data["vol_ratio"] = data["Volume"] / vol_mean.replace(0, np.nan)

    body = (data["Close"] - data["Open"]).abs()
    rng = (data["High"] - data["Low"]).replace(0, np.nan)
    data["body_ratio"] = body / rng

    # --- Neu: Volatilitäts-Regime (ATR) ---
    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low, (high - previous_close).abs(), (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    data["atr"] = atr
    data["atr_norm"] = atr / close.replace(0, np.nan)

    # --- Neu: Trend-Regime (liegt der Kurs strukturell über/unter langfristigem Trend) ---
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    data["trend_strength"] = (sma50 - sma200) / close.replace(0, np.nan)

    labels, realized = _triple_barrier_labels(
        close.to_numpy(dtype=float), high.to_numpy(dtype=float),
        low.to_numpy(dtype=float), atr.to_numpy(dtype=float),
    )
    data["future_return"] = realized
    data["target"] = labels

    return data, ML_FEATURE_COLS

def train_walkforward_ml(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    Walk-Forward-Validierung: Das Modell wird immer nur auf der Vergangenheit
    trainiert und auf dem direkt folgenden, ihm unbekannten Abschnitt getestet.
    So bekommst du eine ehrliche Einschätzung, ob das Modell wirklich etwas
    "gelernt" hat, statt nur die Vergangenheit auswendig zu kennen.
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn ist nicht installiert. Bitte 'pip install scikit-learn' ausführen."}

    data, feature_cols = build_ml_features(df)
    usable = data.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    if len(usable) < 200:
        return {"error": "Zu wenig nutzbare Datenpunkte für ein Walk-Forward-Training (mind. ~200 benötigt)."}

    fold_size = len(usable) // (n_folds + 1)
    fold_accuracies = []
    fold_details = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end = fold_size * (fold + 2)
        train_slice = usable.iloc[:train_end]
        test_slice = usable.iloc[train_end:test_end]
        if len(train_slice) < 50 or len(test_slice) < 10:
            continue

        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42,
        )
        model.fit(train_slice[feature_cols], train_slice["target"])
        predictions = model.predict(test_slice[feature_cols])
        accuracy = float((predictions == test_slice["target"].values).mean())
        fold_accuracies.append(accuracy)
        fold_details.append({
            "fold": fold + 1,
            "train_size": len(train_slice),
            "test_size": len(test_slice),
            "accuracy": round(accuracy * 100, 1),
        })

    if not fold_accuracies:
        return {"error": "Nicht genug Daten, um Walk-Forward-Folds zu bilden. Größeres Intervall oder mehr Historie wählen."}

    final_model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42,
    )
    final_model.fit(usable[feature_cols], usable["target"])

    last_row = data.iloc[[-1]][feature_cols]
    if last_row.isna().any(axis=1).iloc[0]:
        live_probability = None
    else:
        live_probability = float(final_model.predict_proba(last_row)[0][1])

    baseline_up_rate = float(usable["target"].mean())

    return {
        "mean_accuracy": round(100 * float(np.mean(fold_accuracies)), 1),
        "fold_details": fold_details,
        "baseline_up_rate": round(100 * baseline_up_rate, 1),
        "live_probability_up": round(100 * live_probability, 1) if live_probability is not None else None,
        "feature_importances": dict(zip(feature_cols, final_model.feature_importances_.round(3))),
        "horizon": ML_HORIZON,
        "sample_size": len(usable),
    }

def render_ml_predictor():
    with st.expander("KI-Prognose (gemeinsames lernendes Modell)", expanded=False):
        st.caption(
            "Das Modell sammelt aus den Paper-Analysen gemeinsame Trainingsbeispiele in Supabase - "
            "getrennt nach Krypto und Aktien, da beide sich zu unterschiedlich verhalten für ein "
            "gemeinsames Modell. Das Label ist ein simulierter ATR-Stop/Ziel-Trade (2:1 Reward:Risk), "
            "kein reiner Richtungs-Tipp. Neue Modelle werden nur übernommen, wenn ihr Erwartungswert "
            "nach angenommenen Kosten positiv ist UND das aktive Modell nicht verschlechtert wird."
        )

        if not SKLEARN_AVAILABLE:
            st.warning("Für dieses Feature fehlt scikit-learn.")
            return

        ml_options = list(ASSETS.keys()) + st.session_state.watchlist
        ml_asset = st.selectbox("Asset", ml_options, key="ml_asset")
        ml_interval = st.selectbox(
            "Intervall", list(INTERVAL_CONFIG.keys()),
            index=list(INTERVAL_CONFIG.keys()).index("1d"), key="ml_interval",
        )
        ticker = ASSETS.get(ml_asset) or ml_asset
        active_class = asset_class_for_symbol(ticker)
        st.caption(f"Asset-Klasse für dieses Asset: **{active_class}** (eigenes Modell, getrennt von der anderen Klasse).")

        status = get_learning_status(active_class)
        if not status["ready"]:
            diag = get_supabase_diagnosis()
            st.warning(f"Gemeinsames Lernen ist noch nicht verbunden: {diag['reason']}")
        else:
            a, b, c = st.columns(3)
            a.metric(f"Beispiele ({active_class})", str(status["examples"]))
            b.metric("Aktives Modell", "Ja" if status["active_model"] else "Noch keins")
            c.metric("Letztes Training", str(status["last_training"] or "Noch keins")[:19])

        if st.button("Daten sammeln & Modell trainieren", key="run_ml"):
            with st.spinner("Sammle Paper-Daten und prüfe das Modell..."):
                try:
                    ml_df = load_data(ticker, ml_interval, "Yahoo Finance")
                    if ml_df.empty or len(ml_df) < 250:
                        st.error("Für sinnvolles Training werden mindestens 250 Kerzen benötigt.")
                    else:
                        feature_df, feature_cols = build_ml_features(ml_df)
                        saved, save_errors = save_learning_examples(ticker, ml_interval, feature_df, feature_cols)
                        st.info(f"{saved} Trainingsbeispiele aus {ticker} ({active_class}) wurden synchronisiert.")
                        if save_errors:
                            for msg in save_errors:
                                st.warning(msg)
                        status_after = get_learning_status(active_class)
                        if status_after["examples"] >= 200:
                            result = train_and_maybe_promote_shared_model(active_class)
                            st.session_state.ml_result = result
                        else:
                            st.session_state.ml_result = {"error": f"Noch {200 - status_after['examples']} Beispiele bis zum ersten Training für '{active_class}'."}
                except Exception as exc:
                    st.session_state.ml_result = {"error": str(exc)}

        result = st.session_state.ml_result
        if result is not None:
            if "error" in result:
                st.warning(result["error"])
            elif result.get("ok"):
                metrics = result["metrics"]
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Walk-Forward-Trefferquote", f'{metrics["mean_accuracy"]}%')
                expectancy = metrics.get("mean_expectancy_r")
                metric_b.metric("Erwartungswert/Trade", f'{expectancy:.2f}R' if expectancy is not None else "n/a")
                metric_c.metric("Modellstatus", "Übernommen" if result["promoted"] else "Verworfen")
                if result.get("rejection_reason"):
                    st.warning(result["rejection_reason"])
                if result.get("previous_expectancy") is not None:
                    st.caption(f'Vorheriges aktives Modell: {float(result["previous_expectancy"]):.3f}R Erwartungswert/Trade.')
                st.caption(
                    "Erwartungswert/Trade (R) = wie viele Vielfache des Stop-Risikos das Modell im Schnitt "
                    "pro Long-Signal macht, nach Abzug angenommener Fees/Slippage. Nur > 0 ist grundsätzlich handelbar."
                )
                st.dataframe(pd.DataFrame(metrics["fold_details"]), use_container_width=True, hide_index=True)
                importance_df = pd.DataFrame(
                    sorted(metrics["feature_importances"].items(), key=lambda kv: kv[1], reverse=True),
                    columns=["Feature", "Gewichtung"],
                )
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
                st.caption("Das System ist weiterhin Paper-Trading/Analyse. Historische Modelltests garantieren keine zukünftigen Ergebnisse.")


def analyze_ticker(ticker: str, interval_key: str = "1d", source: str = "Yahoo Finance"):
    try:
        df = load_data(ticker, interval_key, source)
    except Exception:
        return None
    if df.empty or len(df) < 20:
        return None

    patterns_list = detect_pattern(df)
    pattern_str = ", ".join(patterns_list)

    probability, hits = None, 0
    for p in patterns_list:
        if p in PATTERNS:
            prob, h = historical_probability(df, p)
            if prob is not None:
                probability, hits = prob, h
                break

    return {
        "ticker": ticker, "df": df, "pattern": pattern_str,
        "patterns_list": patterns_list, "probability": probability, "hits": hits,
        "trade_setup": calculate_trade_setup(df),
    }

# ------------------------------------------------------------
# Candlestick-Chart (ohne Nacht-Lücken & mit Zoom-Reset)
# ------------------------------------------------------------
def render_candlestick_chart(df: pd.DataFrame, pattern: str, ticker: str, n_candles: int = 40):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    plot_df = df.tail(n_candles).copy().reset_index(drop=True)

    if pd.api.types.is_datetime64_any_dtype(plot_df["Date"]):
        dates = pd.to_datetime(plot_df["Date"])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert("Europe/Berlin")
        has_intraday_time = bool((dates.dt.hour != 0).any() or (dates.dt.minute != 0).any())
        plot_df["DateStr"] = dates.dt.strftime("%d.%m. %H:%M" if has_intraday_time else "%d.%m.%Y")
    else:
        plot_df["DateStr"] = plot_df["Date"].astype(str)

    fig = go.Figure(data=[go.Candlestick(
        x=plot_df["DateStr"],
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"],
        increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
        name=ticker,
    )])

    if pattern and "Neutrale" not in pattern:
        last_row = plot_df.iloc[-1]
        h = last_row["High"]
        short_pattern = pattern.split(", ")[0]
        if len(short_pattern) > 28:
            short_pattern = short_pattern[:25] + "..."

        fig.add_annotation(
            x=last_row["DateStr"], y=h,
            text=short_pattern, showarrow=True, arrowhead=2, arrowcolor="#ffd60a",
            font=dict(color="#ffd60a", size=10), bgcolor="#30384a",
            bordercolor="#ffd60a", borderwidth=1, borderpad=4,
            xanchor="right", ax=-8, ay=-42,
        )

    tick_step = max(1, len(plot_df) // 6)
    tick_values = plot_df["DateStr"].iloc[::tick_step].tolist()
    fig.update_layout(
        paper_bgcolor="#252b39", plot_bgcolor="#252b39",
        font=dict(color="#e0e5ef"),
        margin=dict(l=12, r=12, t=58, b=54),
        height=340,
        xaxis=dict(
            type="category",
            showgrid=False,
            tickmode="array",
            tickvals=tick_values,
            tickangle=-30,
            tickfont=dict(size=10, color="#b8c2d3"),
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(showgrid=True, gridcolor="#3b4354", tickfont=dict(color="#b8c2d3")),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Gemini-Analyse
# ------------------------------------------------------------
def get_gemini_analysis(api_key: str, asset: str, pattern: str, probability, df: pd.DataFrame) -> str:
    last_close = round(float(df.iloc[-1]["Close"]), 2)
    cache_key = f"{asset}|{pattern}|{probability}|{last_close}"
    if cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key]

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "⚠️ Paket 'google-genai' fehlt. Bitte requirements.txt prüfen."

    try:
        client = genai.Client(api_key=api_key)

        close = df["Close"]
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        current_price = float(close.iloc[-1])

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = round(float(rsi_series.iloc[-1]), 1) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        trend = "Aufwärtstrend (über 200-Tage-Linie)" if current_price > sma200 else "Abwärtstrend (unter 200-Tage-Linie)"
        last_candles = df.tail(3)[["Open", "High", "Low", "Close"]].round(1).values.tolist()
        prob_str = f"{probability}%" if probability is not None else "Kein historischer Vorteil"

        prompt = (
            f"Analysiere {asset} als Trading-Experte:\n"
            f"- Aktueller Kurs: {round(current_price, 2)}\n"
            f"- Übergeordneter Trend: {trend}\n"
            f"- RSI (14): {rsi} (Unter 30 = überverkauft/Einstiegschance, Über 70 = überhitzt/Korrekturgefahr)\n"
            f"- Aktive Muster & Signale: {pattern} (Trefferquote: {prob_str})\n"
            f"- Letzte 3 Kerzen (OHLC): {last_candles}\n\n"
            f"Antworte auf Deutsch in GENAU diesem Format (4 Zeilen, je 1 kurzer Satz):\n"
            f"1) EMPFEHLUNG: Kaufen / Verkaufen / Abwarten – mit kurzem Warum\n"
            f"2) BEGRÜNDUNG: Kombiniere Trend, RSI und Signale\n"
            f"3) CHANCE: Chance auf Plus: X% / Chance auf Minus: Y% (X+Y=100)\n"
            f"4) RISIKO: Kurzer Risikohinweis"
        )

        try:
            config = types.GenerateContentConfig(
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        except Exception:
            config = types.GenerateContentConfig(max_output_tokens=500)

        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash", contents=prompt, config=config,
            )
            text = (response.text or "").strip()
            if not text:
                return "Gemini hat keine Antwort geliefert. Die mathematische Analyse oben bleibt verfügbar."
            st.session_state.ai_cache[cache_key] = text
            return text
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                return "Gemini-Limit erreicht. Bitte später erneut auf KI-Einschätzung laden klicken."
            if "UNAVAILABLE" in err or "503" in err:
                return "Gemini ist gerade nicht verfügbar. Die mathematische Analyse oben bleibt verfügbar."
            raise

    except Exception as e:
        return f"⚠️ Gemini-Fehler: {e}"

def get_ai_analysis(provider: str, api_key: str, asset: str, pattern: str, probability, df: pd.DataFrame) -> str:
    if provider == "Gemini":
        return get_gemini_analysis(api_key, asset, pattern, probability, df)

    last_close = round(float(df.iloc[-1]["Close"]), 2)
    cache_key = f"{provider}|{asset}|{pattern}|{probability}|{last_close}"
    if cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key]

    close = df["Close"]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
    current_price = float(close.iloc[-1])
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = round(float(rsi_series.iloc[-1]), 1) if not np.isnan(rsi_series.iloc[-1]) else 50.0
    trend = "Aufwärtstrend" if current_price > sma200 else "Abwärtstrend"
    prompt = (
        f"Analysiere {asset} kurz auf Deutsch. Kurs {current_price:.2f}, Trend {trend}, "
        f"RSI {rsi}, Signale {pattern}, historische Chance {probability if probability is not None else 'unbekannt'}%. "
        "Antworte in genau 3 kurzen Zeilen: Empfehlung (Kaufen/Verkaufen/Abwarten), "
        "Begründung, Risiko. Keine Anlageberatung."
    )
    try:
        if provider == "OpenAI":
            payload = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 180,
                "temperature": 0.2,
            }).encode("utf-8")
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions", data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                text = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
        else:
            payload = json.dumps({
                "model": "claude-3-5-haiku-latest",
                "max_tokens": 180,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }).encode("utf-8")
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=payload,
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                text = json.loads(response.read().decode("utf-8"))["content"][0]["text"].strip()
        st.session_state.ai_cache[cache_key] = text
        return text
    except urllib.error.HTTPError as error:
        if error.code in (429, 529):
            return f"{provider}-Limit erreicht. Bitte später erneut versuchen."
        return f"{provider}-Fehler ({error.code}). Die mathematische Analyse bleibt verfügbar."
    except Exception as error:
        return f"{provider}-Fehler: {error}"

def render_result_card(ticker: str, pattern: str, probability, hits: int, n_candles: int, interval_key: str):
    if probability is None:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="pattern-name">{ticker} · {pattern}</div>
                <div class="pattern-meta">Erkannte Signale · letzte {n_candles} Kerzen ({interval_key})</div>
                <div class="prob-box neutral-box">Aktiviertes Setup</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    is_bullish = probability >= 50
    box_class = "green-box" if is_bullish else "red-box"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="pattern-name">{ticker} · {pattern}</div>
            <div class="pattern-meta">{hits} historische Vergleichsfälle für Kerzenmuster · letzte {n_candles} Kerzen ({interval_key}) · Chance auf Plus</div>
            <div class="prob-box {box_class}">{probability}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_trade_setup(setup: dict, ticker: str, interval_key: str):
    st.markdown('<div class="section-label">Daytrading-Setup</div>', unsafe_allow_html=True)
    st.caption("Marktdaten von Yahoo Finance: je nach Börse und Intervall möglicherweise verzögert, kein garantierter Tick-Livefeed. Vor einer Order bitte den Brokerkurs prüfen.")

    signal_color = "#26a69a" if setup["direction"] == "long" else "#ef5350" if setup["direction"] == "short" else "#ffd60a"
    win_loss = f'{setup["wins"]} W / {setup["losses"]} L'
    win_rate = f'{setup["win_rate"]}%' if setup["win_rate"] is not None else "Nicht genug Fälle"
    rsi = f'{setup["rsi"]:.1f}' if setup["rsi"] is not None else "–"
    volume_ratio = f'{setup["volume_ratio"]:.2f}x' if setup["volume_ratio"] is not None else "–"
    entry_value = f'{setup["entry"]:.2f}'
    stop_value = f'{setup["stop"]:.2f}' if setup["stop"] is not None else "–"
    target_value = f'{setup["target"]:.2f}' if setup["target"] is not None else "–"
    levels = "Kein aktives Setup: Stop-Loss und Take-Profit werden erst bei einem klaren Long- oder Short-Signal berechnet."
    if setup["direction"] != "neutral":
        levels = "Stop-Loss begrenzt den geplanten Verlust; Take-Profit ist das automatisch berechnete Kursziel bei einem Chance-Risiko-Verhältnis von 2:1."

    st.markdown(
        f'<div class="trade-card">'
        f'<div class="trade-title">{ticker} · {interval_key}</div>'
        f'<div class="trade-subtitle">Regelbasiert aus EMA/RSI/ATR und Volumen</div>'
        f'<div class="trade-signal" style="color:{signal_color};">{setup["signal"]}</div>'
        f'<div class="trade-grid">'
        f'<div class="trade-metric"><span class="trade-metric-label">Entry / aktueller Kurs</span><span class="trade-metric-value">{entry_value}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Stop-Loss</span><span class="trade-metric-value">{stop_value}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Take-Profit / Kursziel</span><span class="trade-metric-value">{target_value}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">RSI (0–100)</span><span class="trade-metric-value">{rsi}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Volumen vs. 20er-Schnitt</span><span class="trade-metric-value">{volume_ratio}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Historische Trefferquote</span><span class="trade-metric-value">{win_rate} · {win_loss}</span></div>'
        f'</div><div class="trade-note">{levels}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Risiko- und Hebel-Rechner</div>', unsafe_allow_html=True)
    capital_col, risk_col, leverage_col = st.columns(3)
    with capital_col:
        capital = st.number_input("Kapital", min_value=50.0, value=1000.0, step=50.0, key=f"capital_{ticker}_{interval_key}")
    with risk_col:
        risk_percent = st.number_input("Risiko %", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key=f"risk_{ticker}_{interval_key}")
    with leverage_col:
        leverage = st.number_input("Hebel", min_value=1.0, max_value=10.0, value=1.0, step=0.5, key=f"leverage_{ticker}_{interval_key}")

    calculation_mode = st.selectbox(
        "Rechenrichtung",
        ["Automatisch (Signal)", "Long berechnen", "Short berechnen"],
        key=f"calculation_mode_{ticker}_{interval_key}",
        help="Automatisch verwendet nur ein erkanntes Setup. Long/Short berechnen ist eine separate Beispielrechnung und keine Empfehlung.",
    )
    calculation_direction = setup["direction"]
    if calculation_mode == "Long berechnen":
        calculation_direction = "long"
    elif calculation_mode == "Short berechnen":
        calculation_direction = "short"

    if calculation_direction == "neutral":
        st.info("Das aktuelle Marktsignal lautet Abwarten. Wähle oben Long oder Short berechnen, um nur die Positionsgröße zu simulieren.")
        return

    calculation_stop = setup["entry"] - setup["atr"] if calculation_direction == "long" else setup["entry"] + setup["atr"]
    calculation_target = setup["entry"] + (2 * setup["atr"]) if calculation_direction == "long" else setup["entry"] - (2 * setup["atr"])
    if calculation_mode != "Automatisch (Signal)":
        st.caption(f"Beispielrechnung: Entry {setup['entry']:.2f} · Stop-Loss {calculation_stop:.2f} · Take-Profit {calculation_target:.2f}")

    risk_per_unit = abs(setup["entry"] - calculation_stop)
    risk_amount = capital * risk_percent / 100
    risk_based_units = risk_amount / risk_per_unit if risk_per_unit else 0
    margin_limited_units = capital * leverage / setup["entry"] if setup["entry"] else 0
    units = min(risk_based_units, margin_limited_units)
    notional = units * setup["entry"]
    margin = notional / leverage
    actual_risk = units * risk_per_unit
    st.markdown(
        f'<div class="trade-card"><div class="trade-grid">'
        f'<div class="trade-metric"><span class="trade-metric-label">Max. Stückzahl (Einheiten)</span><span class="trade-metric-value">{units:.4f}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Positionswert</span><span class="trade-metric-value">{notional:.2f}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Gebundene Margin</span><span class="trade-metric-value">{margin:.2f}</span></div>'
        f'<div class="trade-metric"><span class="trade-metric-label">Max. Verlust am Stop (€)</span><span class="trade-metric-value">{actual_risk:.2f} ({actual_risk / capital * 100:.2f}%)</span></div>'
        f'</div><div class="trade-note">Max. Stückzahl bedeutet: so viele Einheiten können bis zum Stop gehalten werden. Risiko % wird zuerst in Euro umgerechnet. Gebühren, Slippage, Finanzierungskosten und Gaps sind nicht eingerechnet.</div></div>',
        unsafe_allow_html=True,
    )

def render_mini_card(ticker: str, pattern: str, probability):
    if probability is None:
        st.markdown(
            f"""
            <div class="mini-card">
                <div>
                    <div class="mini-ticker">{ticker}</div>
                    <div class="mini-pattern">{pattern}</div>
                </div>
                <div class="mini-prob" style="color:#8e8e93; background:#2c2c2e;">–</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    is_bullish = probability >= 50
    color = "#26a69a" if is_bullish else "#ef5350"
    bg = "rgba(38,166,154,0.14)" if is_bullish else "rgba(239,83,80,0.14)"
    st.markdown(
        f"""
        <div class="mini-card">
            <div>
                <div class="mini-ticker">{ticker}</div>
                <div class="mini-pattern">{pattern}</div>
            </div>
            <div class="mini-prob" style="color:{color}; background:{bg};">{probability}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# Candlestick-Lehrbuch
# ------------------------------------------------------------
def render_pattern_text(pattern: str) -> str:
    return PATTERN_EXPLAIN.get(pattern, "Dieses Muster beschreibt eine mögliche Veränderung des Kauf- und Verkaufsdrucks.")

def render_pattern_visual(pattern: str) -> str:
    single_patterns = {"Hammer (bullisch)", "Hanging Man (bärisch)", "Inverted Hammer (bullisch)", "Shooting Star (bärisch)", "Doji (Unentschlossenheit)", "Bullish Marubozu (bullisch)", "Bearish Marubozu (bärisch)", "Spinning Top (Unentschlossenheit)", "Dragonfly Doji (bullisch)", "Gravestone Doji (bärisch)", "Bullish Belt Hold (bullisch)", "Bearish Belt Hold (bärisch)"}
    three_patterns = {"Morning Star (bullisch)", "Evening Star (bärisch)", "3 weiße Soldaten (bullisch)", "3 schwarze Krähen (bärisch)", "Three Inside Up (bullisch)", "Three Inside Down (bärisch)"}
    if pattern == "Doji (Unentschlossenheit)":
        candles = ["doji"]
    elif pattern == "Dragonfly Doji (bullisch)":
        candles = ["dragonfly"]
    elif pattern == "Gravestone Doji (bärisch)":
        candles = ["gravestone"]
    elif pattern == "Spinning Top (Unentschlossenheit)":
        candles = ["doji"]
    elif pattern in single_patterns:
        candles = ["bear long" if any(word in pattern for word in ["Bearish", "Hanging", "Shooting"]) else "bull long"]
    elif pattern in three_patterns:
        if "Morning" in pattern:
            candles = ["bear long", "bull small", "bull long"]
        elif "Evening" in pattern:
            candles = ["bull long", "bull small", "bear long"]
        elif "Inside Up" in pattern:
            candles = ["bear long", "bull small", "bull long"]
        elif "Inside Down" in pattern:
            candles = ["bull long", "bear small", "bear long"]
        elif "Soldaten" in pattern:
            candles = ["bull", "bull", "bull"]
        else:
            candles = ["bear", "bear", "bear"]
    elif "Rising Three" in pattern:
        candles = ["bull long", "bear small", "bear small", "bear small", "bull long"]
    elif "Falling Three" in pattern:
        candles = ["bear long", "bull small", "bull small", "bull small", "bear long"]
    elif "Three Line Strike" in pattern:
        candles = ["bear", "bear", "bear", "bull long"] if "Bullish" in pattern else ["bull", "bull", "bull", "bear long"]
    elif "Kicker" in pattern:
        candles = ["bear", "bull long"] if "Bullish" in pattern else ["bull", "bear long"]
    elif "Harami" in pattern:
        candles = ["bear long", "bull small"] if "Bullish" in pattern else ["bull long", "bear small"]
    elif "Engulfing" in pattern:
        candles = ["bear small", "bull long"] if "Bullish" in pattern else ["bull small", "bear long"]
    elif "Piercing" in pattern:
        candles = ["bear long", "bull"]
    elif "Dark Cloud" in pattern:
        candles = ["bull long", "bear"]
    elif "Tweezer Bottom" in pattern:
        candles = ["bear", "bull"]
    else:
        candles = ["bull", "bear"]
    candle_html = "".join(
        f'<div class="candle {candle}"><span class="candle-wick"></span><span class="candle-body"></span></div>'
        for candle in candles
    )
    return f'<div class="pattern-visual">{candle_html}</div>'

def render_pattern_book():
    st.markdown(
        '<div class="pattern-book-heading">'
        '<div><div class="pattern-book-heading-title">Trading lernen</div>'
        '<div class="pattern-book-heading-subtitle">Candlestick-Muster verstehen und visuell erkennen</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Musterbibliothek öffnen", expanded=False):
        st.caption("Muster sind Hinweise, keine sicheren Vorhersagen. Bestätige sie immer mit Trend, Volumen und Risikomanagement.")
        single_patterns = {"Hammer (bullisch)", "Hanging Man (bärisch)", "Inverted Hammer (bullisch)", "Shooting Star (bärisch)", "Doji (Unentschlossenheit)", "Bullish Marubozu (bullisch)", "Bearish Marubozu (bärisch)", "Spinning Top (Unentschlossenheit)", "Dragonfly Doji (bullisch)", "Gravestone Doji (bärisch)", "Bullish Belt Hold (bullisch)", "Bearish Belt Hold (bärisch)"}
        three_patterns = {"Morning Star (bullisch)", "Evening Star (bärisch)", "3 weiße Soldaten (bullisch)", "3 schwarze Krähen (bärisch)", "Three Inside Up (bullisch)", "Three Inside Down (bärisch)"}
        long_patterns = {"Rising Three Methods (bullisch)", "Falling Three Methods (bärisch)", "Bullish Three Line Strike (bullisch)", "Bearish Three Line Strike (bärisch)"}
        groups = {
            "Einzelkerzen": [pattern for pattern in PATTERNS if pattern in single_patterns],
            "Zwei-Kerzen-Formationen": [pattern for pattern in PATTERNS if pattern not in single_patterns and pattern not in three_patterns and pattern not in long_patterns and "Kicker" not in pattern],
            "Drei-Kerzen-Formationen": [pattern for pattern in PATTERNS if pattern in three_patterns],
            "Mehrkerzen-Formationen": [pattern for pattern in PATTERNS if pattern in long_patterns or "Kicker" in pattern],
        }
        category_tabs = st.tabs([f"{group_name} ({len(patterns)})" for group_name, patterns in groups.items()])
        for category_tab, (group_name, patterns) in zip(category_tabs, groups.items()):
            with category_tab:
                for pattern in patterns:
                    if "bullisch" in pattern:
                        direction, direction_class = "Bullisch", ""
                    elif "bärisch" in pattern:
                        direction, direction_class = "Bärisch", "bearish"
                    else:
                        direction, direction_class = "Neutral", "neutral"
                    st.markdown(
                        f'<div class="pattern-book-card">'
                        f'<div class="pattern-book-title">{pattern}</div>'
                        f'<div class="pattern-book-direction {direction_class}">{direction}</div>'
                        f'{render_pattern_visual(pattern)}'
                        f'<div class="pattern-book-text">{render_pattern_text(pattern)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

render_pattern_book()

def render_market_hours():
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    weekday = now.weekday() < 5
    xetra_open = weekday and time(9, 0) <= now.time() < time(17, 30)
    us_open = weekday and time(15, 30) <= now.time() < time(22, 0)
    xetra_status = "Geöffnet" if xetra_open else "Geschlossen"
    us_status = "Geöffnet" if us_open else "Geschlossen"
    with st.expander("Handelszeiten in deutscher Zeit", expanded=False):
        st.caption("Regelhandel, Montag bis Freitag. Feiertage und Brokerzeiten können abweichen.")
        st.markdown(
            f"**Deutsche Börse / Xetra:** 09:00–17:30 · aktuell: **{xetra_status}**  \n"
            f"**USA / NYSE und Nasdaq:** 15:30–22:00 · aktuell: **{us_status}**  \n"
            "**Krypto:** 24 Stunden, 7 Tage die Woche · Liquidität und Spreads schwanken."
        )
        st.caption("Datenquelle: Yahoo Finance. Aktienkurse können je nach Börse typischerweise verzögert sein; Krypto ist oft näher an Echtzeit, aber nicht garantiert tickgenau. Für echte Echtzeitdaten brauchst du einen lizenzierten Börsen- oder Brokerfeed.")

render_market_hours()

def render_paper_bot():
    with st.expander("Paper-Bot trainieren", expanded=False):
        st.caption(
            "Das hier ist kein KI-Modell, sondern eine feste Regel (EMA-Kreuzung + RSI + ATR-Stop). "
            "Die Regel lernt nichts - sie feuert einfach nur, wenn ihre Bedingung erfüllt ist. "
            "Yahoo-Finance-Daten werden in Trainings- und Testabschnitt geteilt, es werden nur virtuelle Trades simuliert."
        )
        bot_options = list(ASSETS.keys()) + st.session_state.watchlist
        bot_asset = st.selectbox("Bot-Asset", bot_options, key="bot_asset")
        bot_interval = st.selectbox("Bot-Intervall", list(INTERVAL_CONFIG.keys()), index=list(INTERVAL_CONFIG.keys()).index("1d"), key="bot_interval")

        history_options = {
            "1 Jahr": "1y", "2 Jahre": "2y", "5 Jahre": "5y", "10 Jahre": "10y", "Maximal verfügbar": "max",
        }
        bot_history_label = st.selectbox(
            "Wie viel Historie simulieren?", list(history_options.keys()),
            index=2, key="bot_history",
            help="Mehr Historie = mehr mögliche Trades, aber bei Intraday-Intervallen begrenzt Yahoo Finance die Verfügbarkeit ohnehin (z.B. 1m nur 7 Tage).",
        )
        bot_period_override = history_options[bot_history_label]

        bot_capital, bot_risk = st.columns(2)
        with bot_capital:
            bot_initial_capital = st.number_input("Startkapital (€)", min_value=100.0, value=1000.0, step=100.0, key="bot_capital")
        with bot_risk:
            bot_risk_percent = st.number_input("Risiko pro Trade (%)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="bot_risk")

        auto_optimize = st.checkbox(
            "Parameter automatisch optimieren (Grid-Search, out-of-sample getestet)",
            value=True, key="bot_optimize",
        )
        bot_min_trades = st.number_input(
            "Mind. Trades im Trainingsabschnitt, damit eine Parameter-Kombi zählt", min_value=1, max_value=100,
            value=8, step=1, key="bot_min_trades",
            help="Niedriger = mehr Kombinationen werden zugelassen, aber die Statistik pro Kombi wird unsicherer.",
        )
        bot_min_holdout_trades = st.number_input(
            "Mind. Trades im Test-/Holdout-Abschnitt", min_value=1, max_value=50,
            value=3, step=1, key="bot_min_holdout_trades",
        )

        if st.button("Simulation starten", key="run_paper_bot"):
            ticker = ASSETS.get(bot_asset) or bot_asset
            with st.spinner("Historische Yahoo-Finance-Daten werden getestet..."):
                bot_df = load_data(ticker, bot_interval, "Yahoo Finance", period_override=bot_period_override, max_candles=0)
                if bot_df.empty or len(bot_df) < 150:
                    st.error("Für diese Simulation werden mindestens 150 Kerzen benötigt.")
                    st.session_state.paper_bot_result = None
                    st.session_state.paper_bot_best_params = None
                elif auto_optimize:
                    best = optimize_paper_bot(bot_df, bot_initial_capital, bot_risk_percent, min_trades=int(bot_min_trades), min_holdout_trades=int(bot_min_holdout_trades))
                    if best is None:
                        st.warning("Keine Parameter-Kombination hat genug Trades erzeugt. Versuch mehr Historie, ein anderes Intervall/Asset, oder senk die Mindest-Trades-Schwelle.")
                        st.session_state.paper_bot_result = None
                        st.session_state.paper_bot_best_params = None
                    else:
                        st.session_state.paper_bot_result = best["holdout"]
                        st.session_state.paper_bot_best_params = best["params"]
                else:
                    st.session_state.paper_bot_result = simulate_paper_bot(bot_df, bot_initial_capital, bot_risk_percent)
                    st.session_state.paper_bot_best_params = None

        result = st.session_state.paper_bot_result
        if result is not None:
            if st.session_state.get("paper_bot_best_params"):
                bp = st.session_state.paper_bot_best_params
                st.caption(
                    f"Beste gefundene Parameter (out-of-sample getestet): "
                    f"EMA {bp['ema_fast']}/{bp['ema_slow']}, RSI-Periode {bp['rsi_period']}, "
                    f"ATR×{bp['atr_mult']}, Chance/Risiko {bp['reward_risk']}"
                )
            metric_a, metric_b, metric_c, metric_d = st.columns(4)
            metric_a.metric("Endkapital", f'{result["final_equity"]:.2f} €')
            metric_b.metric("Rendite", f'{result["return_percent"]:.1f}%')
            metric_c.metric("Trades", str(result["wins"] + result["losses"]))
            metric_d.metric("Winrate", f'{result["win_rate"]:.1f}%' if result["win_rate"] is not None else "–")
            st.caption(f'{result["wins"]} Gewinne / {result["losses"]} Verluste · Test ausschließlich auf Daten, die der Optimierung nicht bekannt waren.')
            if result["trades"]:
                st.dataframe(pd.DataFrame(result["trades"]).tail(20), use_container_width=True, hide_index=True)

render_paper_bot()
render_ml_predictor()

# ------------------------------------------------------------
# Tabs: Einzelanalyse vs. Meine Positionen
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["Einzelanalyse", "Meine Positionen"])

# ============================================================
# TAB 1 – Einzelanalyse
# ============================================================
with tab1:
    st.markdown('<div class="section-label">Asset</div>', unsafe_allow_html=True)
    combined_options = list(ASSETS.keys()) + st.session_state.watchlist
    asset_choice = st.selectbox("Asset wählen", combined_options, label_visibility="collapsed", key="single_asset")

    st.markdown('<div class="section-label">Zeitrahmen (Kerzen-Intervall)</div>', unsafe_allow_html=True)
    interval_label = st.selectbox(
        "Zeitrahmen",
        list(INTERVAL_CONFIG.keys()),
        index=list(INTERVAL_CONFIG.keys()).index("1d"),
        label_visibility="collapsed",
        key="single_interval",
    )

    button_left, button_center, button_right = st.columns([1, 2, 1])
    with button_center:
        run = st.button("Chart analysieren", key="single_run", use_container_width=True)

    if run:
        ticker = ASSETS.get(asset_choice) or asset_choice

        with st.spinner("Lade Kursdaten..."):
            try:
                st.session_state.single_result = analyze_ticker(ticker, interval_label, data_source)
            except Exception as error:
                st.session_state.single_result = None
                st.error(str(error))
            st.session_state.single_ai_text = None
            st.session_state.single_ai_key = None

        if st.session_state.single_result is None:
            st.error("Nicht genügend Kursdaten gefunden. Bitte anderes Asset/Intervall wählen.")

    result = st.session_state.single_result
    if result is not None:
        render_result_card(
            result["ticker"], result["pattern"], result["probability"],
            result["hits"], len(result["df"]), interval_label,
        )
        render_trade_setup(result["trade_setup"], result["ticker"], interval_label)
        render_candlestick_chart(result["df"], result["pattern"], result["ticker"])

        provider_key = {
            "Gemini": st.session_state.gemini_api_key,
            "OpenAI": st.session_state.openai_api_key,
            "Claude": st.session_state.anthropic_api_key,
        }.get(ai_provider, "")
        if ai_provider != "Keiner" and provider_key:
            st.caption("Die KI ist optional und wird nur nach Klick auf den folgenden Button angefragt.")
            if st.button("KI-Einschätzung laden", key=f"single_ai_{result['ticker']}"):
                with st.spinner(f"{ai_provider} erstellt eine kurze Einschätzung..."):
                    st.session_state.single_ai_text = get_ai_analysis(
                        ai_provider, provider_key, result["ticker"], result["pattern"],
                        result["probability"], result["df"],
                    )
            if st.session_state.single_ai_text:
                st.markdown(
                    f'<div class="ai-card"><div class="ai-label">KI-Einschätzung</div>{st.session_state.single_ai_text.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Wähle oben einen KI-Anbieter und hinterlege den passenden API-Key für eine optionale Einschätzung.")

        st.markdown(
            '<div class="disclaimer">Keine Anlageberatung. Rein statistische/historische '
            'Auswertung, keine Garantie für zukünftige Kursbewegungen.</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# TAB 2 – Meine Positionen (Watchlist + CSV-Import)
# ============================================================
with tab2:
    st.markdown(
        """
        <div class="info-card">
        Trade Republic bietet keine offizielle Schnittstelle für Drittanbieter-Logins –
        ein direkter Login mit deinem TR-Passwort in einer fremden App wäre nicht sicher.
        Trag deine Positionen stattdessen hier ein (manuell oder per CSV),
        die Kurse holt sich die App automatisch über Yahoo Finance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Ticker manuell hinzufügen</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        new_ticker = st.text_input(
            "Ticker", label_visibility="collapsed",
            placeholder="z.B. SAP.DE, AAPL, MSFT, BTC-USD",
            key="new_ticker_input",
        )
    with col_b:
        add_clicked = st.button("Add", key="add_ticker_btn")

    if add_clicked and new_ticker.strip():
        t = new_ticker.strip().upper()
        if t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
        st.rerun()

    st.caption("Tipp: Deutsche Aktien meist mit **.DE** (z.B. SAP.DE), US-Aktien ohne Zusatz (z.B. AAPL).")

    if st.session_state.watchlist:
        st.markdown('<div class="section-label">Deine Watchlist</div>', unsafe_allow_html=True)
        chips_html = "".join(f'<span class="watch-chip">{t}</span>' for t in st.session_state.watchlist)
        st.markdown(chips_html, unsafe_allow_html=True)

        remove_choice = st.multiselect(
            "Ticker entfernen", st.session_state.watchlist,
            placeholder="Ticker zum Entfernen auswählen", key="remove_select",
        )
        if remove_choice and st.button("Entfernen", key="remove_btn"):
            st.session_state.watchlist = [t for t in st.session_state.watchlist if t not in remove_choice]
            st.rerun()

    st.markdown('<div class="section-label">Positionen per CSV importieren</div>', unsafe_allow_html=True)
    st.caption(
        "Lade eine CSV-Datei mit deinen Positionen hoch (z.B. selbst exportiert oder "
        "abgetippt). Erwartet wird mindestens eine Spalte mit dem Ticker-Symbol."
    )
    csv_file = st.file_uploader("CSV-Datei", type=["csv"], label_visibility="collapsed")

    if csv_file is not None:
        try:
            csv_df = pd.read_csv(csv_file)
        except Exception:
            csv_file.seek(0)
            csv_df = pd.read_csv(io.StringIO(csv_file.getvalue().decode("utf-8", errors="ignore")), sep=";")

        st.dataframe(csv_df.head(20), use_container_width=True, height=180)

        ticker_col = None
        for candidate in ["Ticker", "ticker", "Symbol", "symbol", "TICKER"]:
            if candidate in csv_df.columns:
                ticker_col = candidate
                break

        if ticker_col is None:
            ticker_col = st.selectbox(
                "Welche Spalte enthält den Ticker?", csv_df.columns.tolist(), key="csv_ticker_col",
            )
        else:
            st.caption(f"Ticker-Spalte automatisch erkannt: **{ticker_col}**")

        if st.button("Aus CSV in Watchlist übernehmen", key="import_csv_btn"):
            new_tickers = (
                csv_df[ticker_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
            )
            added = 0
            for t in new_tickers:
                if t and t not in st.session_state.watchlist:
                    st.session_state.watchlist.append(t)
                    added += 1
            st.success(f"{added} neue Ticker zur Watchlist hinzugefügt.")
            st.rerun()

    if st.session_state.watchlist:
        st.markdown('<div class="section-label">Alle Positionen analysieren</div>', unsafe_allow_html=True)
        batch_interval = st.selectbox(
            "Zeitrahmen für alle",
            list(INTERVAL_CONFIG.keys()),
            index=list(INTERVAL_CONFIG.keys()).index("1d"),
            key="batch_interval",
        )

        st.markdown('<div class="cta-btn">', unsafe_allow_html=True)
        analyze_all = st.button("Alle analysieren", key="analyze_all_btn")
        st.markdown('</div>', unsafe_allow_html=True)

        batch_provider_key = {
            "Gemini": st.session_state.gemini_api_key,
            "OpenAI": st.session_state.openai_api_key,
            "Claude": st.session_state.anthropic_api_key,
        }.get(ai_provider, "")
        use_ai_batch = st.checkbox(
            "KI-Einschätzungen für alle Ticker anfordern",
            value=False,
            disabled=not bool(batch_provider_key) or ai_provider == "Keiner",
            key="use_ai_batch",
            help="Verbraucht eine Gemini-Anfrage pro erfolgreich geladenem Ticker.",
        )
        if not batch_provider_key or ai_provider == "Keiner":
            st.caption("Wähle oben einen KI-Anbieter und hinterlege den passenden API-Key für KI-Einschätzungen.")

        if analyze_all:
            progress = st.progress(0.0, text="Starte Analyse...")
            results = []
            for i, t in enumerate(st.session_state.watchlist):
                progress.progress((i + 1) / len(st.session_state.watchlist), text=f"Analysiere {t}...")
                try:
                    res = analyze_ticker(t, batch_interval, data_source)
                except Exception as error:
                    st.warning(f"{t}: {error}")
                    res = None
                if res:
                    if batch_provider_key and ai_provider != "Keiner" and use_ai_batch:
                        res["ai_text"] = get_ai_analysis(
                            ai_provider, batch_provider_key, t, res["pattern"], res["probability"], res["df"]
                        )
                    else:
                        res["ai_text"] = None
                    results.append(res)
            progress.empty()

            if not results:
                st.error("Für keinen deiner Ticker konnten Daten geladen werden. Bitte Symbole prüfen.")
            else:
                for res in sorted(
                    results,
                    key=lambda r: r["probability"] if r["probability"] is not None else -1,
                    reverse=True,
                ):
                    render_mini_card(res["ticker"], res["pattern"], res["probability"])
                    render_candlestick_chart(res["df"], res["pattern"], res["ticker"], n_candles=25)
                    if res["ai_text"]:
                        ai_html = res["ai_text"].replace(chr(10), "<br>")
                        st.markdown(
                            f'<div class="ai-card" style="margin-top:-0.3em;">'
                            f'<div class="ai-label">{res["ticker"]}</div>{ai_html}</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    '<div class="disclaimer">Keine Anlageberatung. Rein statistische/historische '
                    'Auswertung, keine Garantie für zukünftige Kursbewegungen.</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("Noch keine Positionen in der Watchlist.")
