# ============================================================
# INTEGRATION in app.py
# ============================================================
#
# 1) Oben bei den Imports:
#
# from learning_engine import SharedLearningEngine
#
# 2) Einmal nach den Imports:
#
# LEARNING_ENGINE = SharedLearningEngine(
#     root="learning_store",
#     horizon=5,
#     min_samples=250,
#     retrain_after_new_samples=25,
# )
#
# 3) In deinen bestehenden render_ml_predictor()-Bereich kannst du
#    zusätzlich diesen Block verwenden:
#
def render_shared_learning_status(engine, asset_label: str, interval: str, ticker: str, df):
    import streamlit as st

    st.markdown("### 🧠 Gemeinsamer Lernstand")

    prediction = engine.predict_latest(df, asset_label, interval)
    model = engine.load_active_model(asset_label, interval)

    if model is None:
        st.info(
            "Noch kein freigegebenes gemeinsames Modell vorhanden. "
            "Starte den Lern-Worker, damit historische Daten gesammelt und "
            "das erste Modell validiert wird."
        )
    else:
        p = prediction["probability_up"] if prediction else None
        c1, c2, c3 = st.columns(3)
        c1.metric("Modell", f"v{model['version']}")
        c2.metric(
            "Prognose",
            f"{p*100:.1f}% ↑" if p is not None else "–",
        )
        c3.metric("Trainiert", str(model["trained_at"])[:19].replace("T", " "))

        st.caption(
            "Dieses Modell liegt zentral im learning_store und wird von allen "
            "Nutzern derselben App-Instanz gemeinsam verwendet."
        )

    if st.button("Lernstand jetzt synchronisieren", key=f"sync_{asset_label}_{interval}"):
        added = engine.sync_from_yfinance(
            asset=asset_label,
            interval=interval,
            ticker=ticker,
            period="2y",
        )
        result = engine.train_if_needed(asset_label, interval, force=True)
        st.success(
            f"{added} Datenpunkte synchronisiert · "
            f"Status: {result.status} · {result.message}"
        )
        st.rerun()
