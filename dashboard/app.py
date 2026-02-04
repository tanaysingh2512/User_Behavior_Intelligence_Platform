from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import settings
from src.utils.db import read_df

st.set_page_config(page_title="UBIP Dashboard", layout="wide")

st.title("User Behavior Intelligence Platform")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Risk Segments")
    preds = read_df("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5000")
    if preds.empty:
        st.info("No predictions yet. Run the API batch scoring to populate.")
    else:
        bins = pd.cut(
            preds["churn_proba"],
            bins=[0, 0.4, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
        )
        segment_counts = bins.value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        st.bar_chart(segment_counts)

with col2:
    st.subheader("Drift Monitor (Latest)")
    drift = read_df(
        """
        SELECT feature_name, psi, ks, computed_at
        FROM drift_metrics
        ORDER BY computed_at DESC
        LIMIT 50
        """
    )
    if drift.empty:
        st.info("No drift metrics yet. Run the monitor job.")
    else:
        latest = drift.sort_values("computed_at").groupby("feature_name").tail(1)
        st.dataframe(latest, use_container_width=True)

st.subheader("Global Feature Importance (SHAP)")
shap_path = settings.models_dir / "shap_global.json"
if shap_path.exists():
    with shap_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    shap_df = pd.DataFrame(payload["global_importance"]).head(15)
    st.bar_chart(shap_df.set_index("feature")["mean_abs_shap"])
else:
    st.info("No SHAP summary found. Run the explain job.")

st.subheader("Feature Store Snapshot")
features = read_df("SELECT * FROM features LIMIT 100")
if features.empty:
    st.info("No features in store. Load features first.")
else:
    st.dataframe(features, use_container_width=True)
