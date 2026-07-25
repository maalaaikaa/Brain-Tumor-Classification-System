from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from config import HISTORY_PATH, METRICS_PATH, TRAINING_SUMMARY_PATH


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f6f8fb; color: #111827; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #d9e2ec; }
        [data-testid="stSidebar"] * { font-size: 0.96rem; }
        .block-container { padding-top: 1.5rem; max-width: 1180px; }
        .hero {
            padding: 2rem;
            border-radius: 8px;
            background: linear-gradient(135deg, #ecfeff 0%, #f8fafc 52%, #eef2ff 100%);
            border: 1px solid #dbe4ee;
            box-shadow: 0 12px 30px rgba(15, 23, 42, .08);
        }
        .hero h1 { margin: 0 0 .5rem 0; font-size: 2.4rem; letter-spacing: 0; }
        .hero p { margin: 0; max-width: 760px; color: #475569; font-size: 1.05rem; }
        .info-strip {
            margin-top: 1rem;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
        }
        .info-strip div {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: .9rem 1rem;
            color: #334155;
        }
        .disclaimer {
            padding: 1rem;
            border-left: 4px solid #0f766e;
            background: #f0fdfa;
            border-radius: 6px;
            color: #134e4a;
        }
        .login-card {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 12px;
            padding: 2.5rem;
            max-width: 450px;
            margin: 3rem auto;
            box-shadow: 0 10px 30px rgba(15, 23, 42, .06);
        }
        .login-title {
            color: #0f766e;
            text-align: center;
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: .8rem 1rem;
        }
        .stButton>button, .stDownloadButton>button {
            border-radius: 8px;
            border: 1px solid #0f766e;
        }
        @media (max-width: 760px) {
            .hero h1 { font-size: 1.8rem; }
            .info-strip { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Brain Tumor Classification System</h1>
          <p>Deep learning MRI classification with confidence scores, model evaluation, and Grad-CAM visual explanation.</p>
          <div class="info-strip">
            <div><strong>4 classes</strong><br>Glioma, meningioma, pituitary, no tumor</div>
            <div><strong>Fast review</strong><br>Upload common MRI image formats</div>
            <div><strong>Explainable</strong><br>Visual highlight when the model supports it</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metrics_dashboard() -> None:
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        cols = st.columns(3)
        cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        cols[1].metric("Macro F1", f"{metrics.get('macro_f1', 0):.2%}")
        cols[2].metric("AUC", f"{metrics.get('auc', 0):.3f}")
    elif TRAINING_SUMMARY_PATH.exists():
        summary = json.loads(TRAINING_SUMMARY_PATH.read_text())
        cols = st.columns(3)
        cols[0].metric("Best Validation Accuracy", f"{summary.get('best_val_accuracy', 0):.2%}")
        cols[1].metric("Best Validation Loss", f"{summary.get('best_val_loss', 0):.3f}")
        cols[2].metric("Epochs", summary.get("epochs", 0))
        st.caption("Full test metrics are not generated yet. Run evaluation to create reports/model_metrics.json.")
    else:
        st.info("Train and evaluate a model to populate performance metrics.")

    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
        loss_cols = [c for c in history.columns if "loss" in c]
        accuracy_cols = [c for c in history.columns if "accuracy" in c]
        if loss_cols:
            st.plotly_chart(px.line(history, y=loss_cols, title="Loss Curves"), use_container_width=True)
        if accuracy_cols:
            st.plotly_chart(px.line(history, y=accuracy_cols, title="Accuracy Curves"), use_container_width=True)
