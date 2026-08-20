"""LeadGuard Streamlit Dashboard.

Calls the running FastAPI API for all data — no logic duplicated here.
Every number displayed traces to a real API response.

Architecture §10 (Phase 10):
  Panel 1: CSV upload → calls /v1/predict → ranked table
  Panel 2: SHAP explanation per selected property
  Panel 3: Fairness summary (/v1/fairness-report)
  Panel 4: Cost-curve chart from active_learning_curve.csv

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import pandas as pd
import requests
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LeadGuard — Lead Service Line Prioritization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_get(path: str, params: dict | None = None) -> dict | None:
    """Call a GET endpoint on the LeadGuard API.

    Args:
        path: API path (e.g. '/v1/health').
        params: Optional query params.

    Returns:
        Parsed JSON dict, or None on error.
    """
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({path}): {e}")
        return None


def _api_post(path: str, json_body: dict) -> dict | None:
    """Call a POST endpoint on the LeadGuard API.

    Args:
        path: API path.
        json_body: Request body.

    Returns:
        Parsed JSON dict, or None on error.
    """
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_body, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error (POST {path}): {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar — API health check
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/water.png", width=60)
    st.title("LeadGuard")
    st.caption("Lead service line prioritization")
    st.divider()

    health = _api_get("/v1/health")
    if health:
        status = health.get("status", "unknown")
        if status == "ok":
            st.success("✅ API connected")
        else:
            st.warning(f"⚠️ API status: {status}")
        st.caption(f"Model loaded: {health.get('model_loaded', '?')}")
        st.caption(f"Conformal: {health.get('conformal_loaded', '?')}")
        st.caption(f"Fairness ref: {health.get('fairness_ref_loaded', '?')}")
    else:
        st.error("❌ API not reachable. Start with: `make serve`")

    st.divider()
    meta = _api_get("/v1/model/metadata")
    if meta:
        st.caption(f"Model: `{meta.get('model_version', 'unknown')}`")
        geo_pr = meta.get("pr_auc_geo")
        if geo_pr:
            st.caption(f"Geo PR-AUC: {geo_pr:.4f}")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Priority Queue", "🔍 Explain Prediction", "⚖️ Fairness Report", "📈 Cost Curve"]
)

# ---------------------------------------------------------------------------
# Tab 1: Priority Queue
# ---------------------------------------------------------------------------

with tab1:
    st.header("Inspection Priority Queue")
    st.markdown(
        "Upload a CSV of property IDs to get a ranked inspection queue, or use the live queue endpoint."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        budget = st.number_input(
            "Budget ($)", min_value=1000, max_value=10_000_000, value=100_000, step=10_000
        )
        limit = st.number_input("Max properties", min_value=10, max_value=5000, value=500, step=50)
        cost_per = st.number_input(
            "Cost per inspection ($)", min_value=100, max_value=5000, value=500
        )
        load_btn = st.button("🔄 Load Priority Queue", type="primary")

    with col2:
        uploaded = st.file_uploader(
            "Upload CSV (optional: must have 'property_id' column)", type=["csv"]
        )

    if load_btn:
        with st.spinner("Fetching priority queue from API..."):
            if uploaded is not None:
                # CSV upload mode → use /v1/predict
                df_upload = pd.read_csv(uploaded)
                if "property_id" not in df_upload.columns:
                    st.error("CSV must have a 'property_id' column")
                else:
                    pids = df_upload["property_id"].astype(str).tolist()[:500]
                    result = _api_post("/v1/predict", {"property_ids": pids})
                    if result:
                        preds = result.get("predictions", [])
                        rows = [
                            {
                                "Property ID": p["property_id"],
                                "Priority Score": p["priority_score"],
                                "P(Lead)": p["p_lead_calibrated"],
                                "Uncertainty": p["uncertainty_score"],
                                "Conformal Set": ", ".join(p["conformal_set"]),
                                "Top Feature": p["shap_top_features"][0]["feature"]
                                if p["shap_top_features"]
                                else "—",
                            }
                            for p in sorted(preds, key=lambda x: -x["priority_score"])
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                # Live queue mode
                result = _api_get(
                    "/v1/priority-queue",
                    params={"budget_usd": budget, "limit": limit, "cost_per_inspection": cost_per},
                )
                if result:
                    items = result.get("items", [])
                    st.metric("Properties within budget", result.get("properties_within_budget", 0))
                    st.metric("Total ranked", result.get("total_properties_ranked", 0))
                    if items:
                        df_q = pd.DataFrame(
                            [
                                {
                                    "Rank": i["rank"],
                                    "Property ID": i["property_id"],
                                    "Address": i.get("address", "—"),
                                    "Priority Score": i["priority_score"],
                                    "P(Lead)": i["p_lead_calibrated"],
                                    "Uncertainty": i["uncertainty_score"],
                                    "Est. Cost ($)": i["estimated_cost_usd"],
                                }
                                for i in items
                            ]
                        )
                        st.dataframe(df_q, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Explain Prediction
# ---------------------------------------------------------------------------

with tab2:
    st.header("Explain a Single Prediction")
    pid = st.text_input("Property ID", placeholder="e.g. chi-00000001")

    if st.button("🔍 Explain", type="primary") and pid:
        with st.spinner(f"Fetching prediction for {pid}..."):
            result = _api_get(f"/v1/properties/{pid}/prediction")
            if result:
                col1, col2, col3 = st.columns(3)
                col1.metric("P(Lead)", f"{result['p_lead_calibrated']:.3f}")
                col2.metric("Uncertainty", f"{result['uncertainty_score']:.3f}")
                col3.metric("Priority Score", f"{result['priority_score']:.3f}")

                st.info(
                    f"**Conformal Set:** {', '.join(result['conformal_set'])} (at {result['confidence_level'] * 100:.0f}% confidence)"
                )

                st.subheader("Top SHAP Feature Contributions")
                feats = result.get("shap_top_features", [])
                if feats:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    names = [f["feature"] for f in feats]
                    vals = [f["contribution"] for f in feats]
                    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]
                    ax.barh(names, vals, color=colors)
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set_xlabel("SHAP contribution (positive = increases P(Lead))")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("No SHAP features available")

# ---------------------------------------------------------------------------
# Tab 3: Fairness Report
# ---------------------------------------------------------------------------

with tab3:
    st.header("Fairness Audit Report")
    report = _api_get("/v1/fairness-report")
    if report:
        col1, col2 = st.columns(2)
        col1.metric(
            "FNR Disparity (pp)",
            f"{report.get('fnr_disparity_pp', 0):.1f}",
            delta="⚠️ Flagged" if report.get("disparity_flagged") else "✅ OK",
        )
        col2.metric("Properties with Quartile", report.get("n_properties_with_quartile", "—"))

        st.subheader("False Negative Rate by Income Quartile")
        fnr = report.get("fnr_by_quartile", {})
        if fnr:
            df_fnr = pd.DataFrame(
                [
                    {
                        "Income Quartile": f"Q{k} ({'Lowest' if k == '1' else 'Highest' if k == '4' else 'Middle'})",
                        "FNR": round(v * 100, 1),
                    }
                    for k, v in sorted(fnr.items())
                    if not isinstance(v, float) or not __import__("math").isnan(v)
                ]
            )
            st.bar_chart(df_fnr.set_index("Income Quartile"))

        if report.get("disparity_flagged"):
            st.warning(
                f"⚠️ **Disparity flagged**: {report['fnr_disparity_pp']:.1f} percentage point gap "
                f"across income quartiles exceeds the 5 pp threshold. "
                "This does not block model use but must be reported."
            )
        else:
            st.success(
                "✅ FNR disparity within 5 percentage point threshold across all income quartiles."
            )

        with st.expander("Equity Boost Sample (first 10 tracts)"):
            eq_sample = report.get("equity_boost_sample", {})
            st.json(eq_sample)

# ---------------------------------------------------------------------------
# Tab 4: Cost Curve
# ---------------------------------------------------------------------------

with tab4:
    st.header("Active Learning Cost Curve")
    st.markdown(
        "Shows PR-AUC vs. cumulative inspections for uncertainty-driven vs. random sampling strategies."
    )

    curve_path = Path("reports/active_learning_curve.csv")
    if curve_path.exists():
        df_curve = pd.read_csv(curve_path)
        if not df_curve.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for strategy, color, label in [
                ("uncertainty", "#d73027", "Uncertainty-driven (LeadGuard)"),
                ("random", "#4575b4", "Random baseline"),
            ]:
                sub = df_curve[df_curve["strategy"] == strategy].dropna(subset=["pr_auc"])
                if not sub.empty:
                    ax.plot(
                        sub["cumulative_inspections"], sub["pr_auc"], "o-", color=color, label=label
                    )
            ax.set_xlabel("Cumulative Inspections")
            ax.set_ylabel("PR-AUC")
            ax.set_title("Active Learning: PR-AUC vs. Cumulative Inspections")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Raw data"):
                st.dataframe(df_curve, use_container_width=True)
    else:
        st.info(
            "Active learning curve not yet generated. Run Phase 7 first (`python -m leadguard.models.active_learning`)."
        )
