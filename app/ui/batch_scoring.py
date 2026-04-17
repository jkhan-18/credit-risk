"""
batch_scoring.py — Tab 2: Multi-applicant batch CSV scoring UI.

Exports:
    render_batch_scoring_tab()  — Renders the full batch scoring workflow:
                                  CSV template download → file upload → column
                                  validation → scoring trigger → colour-coded
                                  results table → CSV export → summary dashboard.
"""

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import BATCH_COLUMNS
from model.predictor import predict_batch


def render_batch_scoring_tab() -> None:
    """Render Tab 2: batch applicant CSV scoring workflow.

    Steps presented to the user:
        1. Download a pre-filled CSV template (one example row).
        2. Upload a CSV file with applicant data.
        3. Validation: check for missing required columns.
        4. Click 'Score All Applicants' to run the model.
        5. View colour-coded results table and download as CSV.
        6. View a summary metrics row and a rating distribution bar chart.
    """
    st.subheader("Batch Scoring")
    st.markdown("Score multiple applicants at once by uploading a CSV file.")

    # ── Step 1: CSV Template Download ─────────────────────────────────────────
    # Pre-fill one example row so users don't have to guess column names or
    # valid value formats.  The column order matches BATCH_COLUMNS exactly.
    template_df = pd.DataFrame([{
        "age":                      28,
        "income":                   1_200_000,
        "loan_amount":              2_560_000,
        "loan_tenure_months":       36,
        "avg_dpd_per_delinquency":  20,
        "delinquency_ratio":        30,
        "credit_utilization_ratio": 30,
        "num_open_accounts":        2,
        "residence_type":           "Owned",
        "loan_purpose":             "Personal",
        "loan_type":                "Unsecured",
    }])

    st.download_button(
        label="⬇️ Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="batch_template.csv",
        mime="text/csv",
    )
    st.caption("Required columns: " + ", ".join(BATCH_COLUMNS))

    # ── Step 2: File Upload ───────────────────────────────────────────────────
    uploaded_file = st.file_uploader("Upload applicant CSV", type=["csv"])

    if uploaded_file is None:
        # Nothing more to render until the user uploads a file.
        return

    input_df = pd.read_csv(uploaded_file)

    # ── Step 3: Column Validation ─────────────────────────────────────────────
    # Identify any required columns that are absent before attempting to score.
    missing = [col for col in BATCH_COLUMNS if col not in input_df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.info(f"Found **{len(input_df)} applicants**. Click below to score.")

    if not st.button("Score All Applicants", type="primary", use_container_width=True):
        # Wait for the user to explicitly trigger scoring.
        return

    # ── Step 4: Batch Scoring ─────────────────────────────────────────────────
    with st.spinner("Scoring applicants..."):
        # Pass only the required columns in the correct order to the scorer.
        results_df = predict_batch(input_df[BATCH_COLUMNS])

    # Merge original input columns with the three scoring result columns.
    output_df = pd.concat([input_df.reset_index(drop=True), results_df], axis=1)

    # ── Step 5a: Colour-coded Results Table ───────────────────────────────────
    def _highlight_rating(val: str) -> str:
        """Return a CSS style string for a rating cell, keyed on the rating value."""
        palette = {
            "Poor":      "background-color: #c62828; color: #ffffff; font-weight: bold",
            "Average":   "background-color: #e65100; color: #ffffff; font-weight: bold",
            "Good":      "background-color: #558b2f; color: #ffffff; font-weight: bold",
            "Excellent": "background-color: #1565c0; color: #ffffff; font-weight: bold",
        }
        return palette.get(val, "")

    st.dataframe(
        output_df.style.map(_highlight_rating, subset=["rating"]),
        use_container_width=True,
    )

    # ── Step 5b: Results CSV Download ─────────────────────────────────────────
    st.download_button(
        label="⬇️ Download Results CSV",
        data=output_df.to_csv(index=False),
        file_name=f"batch_results_{date.today()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Step 6: Summary Dashboard ─────────────────────────────────────────────
    _render_batch_summary(results_df, len(input_df))


def _render_batch_summary(results_df: pd.DataFrame, total_count: int) -> None:
    """Render the summary metrics strip and rating distribution bar chart.

    Args:
        results_df:   DataFrame containing 'credit_score', 'default_probability',
                      and 'rating' columns (output of predict_batch()).
        total_count:  Total number of rows in the original uploaded file,
                      including any rows that resulted in errors.
    """
    st.subheader("Batch Summary")

    # Exclude error rows (those where rating is not a valid category string).
    valid = results_df[results_df["rating"].isin(["Poor", "Average", "Good", "Excellent"])]

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Applicants", total_count)
    with s2:
        avg_sc = valid["credit_score"].mean()
        st.metric("Avg Credit Score", f"{avg_sc:.0f}" if not pd.isna(avg_sc) else "N/A")
    with s3:
        st.metric("High Risk (Poor)", int((valid["rating"] == "Poor").sum()))
    with s4:
        st.metric("Low Risk (Good+)", int(valid["rating"].isin(["Good", "Excellent"]).sum()))

    if len(valid) == 0:
        # No valid results to chart — nothing more to render.
        return

    # ── Rating Distribution Bar Chart ─────────────────────────────────────────
    # Ensure all four categories appear even when some have a zero count.
    rating_counts = valid["rating"].value_counts().reindex(
        ["Excellent", "Good", "Average", "Poor"], fill_value=0
    )

    fig_dist = go.Figure(go.Bar(
        x=rating_counts.index,
        y=rating_counts.values,
        # Colours match the RATING_CONFIG palette for visual consistency.
        marker_color=["#1565c0", "#558b2f", "#e65100", "#c62828"],
        text=rating_counts.values,
        textposition="outside",
    ))
    fig_dist.update_layout(
        title="Rating Distribution",
        height=300,
        margin=dict(t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="#eee"),
        xaxis=dict(title=None),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
