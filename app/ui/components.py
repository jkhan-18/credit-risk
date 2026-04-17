"""
components.py — Reusable Streamlit UI components for displaying prediction results.

Exports:
    render_results()  — Renders the three-column results panel:
                        (1) Plotly gauge chart, (2) rating badge card,
                        (3) SHAP horizontal bar chart.

These components are shared between the single-applicant tab and any future
tabs that need to display a prediction result.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import FEATURE_LABELS, RATING_CONFIG


def render_results(
    probability: float,
    credit_score: int,
    rating: str,
    shap_values: pd.Series,
) -> None:
    """Render the three-column prediction results panel.

    Layout:
        Column 1 (wide)   – Plotly gauge chart with colour-coded score bands.
        Column 2 (medium) – Rating badge card showing rating emoji, label,
                            and default probability percentage.
        Column 3 (widest) – Horizontal SHAP bar chart for the top 7 features.

    Args:
        probability:  Model's estimated default probability (0.0–1.0).
        credit_score: Credit score in the 300–900 range.
        rating:       Categorical rating ('Poor', 'Average', 'Good', 'Excellent').
        shap_values:  Per-feature SHAP contributions as a pandas Series indexed
                      by feature name (from config.FEATURE_LABELS keys).
    """
    # Look up the colour and emoji for this rating tier.
    cfg = RATING_CONFIG.get(rating, {"color": "#333333", "bg": "#f5f5f5", "emoji": "⚪"})

    col_gauge, col_badge, col_shap = st.columns([1.2, 1, 1.5])

    # ── Column 1: Credit Score Gauge ─────────────────────────────────────────
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            title={"text": "Credit Score", "font": {"size": 15}},
            number={"font": {"size": 44, "color": cfg["color"]}},
            gauge={
                "axis": {"range": [300, 900], "tickwidth": 1, "tickcolor": "#888"},
                "bar":  {"color": cfg["color"], "thickness": 0.25},
                "bgcolor": "white",
                # Four coloured bands corresponding to rating tiers.
                "steps": [
                    {"range": [300, 500], "color": "#ffcdd2"},  # Poor    – light red
                    {"range": [500, 650], "color": "#ffe0b2"},  # Average – light orange
                    {"range": [650, 750], "color": "#dcedc8"},  # Good    – light green
                    {"range": [750, 900], "color": "#bbdefb"},  # Excellent – light blue
                ],
                # Vertical threshold line at the applicant's exact score.
                "threshold": {
                    "line": {"color": cfg["color"], "width": 4},
                    "thickness": 0.8,
                    "value": credit_score,
                },
            },
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(t=30, b=0, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Column 2: Rating Badge Card ───────────────────────────────────────────
    with col_badge:
        # Rendered as a styled HTML div because Streamlit's native metric widget
        # does not support the card-style layout with a coloured left border.
        st.markdown(
            f"""
<div style="
    background-color:{cfg['bg']};
    border-left:5px solid {cfg['color']};
    border-radius:8px;
    padding:1.5rem 1.2rem;
    height:240px;
    display:flex;
    flex-direction:column;
    justify-content:center;">
  <div style="font-size:2.5rem;margin-bottom:0.3rem">{cfg['emoji']}</div>
  <div style="font-size:2rem;font-weight:700;color:{cfg['color']};line-height:1">{rating}</div>
  <div style="color:#666;margin-top:0.4rem;font-size:0.9rem">Credit Rating</div>
  <hr style="margin:0.8rem 0;border-color:{cfg['color']}40">
  <div style="font-size:0.95rem;color:#444">
    Default Probability<br>
    <strong style="font-size:1.7rem;color:{cfg['color']}">{probability:.1%}</strong>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    # ── Column 3: SHAP Feature Contribution Bar Chart ─────────────────────────
    with col_shap:
        st.markdown(
            "**Top Risk Factors — SHAP Analysis**",
            help=(
                "SHAP (SHapley Additive exPlanations) shows how much each input "
                "pushed this applicant's score up or down from the baseline. "
                "Red = increased default risk. Green = reduced default risk."
            ),
        )
        st.caption("🔴 Increases default risk  ·  🟢 Reduces default risk  ·  Bar length = strength of influence")

        # Select the 7 features with the highest absolute SHAP values.
        top_idx      = shap_values.abs().nlargest(7).index
        contrib_top  = shap_values[top_idx].sort_values()
        labels       = [FEATURE_LABELS.get(f, f) for f in contrib_top.index]

        # Red for risk-increasing (positive SHAP) features; green for risk-reducing.
        colors = ["#c62828" if v > 0 else "#2e7d32" for v in contrib_top.values]

        fig_shap = go.Figure(go.Bar(
            x=contrib_top.values,
            y=labels,
            orientation="h",
            marker_color=colors,
        ))
        fig_shap.update_layout(
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True, gridcolor="#eee",
                zeroline=True, zerolinecolor="#aaa", zerolinewidth=1.5,
                title=None,
            ),
            yaxis=dict(title=None),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
