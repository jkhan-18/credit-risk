"""
main.py — Streamlit application entry point.

This module is intentionally kept thin.  Its only responsibilities are:
  1. Configure the Streamlit page (title, icon, layout).
  2. Render the sidebar (help guide and model metadata).
  3. Render the page title and divider.
  4. Delegate to the two feature tabs: single applicant and batch scoring.

All UI logic lives in the ui/ sub-package; all inference logic lives in model/.

Run with:
    streamlit run app/main.py
"""

import streamlit as st

from ui.sidebar import render_sidebar
from ui.single_applicant import render_single_applicant_tab
from ui.batch_scoring import render_batch_scoring_tab

# ── Page Configuration ────────────────────────────────────────────────────────
# Must be the very first Streamlit call in the script — before any other st.*
# call, including those in imported modules.
st.set_page_config(
    page_title="Quantum Finance: Credit Risk Modelling",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar()

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("Quantum Finance: Credit Risk Modelling")
st.markdown("Enter the applicant's details below to generate a credit risk assessment and score.")
st.divider()

# ── Feature Tabs ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋  Single Applicant", "📤  Batch Scoring"])

with tab1:
    render_single_applicant_tab()

with tab2:
    render_batch_scoring_tab()
