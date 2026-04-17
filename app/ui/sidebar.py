"""
sidebar.py — Streamlit sidebar content.

Exports:
    render_sidebar()  — Renders the full sidebar: branding header, step-by-step
                        usage guide, input field reference, result interpretation
                        table, SHAP background explainer, and model metadata footer.
"""

import streamlit as st


def render_sidebar() -> None:
    """Render the full application sidebar.

    The sidebar is split into collapsible expanders grouped by topic, so users
    can look up guidance for the specific section they need without scrolling
    through unrelated content.
    """
    with st.sidebar:
        # ── Branding Header ───────────────────────────────────────────────────
        st.title("📊 Quantum Finance")
        st.caption("Credit Risk Assessment Tool")
        st.divider()

        # ── Step-by-step Usage Guide ──────────────────────────────────────────
        st.subheader("📖 How to Use")
        st.markdown("""
1. Fill in the applicant's **personal**, **loan**, and **credit history** details.
2. Click **Calculate Risk** to run the model.
3. Review the **Credit Score**, **Default Probability**, and **Rating**.
4. Use **What-If Simulator** to explore how changes affect the score.
5. Download a **PDF report** for record-keeping.
6. Use **Batch Scoring** tab to score multiple applicants at once.
""")

        st.divider()
        st.subheader("📋 Input Field Guide")

        # ── Input Field Descriptions ──────────────────────────────────────────
        with st.expander("Personal & Loan Details"):
            st.markdown("""
**Age** — Applicant's age (18–100). Younger applicants statistically default more often.

**Income** — Annual gross income (₹). Higher income lowers default risk.

**Loan Amount** — Total loan amount requested.

**Loan-to-Income Ratio** — Auto-calculated (Loan ÷ Income). Above 3× is high risk.

**Loan Tenure** — Repayment period in months.

**Loan Purpose** — Education / Home / Auto / Personal.

**Loan Type** — *Secured* loans are backed by collateral; *Unsecured* carry higher risk.

**Residence Type** — Owned / Rented / Mortgage.
""")

        with st.expander("Credit History Details"):
            st.markdown("""
**Avg DPD (Days Past Due)** — Average days late per missed payment. Higher = riskier.

**Delinquency Ratio (%)** — % of loan months with a missed payment.

**Credit Utilization (%)** — % of available credit currently in use. Above 30% is elevated risk.

**Open Loan Accounts** — Number of currently active loan accounts (1–4).
""")

        # ── Result Interpretation ─────────────────────────────────────────────
        with st.expander("How to Interpret Results"):
            st.markdown("""
| Score | Rating | Risk Level |
|---|---|---|
| 750 – 900 | 🌟 Excellent | Very Low |
| 650 – 749 | 🟢 Good | Low |
| 500 – 649 | 🟠 Average | Medium |
| 300 – 499 | 🔴 Poor | High |

**Default Probability** is the model's estimated chance the applicant will fail to repay.
""")

        # ── SHAP Background Explainer ─────────────────────────────────────────
        with st.expander("🔍 What is SHAP? (Risk Factor Explanation)"):
            st.markdown("""
**SHAP** stands for **SH**apley **A**dditive ex**P**lanations.

It is a mathematical technique borrowed from game theory that answers one question every banking officer needs to ask:

> *"Why did this applicant receive this specific credit score?"*

**How it works — in plain terms:**

Every applicant starts from a baseline score (the average across all applicants). SHAP then calculates how much each individual input — age, delinquency history, credit utilization, etc. — **pushed the score up or down** from that baseline for *this specific person*.

The result is a line-item receipt:
- 🔴 **Red bars** = factors that *increased* default risk (lowered the score)
- 🟢 **Green bars** = factors that *reduced* default risk (raised the score)
- **Bar length** = how strongly that factor influenced the outcome

**Why it matters in banking:**

Under regulations such as the **Equal Credit Opportunity Act (ECOA)** and **GDPR Article 22**, lenders may be legally required to provide applicants with a specific reason for an adverse credit decision. SHAP makes this possible automatically — it produces auditable, per-decision explanations rather than vague model-level statistics.
""")

        # ── Model Metadata Footer ─────────────────────────────────────────────
        st.divider()
        st.caption("Model: Logistic Regression")
        st.caption("AUC: 0.98  ·  Gini: 0.96")
        st.caption("Balancing: SMOTETomek  ·  Tuning: Optuna")
