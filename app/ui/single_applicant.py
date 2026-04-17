"""
single_applicant.py — Tab 1: Single-applicant credit risk assessment UI.

Exports:
    render_single_applicant_tab()  — Renders the 12-field input form, triggers
                                     prediction on button click, displays results
                                     (gauge, rating card, SHAP chart), a
                                     collapsible What-If simulator, and a
                                     PDF report download button.
"""

from datetime import date

import streamlit as st

from config import RATING_CONFIG
from model.predictor import predict
from reports.pdf_generator import generate_pdf
from ui.components import render_results


def render_single_applicant_tab() -> None:
    """Render the single-applicant scoring form and its results section.

    The form is laid out as four rows of three columns (12 input fields total).
    Prediction results are stored in st.session_state so they survive Streamlit
    reruns triggered by downstream widget interactions (e.g., What-If sliders).
    """
    st.subheader("Applicant Details")

    # ── Input Grid: 4 rows × 3 columns ───────────────────────────────────────
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)

    with row1[0]:
        age = st.number_input(
            "Age", min_value=18, max_value=100, step=1, value=28,
            help="Applicant's age. Younger applicants tend to have higher default rates.",
        )
    with row1[1]:
        income = st.number_input(
            "Income (₹)", min_value=1, value=1_200_000,
            help="Annual gross income in Indian Rupees.",
        )
    with row1[2]:
        loan_amount = st.number_input(
            "Loan Amount (₹)", min_value=0, value=2_560_000,
            help="Total loan amount being requested.",
        )

    # Auto-derived ratio shown as a live metric — no user input required.
    loan_to_income_ratio = loan_amount / income if income > 0 else 0.0

    with row2[0]:
        st.metric(
            label="Loan-to-Income Ratio",
            value=f"{loan_to_income_ratio:.2f}",
            delta="⚠️ High Risk" if loan_to_income_ratio > 3 else "✅ Acceptable",
            # 'inverse' makes the delta text red for high-risk values.
            delta_color="inverse" if loan_to_income_ratio > 3 else "off",
            help="Auto-calculated: Loan Amount ÷ Income. Values above 3× indicate elevated risk.",
        )
    with row2[1]:
        loan_tenure_months = st.number_input(
            "Loan Tenure (months)", min_value=0, step=1, value=36,
            help="Total loan repayment period in months.",
        )
    with row2[2]:
        avg_dpd_per_delinquency = st.number_input(
            "Avg DPD", min_value=0, value=20,
            help="Average Days Past Due per delinquent month.",
        )

    with row3[0]:
        delinquency_ratio = st.number_input(
            "Delinquency Ratio (%)", min_value=0, max_value=100, step=1, value=30,
            help="% of loan months where a payment was missed or late (0–100).",
        )
    with row3[1]:
        credit_utilization_ratio = st.number_input(
            "Credit Utilization (%)", min_value=0, max_value=100, step=1, value=30,
            help="% of total available credit currently in use.",
        )
    with row3[2]:
        num_open_accounts = st.number_input(
            "Open Loan Accounts", min_value=1, max_value=4, step=1, value=2,
            help="Number of currently active loan accounts (1–4).",
        )

    with row4[0]:
        residence_type = st.selectbox(
            "Residence Type", ["Owned", "Rented", "Mortgage"],
            help="Whether the applicant owns, rents, or has a mortgage.",
        )
    with row4[1]:
        loan_purpose = st.selectbox(
            "Loan Purpose", ["Education", "Home", "Auto", "Personal"],
            help="The stated purpose for the loan.",
        )
    with row4[2]:
        loan_type = st.selectbox(
            "Loan Type", ["Unsecured", "Secured"],
            help="Secured loans are backed by collateral. Unsecured carry higher risk.",
        )

    st.divider()

    # ── Prediction Trigger ────────────────────────────────────────────────────
    # On click: run the full inference pipeline and cache results to session_state.
    # Using session_state ensures the results panel stays visible even after
    # Streamlit reruns caused by interactions with the What-If sliders below.
    if st.button("Calculate Risk", type="primary", use_container_width=True):
        probability, credit_score, rating, shap_values = predict(
            age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type,
        )
        st.session_state["prediction_result"] = (probability, credit_score, rating, shap_values)
        st.session_state["prediction_inputs"] = {
            "age":                      age,
            "income":                   income,
            "loan_amount":              loan_amount,
            "loan_tenure_months":       loan_tenure_months,
            "avg_dpd_per_delinquency":  avg_dpd_per_delinquency,
            "delinquency_ratio":        delinquency_ratio,
            "credit_utilization_ratio": credit_utilization_ratio,
            "num_open_accounts":        num_open_accounts,
            "residence_type":           residence_type,
            "loan_purpose":             loan_purpose,
            "loan_type":                loan_type,
        }

    # ── Results Display ───────────────────────────────────────────────────────
    # Render only when a prediction has been made (i.e., the keys exist in state).
    if "prediction_result" in st.session_state:
        probability, credit_score, rating, shap_values = st.session_state["prediction_result"]
        inputs = st.session_state["prediction_inputs"]

        # Place the PDF download button in the top-right alongside the heading.
        res_hdr, pdf_hdr = st.columns([3, 1])
        with res_hdr:
            st.subheader("Assessment Results")
        with pdf_hdr:
            pdf_bytes = generate_pdf(inputs, probability, credit_score, rating, shap_values)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"credit_report_{date.today()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # Delegate to the shared results component (gauge + badge + SHAP chart).
        render_results(probability, credit_score, rating, shap_values)

        st.divider()

        # ── What-If Simulator ─────────────────────────────────────────────────
        _render_what_if_simulator(inputs, probability, credit_score, rating)


def _render_what_if_simulator(
    inputs: dict,
    original_probability: float,
    original_score: int,
    original_rating: str,
) -> None:
    """Render the collapsible What-If Simulator inside an expander.

    Lets the user adjust six key inputs via sliders and instantly see how the
    credit score would change, without modifying the persisted prediction result.

    Args:
        inputs:               Original form inputs from st.session_state.
        original_probability: Default probability from the persisted prediction.
        original_score:       Credit score from the persisted prediction.
        original_rating:      Rating string from the persisted prediction.
    """
    with st.expander("🔧 What-If Simulator — Explore how changes affect the score"):
        st.caption(
            "Adjust sliders to see how changes in the applicant's profile "
            "would impact the credit score."
        )

        w_col1, w_col2, w_col3 = st.columns(3)

        with w_col1:
            w_avg_dpd = st.slider(
                "Avg DPD", 0, 200, inputs["avg_dpd_per_delinquency"],
                help="Adjust average days past due per delinquent month.",
            )
            w_delinquency = st.slider(
                "Delinquency Ratio (%)", 0, 100, inputs["delinquency_ratio"],
                help="Adjust % of months with late payments.",
            )
        with w_col2:
            w_credit_util = st.slider(
                "Credit Utilization (%)", 0, 100, inputs["credit_utilization_ratio"],
                help="Adjust % of credit in use.",
            )
            w_open_accounts = st.slider(
                "Open Loan Accounts", 1, 4, inputs["num_open_accounts"],
            )
        with w_col3:
            w_loan_tenure = st.slider(
                "Loan Tenure (months)", 0, 360, inputs["loan_tenure_months"],
            )
            w_age = st.slider("Age", 18, 100, inputs["age"])

        # Re-run the model with the modified values; all other inputs stay fixed.
        w_prob, w_score, w_rating, _ = predict(
            w_age, inputs["income"], inputs["loan_amount"], w_loan_tenure,
            w_avg_dpd, w_delinquency, w_credit_util, w_open_accounts,
            inputs["residence_type"], inputs["loan_purpose"], inputs["loan_type"],
        )

        # Score delta: positive = simulated score improved; negative = worsened.
        delta = w_score - original_score
        w_cfg = RATING_CONFIG.get(w_rating, {"color": "#333333", "bg": "#f5f5f5", "emoji": "⚪"})

        st.divider()
        sim_c1, sim_c2, sim_c3 = st.columns(3)

        with sim_c1:
            st.metric(
                "Original Score", original_score,
                help=f"Rating: {original_rating}  |  Default Prob: {original_probability:.1%}",
            )
        with sim_c2:
            st.metric(
                "Simulated Score", w_score,
                delta=delta,
                delta_color="normal",
                help=f"Rating: {w_rating}  |  Default Prob: {w_prob:.1%}",
            )
        with sim_c3:
            # Styled badge card showing the simulated rating.
            st.markdown(
                f"""
<div style="
    background-color:{w_cfg['bg']};
    border-left:4px solid {w_cfg['color']};
    border-radius:6px;
    padding:0.6rem 1rem;
    margin-top:0.25rem;">
  <div style="font-size:1.1rem;font-weight:700;color:{w_cfg['color']}">{w_cfg['emoji']} {w_rating}</div>
  <div style="font-size:0.85rem;color:#555">Default Prob: <strong>{w_prob:.1%}</strong></div>
</div>""",
                unsafe_allow_html=True,
            )
