import streamlit as st
import plotly.graph_objects as go
from prediction_helper import predict

st.set_page_config(
    page_title="Lauki Finance: Credit Risk Modelling",
    page_icon="📊",
    layout="wide"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Lauki Finance")
    st.caption("Credit Risk Assessment Tool")
    st.divider()

    st.subheader("📖 How to Use")
    st.markdown("""
1. Fill in the applicant's **personal**, **loan**, and **credit history** details.
2. Click **Calculate Risk** to run the model.
3. Review the **Credit Score**, **Default Probability**, and **Rating**.
4. Use the **Risk Factor Breakdown** to see which inputs drive the outcome.
""")

    st.divider()
    st.subheader("📋 Input Field Guide")

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

**Delinquency Ratio (%)** — % of loan months with a missed payment. E.g., 30 means 30% of months had a late payment.

**Credit Utilization (%)** — % of available credit currently in use. Above 30% is elevated risk.

**Open Loan Accounts** — Number of currently active loan accounts (1–4).
""")

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

    st.divider()
    st.caption("Model: Logistic Regression")
    st.caption("AUC: 0.98  ·  Gini: 0.96")
    st.caption("Balancing: SMOTETomek  ·  Tuning: Optuna")

# ── Main ──────────────────────────────────────────────────────────────────────────────
st.title("Lauki Finance: Credit Risk Modelling")
st.markdown("Enter the applicant's details below to generate a credit risk assessment and score.")
st.divider()

st.subheader("Applicant Details")

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28,
                          help="Applicant's age. Younger applicants tend to have higher default rates.")
with row1[1]:
    income = st.number_input('Income (₹)', min_value=0, value=1200000,
                             help="Annual gross income in Indian Rupees.")
with row1[2]:
    loan_amount = st.number_input('Loan Amount (₹)', min_value=0, value=2560000,
                                  help="Total loan amount being requested.")

loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row2[0]:
    st.metric(
        label="Loan-to-Income Ratio",
        value=f"{loan_to_income_ratio:.2f}",
        delta="⚠️ High Risk" if loan_to_income_ratio > 3 else "✅ Acceptable",
        delta_color="inverse" if loan_to_income_ratio > 3 else "off",
        help="Auto-calculated: Loan Amount ÷ Income. Values above 3× indicate elevated risk."
    )
with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36,
                                         help="Total loan repayment period in months.")
with row2[2]:
    avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20,
                                              help="Average Days Past Due per delinquent month. Higher values mean more severe late payments.")

with row3[0]:
    delinquency_ratio = st.number_input('Delinquency Ratio (%)', min_value=0, max_value=100, step=1, value=30,
                                        help="Percentage of loan months where a payment was missed or late (0–100).")
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization (%)', min_value=0, max_value=100, step=1, value=30,
                                               help="Percentage of total available credit currently in use. Above 30% is considered elevated.")
with row3[2]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2,
                                        help="Number of currently active loan accounts (1–4).")

with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'],
                                  help="Whether the applicant owns their home, rents, or has a mortgage.")
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'],
                                help="The stated purpose for the loan.")
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'],
                             help="Secured loans are backed by collateral. Unsecured loans carry higher default risk.")

st.divider()

# ── Calculate ───────────────────────────────────────────────────────────────────────────
if st.button('Calculate Risk', type='primary', use_container_width=True):
    probability, credit_score, rating, contributions = predict(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    RATING_CONFIG = {
        'Poor':      {'color': '#c62828', 'bg': '#ffebee', 'emoji': '🔴'},
        'Average':   {'color': '#e65100', 'bg': '#fff3e0', 'emoji': '🟠'},
        'Good':      {'color': '#558b2f', 'bg': '#f1f8e9', 'emoji': '🟢'},
        'Excellent': {'color': '#1565c0', 'bg': '#e3f2fd', 'emoji': '🌟'},
    }
    cfg = RATING_CONFIG.get(rating, {'color': '#333333', 'bg': '#f5f5f5', 'emoji': '⚪'})

    st.subheader("Assessment Results")
    res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1.5])

    # ── Credit Score Gauge ──
    with res_col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            title={'text': "Credit Score", 'font': {'size': 15}},
            number={'font': {'size': 44, 'color': cfg['color']}},
            gauge={
                'axis': {'range': [300, 900], 'tickwidth': 1, 'tickcolor': '#888'},
                'bar': {'color': cfg['color'], 'thickness': 0.25},
                'bgcolor': 'white',
                'steps': [
                    {'range': [300, 500], 'color': '#ffcdd2'},
                    {'range': [500, 650], 'color': '#ffe0b2'},
                    {'range': [650, 750], 'color': '#dcedc8'},
                    {'range': [750, 900], 'color': '#bbdefb'},
                ],
                'threshold': {
                    'line': {'color': cfg['color'], 'width': 4},
                    'thickness': 0.8,
                    'value': credit_score
                }
            }
        ))
        fig.update_layout(
            height=260,
            margin=dict(t=30, b=0, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Rating Card ──
    with res_col2:
        st.markdown(f"""
<div style="
    background-color:{cfg['bg']};
    border-left: 5px solid {cfg['color']};
    border-radius: 8px;
    padding: 1.5rem 1.2rem;
    height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: center;
">
    <div style="font-size:2.5rem; margin-bottom:0.3rem">{cfg['emoji']}</div>
    <div style="font-size:2rem; font-weight:700; color:{cfg['color']}; line-height:1">{rating}</div>
    <div style="color:#666; margin-top:0.4rem; font-size:0.9rem">Credit Rating</div>
    <hr style="margin:0.8rem 0; border-color:{cfg['color']}40">
    <div style="font-size:0.95rem; color:#444">
        Default Probability<br>
        <strong style="font-size:1.7rem; color:{cfg['color']}">{probability:.1%}</strong>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Risk Factor Breakdown ──
    with res_col3:
        st.markdown("**Risk Factor Breakdown**")
        st.caption("🔴 Increases default risk  ·  🟢 Reduces default risk")

        FEATURE_LABELS = {
            'age': 'Age',
            'loan_tenure_months': 'Loan Tenure',
            'number_of_open_accounts': 'Open Accounts',
            'credit_utilization_ratio': 'Credit Utilization',
            'loan_to_income': 'Loan-to-Income Ratio',
            'delinquency_ratio': 'Delinquency Ratio',
            'avg_dpd_per_delinquency': 'Avg DPD',
            'residence_type_Owned': 'Residence: Owned',
            'residence_type_Rented': 'Residence: Rented',
            'loan_purpose_Education': 'Purpose: Education',
            'loan_purpose_Home': 'Purpose: Home',
            'loan_purpose_Personal': 'Purpose: Personal',
            'loan_type_Unsecured': 'Loan Type: Unsecured',
        }

        top_idx = contributions.abs().nlargest(7).index
        contrib_top = contributions[top_idx].sort_values()
        labels = [FEATURE_LABELS.get(f, f) for f in contrib_top.index]
        colors = ['#c62828' if v > 0 else '#2e7d32' for v in contrib_top.values]

        fig2 = go.Figure(go.Bar(
            x=contrib_top.values,
            y=labels,
            orientation='h',
            marker_color=colors,
        ))
        fig2.update_layout(
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True, gridcolor='#eee',
                zeroline=True, zerolinecolor='#aaa', zerolinewidth=1.5,
                title=None
            ),
            yaxis=dict(title=None),
        )
        st.plotly_chart(fig2, use_container_width=True)
