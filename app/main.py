import io
from datetime import date

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from fpdf import FPDF
from prediction_helper import predict, predict_batch, _BATCH_COLUMNS

st.set_page_config(
    page_title='Quantum Finance: Credit Risk Modelling',
    page_icon='📊',
    layout='wide'
)

# ── Constants ─────────────────────────────────────────────────────────────────
RATING_CONFIG = {
    'Poor':      {'color': '#c62828', 'bg': '#ffebee', 'emoji': '🔴'},
    'Average':   {'color': '#e65100', 'bg': '#fff3e0', 'emoji': '🟠'},
    'Good':      {'color': '#558b2f', 'bg': '#f1f8e9', 'emoji': '🟢'},
    'Excellent': {'color': '#1565c0', 'bg': '#e3f2fd', 'emoji': '🌟'},
}

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


# ── PDF Generator ─────────────────────────────────────────────────────────────
def generate_pdf(inputs, probability, credit_score, rating, shap_values):
    RATING_COLORS_RGB = {
        'Poor':      (198,  40,  40),
        'Average':   (230,  81,   0),
        'Good':      ( 85, 139,  47),
        'Excellent': ( 21, 101, 192),
    }
    rc = RATING_COLORS_RGB.get(rating, (51, 51, 51))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    pdf.set_fill_color(21, 101, 192)
    pdf.rect(0, 0, 210, 24, 'F')
    pdf.set_xy(10, 7)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'QUANTUM FINANCE  -  CREDIT RISK ASSESSMENT REPORT')

    pdf.set_xy(10, 28)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f'Generated: {date.today().strftime("%B %d, %Y")}')

    pdf.set_xy(10, 40)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, 'ASSESSMENT RESULT')

    pdf.set_xy(10, 50)
    pdf.set_fill_color(*rc)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 26)
    pdf.cell(55, 18, str(credit_score), align='C', fill=True)

    pdf.set_xy(70, 50)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(*rc)
    pdf.cell(0, 8, rating)

    pdf.set_xy(70, 60)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f'Default Probability: {probability:.1%}')

    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, 74, 200, 74)

    pdf.set_xy(10, 78)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, 'APPLICANT DETAILS')

    details = [
        ('Age',                str(inputs['age'])),
        ('Income',             f"Rs {inputs['income']:,}"),
        ('Loan Amount',        f"Rs {inputs['loan_amount']:,}"),
        ('Loan Tenure',        f"{inputs['loan_tenure_months']} months"),
        ('Loan-to-Income',     f"{inputs['loan_amount']/inputs['income']:.2f}" if inputs['income'] else 'N/A'),
        ('Loan Purpose',       inputs['loan_purpose']),
        ('Loan Type',          inputs['loan_type']),
        ('Residence Type',     inputs['residence_type']),
        ('Open Accounts',      str(inputs['num_open_accounts'])),
        ('Credit Utilization', f"{inputs['credit_utilization_ratio']}%"),
        ('Delinquency Ratio',  f"{inputs['delinquency_ratio']}%"),
        ('Avg DPD',            str(inputs['avg_dpd_per_delinquency'])),
    ]
    y = 89
    for i, (label, value) in enumerate(details):
        col = i % 2
        row = i // 2
        pdf.set_xy(10 + col * 95, y + row * 8)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(42, 6, label + ':')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(50, 6, value)

    y_div = y + (len(details) // 2 + 1) * 8 + 4
    pdf.line(10, y_div, 200, y_div)

    pdf.set_xy(10, y_div + 4)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, 'TOP RISK FACTORS  (SHAP)')

    top_factors = shap_values.abs().nlargest(5)
    for i, feat in enumerate(top_factors.index):
        val = shap_values[feat]
        label = FEATURE_LABELS.get(feat, feat)
        direction = 'Increases default risk' if val > 0 else 'Reduces default risk'
        indicator = '[+]' if val > 0 else '[-]'
        color = (198, 40, 40) if val > 0 else (46, 125, 50)
        pdf.set_xy(10, y_div + 14 + i * 8)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(*color)
        pdf.cell(0, 6, f'  {i + 1}. {label}  {indicator}  {direction}')

    # Footer — anchored to bottom of page, never overflows
    pdf.set_xy(10, 287)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, 286, 200, 286)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 6, 'Quantum Finance Credit Risk Model  |  AUC: 0.98  |  Gini: 0.96  |  For internal use only')

    return bytes(pdf.output())


# ── Results Renderer ──────────────────────────────────────────────────────────
def render_results(probability, credit_score, rating, shap_values):
    cfg = RATING_CONFIG.get(rating, {'color': '#333333', 'bg': '#f5f5f5', 'emoji': '⚪'})
    res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1.5])

    with res_col1:
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=credit_score,
            title={'text': 'Credit Score', 'font': {'size': 15}},
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
        fig.update_layout(height=260, margin=dict(t=30, b=0, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with res_col2:
        st.markdown(f"""
<div style="background-color:{cfg['bg']};border-left:5px solid {cfg['color']};border-radius:8px;
    padding:1.5rem 1.2rem;height:240px;display:flex;flex-direction:column;justify-content:center;">
  <div style="font-size:2.5rem;margin-bottom:0.3rem">{cfg['emoji']}</div>
  <div style="font-size:2rem;font-weight:700;color:{cfg['color']};line-height:1">{rating}</div>
  <div style="color:#666;margin-top:0.4rem;font-size:0.9rem">Credit Rating</div>
  <hr style="margin:0.8rem 0;border-color:{cfg['color']}40">
  <div style="font-size:0.95rem;color:#444">
    Default Probability<br>
    <strong style="font-size:1.7rem;color:{cfg['color']}">{probability:.1%}</strong>
  </div>
</div>""", unsafe_allow_html=True)

    with res_col3:
        st.markdown('**Top Risk Factors (SHAP)**')
        st.caption('🔴 Increases default risk  ·  🟢 Reduces default risk')
        top_idx = shap_values.abs().nlargest(7).index
        contrib_top = shap_values[top_idx].sort_values()
        labels = [FEATURE_LABELS.get(f, f) for f in contrib_top.index]
        colors = ['#c62828' if v > 0 else '#2e7d32' for v in contrib_top.values]
        fig2 = go.Figure(go.Bar(
            x=contrib_top.values, y=labels, orientation='h', marker_color=colors
        ))
        fig2.update_layout(
            height=260, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#eee', zeroline=True,
                       zerolinecolor='#aaa', zerolinewidth=1.5, title=None),
            yaxis=dict(title=None),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('📊 Quantum Finance')
    st.caption('Credit Risk Assessment Tool')
    st.divider()

    st.subheader('📖 How to Use')
    st.markdown("""
1. Fill in the applicant's **personal**, **loan**, and **credit history** details.
2. Click **Calculate Risk** to run the model.
3. Review the **Credit Score**, **Default Probability**, and **Rating**.
4. Use **What-If Simulator** to explore how changes affect the score.
5. Download a **PDF report** for record-keeping.
6. Use **Batch Scoring** tab to score multiple applicants at once.
""")

    st.divider()
    st.subheader('📋 Input Field Guide')

    with st.expander('Personal & Loan Details'):
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

    with st.expander('Credit History Details'):
        st.markdown("""
**Avg DPD (Days Past Due)** — Average days late per missed payment. Higher = riskier.

**Delinquency Ratio (%)** — % of loan months with a missed payment.

**Credit Utilization (%)** — % of available credit currently in use. Above 30% is elevated risk.

**Open Loan Accounts** — Number of currently active loan accounts (1–4).
""")

    with st.expander('How to Interpret Results'):
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
    st.caption('Model: Logistic Regression')
    st.caption('AUC: 0.98  ·  Gini: 0.96')
    st.caption('Balancing: SMOTETomek  ·  Tuning: Optuna')


# ── Title ─────────────────────────────────────────────────────────────────────
st.title('Quantum Finance: Credit Risk Modelling')
st.markdown("Enter the applicant's details below to generate a credit risk assessment and score.")
st.divider()

tab1, tab2 = st.tabs(['📋  Single Applicant', '📤  Batch Scoring'])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Applicant
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader('Applicant Details')

    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)

    with row1[0]:
        age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28,
                              help="Applicant's age. Younger applicants tend to have higher default rates.")
    with row1[1]:
        income = st.number_input('Income (₹)', min_value=1, value=1200000,
                                 help='Annual gross income in Indian Rupees.')
    with row1[2]:
        loan_amount = st.number_input('Loan Amount (₹)', min_value=0, value=2560000,
                                      help='Total loan amount being requested.')

    loan_to_income_ratio = loan_amount / income if income > 0 else 0
    with row2[0]:
        st.metric(
            label='Loan-to-Income Ratio',
            value=f'{loan_to_income_ratio:.2f}',
            delta='⚠️ High Risk' if loan_to_income_ratio > 3 else '✅ Acceptable',
            delta_color='inverse' if loan_to_income_ratio > 3 else 'off',
            help='Auto-calculated: Loan Amount ÷ Income. Values above 3× indicate elevated risk.'
        )
    with row2[1]:
        loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36,
                                             help='Total loan repayment period in months.')
    with row2[2]:
        avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20,
                                                  help='Average Days Past Due per delinquent month.')

    with row3[0]:
        delinquency_ratio = st.number_input('Delinquency Ratio (%)', min_value=0, max_value=100, step=1, value=30,
                                            help='% of loan months where a payment was missed or late (0–100).')
    with row3[1]:
        credit_utilization_ratio = st.number_input('Credit Utilization (%)', min_value=0, max_value=100, step=1, value=30,
                                                   help='% of total available credit currently in use.')
    with row3[2]:
        num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2,
                                            help='Number of currently active loan accounts (1–4).')

    with row4[0]:
        residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'],
                                      help='Whether the applicant owns, rents, or has a mortgage.')
    with row4[1]:
        loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'],
                                    help='The stated purpose for the loan.')
    with row4[2]:
        loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'],
                                 help='Secured loans are backed by collateral. Unsecured carry higher risk.')

    st.divider()

    if st.button('Calculate Risk', type='primary', use_container_width=True):
        probability, credit_score, rating, shap_values = predict(
            age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type
        )
        st.session_state.prediction_result = (probability, credit_score, rating, shap_values)
        st.session_state.prediction_inputs = {
            'age': age, 'income': income, 'loan_amount': loan_amount,
            'loan_tenure_months': loan_tenure_months,
            'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
            'delinquency_ratio': delinquency_ratio,
            'credit_utilization_ratio': credit_utilization_ratio,
            'num_open_accounts': num_open_accounts,
            'residence_type': residence_type,
            'loan_purpose': loan_purpose,
            'loan_type': loan_type,
        }

    if 'prediction_result' in st.session_state:
        probability, credit_score, rating, shap_values = st.session_state.prediction_result
        inputs = st.session_state.prediction_inputs

        res_hdr, pdf_hdr = st.columns([3, 1])
        with res_hdr:
            st.subheader('Assessment Results')
        with pdf_hdr:
            pdf_bytes = generate_pdf(inputs, probability, credit_score, rating, shap_values)
            st.download_button(
                label='📄 Download PDF Report',
                data=pdf_bytes,
                file_name=f'credit_report_{date.today()}.pdf',
                mime='application/pdf',
                use_container_width=True,
            )
        render_results(probability, credit_score, rating, shap_values)

        st.divider()

        # What-If Simulator
        with st.expander('🔧 What-If Simulator — Explore how changes affect the score'):
            st.caption("Adjust sliders to see how changes in the applicant's profile would impact the credit score.")

            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                w_avg_dpd = st.slider('Avg DPD', 0, 200, inputs['avg_dpd_per_delinquency'],
                                      help='Adjust average days past due per delinquent month.')
                w_delinquency = st.slider('Delinquency Ratio (%)', 0, 100, inputs['delinquency_ratio'],
                                          help='Adjust % of months with late payments.')
            with w_col2:
                w_credit_util = st.slider('Credit Utilization (%)', 0, 100, inputs['credit_utilization_ratio'],
                                          help='Adjust % of credit in use.')
                w_open_accounts = st.slider('Open Loan Accounts', 1, 4, inputs['num_open_accounts'])
            with w_col3:
                w_loan_tenure = st.slider('Loan Tenure (months)', 0, 360, inputs['loan_tenure_months'])
                w_age = st.slider('Age', 18, 100, inputs['age'])

            w_prob, w_score, w_rating, _ = predict(
                w_age, inputs['income'], inputs['loan_amount'], w_loan_tenure, w_avg_dpd,
                w_delinquency, w_credit_util, w_open_accounts,
                inputs['residence_type'], inputs['loan_purpose'], inputs['loan_type']
            )
            delta = w_score - credit_score
            w_cfg = RATING_CONFIG.get(w_rating, {'color': '#333333', 'bg': '#f5f5f5', 'emoji': '⚪'})

            st.divider()
            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                st.metric('Original Score', credit_score,
                          help=f'Rating: {rating}  |  Default Prob: {probability:.1%}')
            with sim_c2:
                st.metric('Simulated Score', w_score, delta=delta,
                          delta_color='normal',
                          help=f'Rating: {w_rating}  |  Default Prob: {w_prob:.1%}')
            with sim_c3:
                st.markdown(f"""
<div style="background-color:{w_cfg['bg']};border-left:4px solid {w_cfg['color']};
    border-radius:6px;padding:0.6rem 1rem;margin-top:0.25rem;">
  <div style="font-size:1.1rem;font-weight:700;color:{w_cfg['color']}">{w_cfg['emoji']} {w_rating}</div>
  <div style="font-size:0.85rem;color:#555">Default Prob: <strong>{w_prob:.1%}</strong></div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Scoring
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader('Batch Scoring')
    st.markdown('Score multiple applicants at once by uploading a CSV file.')

    template_df = pd.DataFrame([{
        'age': 28, 'income': 1200000, 'loan_amount': 2560000,
        'loan_tenure_months': 36, 'avg_dpd_per_delinquency': 20,
        'delinquency_ratio': 30, 'credit_utilization_ratio': 30,
        'num_open_accounts': 2, 'residence_type': 'Owned',
        'loan_purpose': 'Personal', 'loan_type': 'Unsecured',
    }])
    st.download_button(
        label='⬇️ Download CSV Template',
        data=template_df.to_csv(index=False),
        file_name='batch_template.csv',
        mime='text/csv',
    )
    st.caption('Required columns: ' + ', '.join(_BATCH_COLUMNS))

    uploaded_file = st.file_uploader('Upload applicant CSV', type=['csv'])

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        missing = [c for c in _BATCH_COLUMNS if c not in input_df.columns]
        if missing:
            st.error(f'Missing required columns: {", ".join(missing)}')
        else:
            st.info(f'Found **{len(input_df)} applicants**. Click below to score.')
            if st.button('Score All Applicants', type='primary', use_container_width=True):
                with st.spinner('Scoring applicants...'):
                    results_df = predict_batch(input_df[_BATCH_COLUMNS])
                output_df = pd.concat([input_df.reset_index(drop=True), results_df], axis=1)

                def highlight_rating(val):
                    palette = {
                        'Poor':      'background-color: #c62828; color: #ffffff; font-weight: bold',
                        'Average':   'background-color: #e65100; color: #ffffff; font-weight: bold',
                        'Good':      'background-color: #558b2f; color: #ffffff; font-weight: bold',
                        'Excellent': 'background-color: #1565c0; color: #ffffff; font-weight: bold',
                    }
                    return palette.get(val, '')

                st.dataframe(
                    output_df.style.map(highlight_rating, subset=['rating']),
                    use_container_width=True
                )

                st.download_button(
                    label='⬇️ Download Results CSV',
                    data=output_df.to_csv(index=False),
                    file_name=f'batch_results_{date.today()}.csv',
                    mime='text/csv',
                    use_container_width=True,
                )

                # Summary metrics
                st.subheader('Batch Summary')
                valid = results_df[results_df['rating'].isin(['Poor', 'Average', 'Good', 'Excellent'])]
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric('Total Applicants', len(input_df))
                with s2:
                    avg_sc = valid['credit_score'].mean()
                    st.metric('Avg Credit Score', f'{avg_sc:.0f}' if not pd.isna(avg_sc) else 'N/A')
                with s3:
                    st.metric('High Risk (Poor)', int((valid['rating'] == 'Poor').sum()))
                with s4:
                    st.metric('Low Risk (Good+)', int(valid['rating'].isin(['Good', 'Excellent']).sum()))

                if len(valid) > 0:
                    rating_counts = valid['rating'].value_counts().reindex(
                        ['Excellent', 'Good', 'Average', 'Poor'], fill_value=0
                    )
                    fig3 = go.Figure(go.Bar(
                        x=rating_counts.index,
                        y=rating_counts.values,
                        marker_color=['#1565c0', '#558b2f', '#e65100', '#c62828'],
                        text=rating_counts.values,
                        textposition='outside',
                    ))
                    fig3.update_layout(
                        title='Rating Distribution',
                        height=300,
                        margin=dict(t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(showgrid=True, gridcolor='#eee'),
                        xaxis=dict(title=None),
                    )
                    st.plotly_chart(fig3, use_container_width=True)
