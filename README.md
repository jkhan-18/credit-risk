# Quantum Finance: Credit Risk Modelling

> An end-to-end ML-powered credit risk assessment platform built with Streamlit. Predicts the probability of loan default, generates a credit score (300–900), and provides SHAP-based explanations — all served through an interactive web interface.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-deihgke4ejwjb9mlsa34p6.streamlit.app/)

---

## Features

| Feature | Description |
|---|---|
| **Single Applicant Scoring** | Fill in 11 inputs and get an instant credit score, default probability, and rating |
| **SHAP Explainability** | Top risk factors shown as a SHAP waterfall bar chart — red = increases risk, green = reduces risk |
| **What-If Simulator** | Adjust key inputs via sliders and compare the original vs. simulated score side-by-side |
| **Batch Scoring** | Upload a CSV of applicants, score all at once, download colour-coded results with a summary dashboard |
| **PDF Report Export** | One-click download of a formatted credit assessment report including score, rating, applicant details, and top risk factors |
| **Sidebar Guide** | Inline documentation explaining every input field and how to interpret results |

---

## Application Architecture

```
Credit_Card_Risk/
├── app/
│   ├── main.py                        # Entry point — page config, sidebar, tabs (thin orchestrator)
│   ├── config.py                      # All constants: colours, labels, paths, score thresholds
│   ├── model/
│   │   ├── loader.py                  # Loads model bundle from disk once; initialises SHAP explainer
│   │   ├── preprocessor.py            # prepare_input(): feature engineering, encoding, scaling
│   │   └── predictor.py               # predict(), predict_batch(), calculate_credit_score()
│   ├── ui/
│   │   ├── components.py              # Shared widgets: gauge chart, rating badge, SHAP bar chart
│   │   ├── sidebar.py                 # Sidebar: usage guide, field reference, model metadata
│   │   ├── single_applicant.py        # Tab 1: input form, results, What-If simulator, PDF download
│   │   └── batch_scoring.py           # Tab 2: CSV upload, scoring, results table, summary dashboard
│   ├── reports/
│   │   └── pdf_generator.py           # generate_pdf(): one-page PDF credit assessment report
│   └── artifacts/
│       └── model_data.joblib          # Serialised model bundle (model + scaler + features)
├── artifacts/                         # Root copy of model bundle (for notebook compatibility)
│   └── model_data.joblib
├── dataset/                           # Raw training data (excluded from repo via .gitignore)
│   ├── customers.csv
│   ├── loans.csv
│   └── bureau_data.csv
├── credit_risk_model_codebasics.ipynb # Full training & experimentation notebook
├── requirements.txt                   # Pinned Python dependencies
├── runtime.txt                        # Python 3.11 pin for Streamlit Community Cloud
└── .gitignore
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Thin Streamlit entry point — wires page config, sidebar, and tabs together |
| `config.py` | Single source of truth for all constants (colours, labels, thresholds, paths) |
| `model/loader.py` | Loads the joblib bundle once at import time; exposes model, scaler, SHAP explainer |
| `model/preprocessor.py` | Replicates the training pipeline: feature engineering → encoding → scaling → selection |
| `model/predictor.py` | Single-applicant prediction, batch scoring, credit score / rating computation |
| `ui/components.py` | Reusable Plotly gauge and SHAP bar chart rendered by any tab |
| `ui/sidebar.py` | User guide, field reference, result interpretation, SHAP explainer, model stats |
| `ui/single_applicant.py` | Tab 1 form, session-state result persistence, What-If simulator |
| `ui/batch_scoring.py` | Tab 2 CSV workflow: upload → validate → score → export → summary chart |
| `reports/pdf_generator.py` | One-page PDF report with score, rating, applicant details, and SHAP factors |

---

## Model Details

### Training Pipeline

| Step | Detail |
|---|---|
| **Data** | 3 merged sources: customer demographics, loan records, bureau data |
| **EDA** | KDE plots, boxplots, correlation heatmap per default class |
| **Feature Engineering** | Loan-to-Income ratio, Delinquency Ratio, Avg DPD per delinquency |
| **Outlier Removal** | Processing fee / loan amount ratio capped at 3% |
| **Feature Selection** | VIF (variance inflation factor) to remove multicollinear features; Information Value (IV > 0.02) for categorical selection |
| **Scaling** | MinMaxScaler on all numeric features |
| **Encoding** | One-hot encoding with `drop_first=True` |
| **Class Imbalance** | SMOTETomek (oversampling minority + cleaning boundary noise) |
| **Hyperparameter Tuning** | Optuna with 50 trials, 3-fold CV, macro F1 objective |
| **Algorithm** | Logistic Regression (best interpretability/performance tradeoff vs. XGBoost) |

### Model Performance

| Metric | Value |
|---|---|
| AUC (ROC) | **0.98** |
| Gini Coefficient | **0.96** |
| KS Statistic | **85.98%** (at decile 8) |
| Rank Ordering | ✅ Confirmed (top 2 deciles capture ~98.6% of default events) |

### Credit Score Mapping

$$\text{Credit Score} = 300 + (1 - P_{\text{default}}) \times 600$$

| Score Range | Rating | Risk Level |
|---|---|---|
| 750 – 900 | 🌟 Excellent | Very Low |
| 650 – 749 | 🟢 Good | Low |
| 500 – 649 | 🟠 Average | Medium |
| 300 – 499 | 🔴 Poor | High |

---

## Input Features

| Input | Type | Description |
|---|---|---|
| Age | Numeric | Applicant's age (18–100) |
| Income | Numeric | Annual gross income (₹) |
| Loan Amount | Numeric | Requested loan amount (₹) |
| Loan Tenure | Numeric | Repayment period in months |
| Avg DPD | Numeric | Average days past due per delinquent month |
| Delinquency Ratio | Numeric (%) | % of loan months with a late payment |
| Credit Utilization | Numeric (%) | % of available credit currently in use |
| Open Loan Accounts | Integer (1–4) | Number of active loan accounts |
| Residence Type | Categorical | Owned / Rented / Mortgage |
| Loan Purpose | Categorical | Education / Home / Auto / Personal |
| Loan Type | Categorical | Secured / Unsecured |

---

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/jkhan-18/credit-risk.git
cd credit-risk

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/main.py
```

App opens at `http://localhost:8501`

---

## Batch Scoring CSV Format

Download the template from the app's **Batch Scoring** tab, or use this column structure:

| Column | Example |
|---|---|
| age | 28 |
| income | 1200000 |
| loan_amount | 2560000 |
| loan_tenure_months | 36 |
| avg_dpd_per_delinquency | 20 |
| delinquency_ratio | 30 |
| credit_utilization_ratio | 30 |
| num_open_accounts | 2 |
| residence_type | Owned |
| loan_purpose | Personal |
| loan_type | Unsecured |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.47 |
| Visualisation | Plotly |
| ML Model | scikit-learn 1.3 (LogisticRegression) |
| Explainability | SHAP (LinearExplainer) |
| PDF Generation | fpdf2 |
| Data Processing | pandas, numpy |
| Model Serialisation | joblib |
| Deployment | Streamlit Community Cloud |

---

## Deployment

This app is deployed on **Streamlit Community Cloud**:

1. GitHub repo is connected at [share.streamlit.io](https://share.streamlit.io)
2. Entry point: `app/main.py`
3. Python version pinned to 3.11 via `runtime.txt` (required for `scikit-learn==1.3.0` wheel compatibility)

---

*Built as part of the Codebasics ML Course — extended with production-grade features.*
