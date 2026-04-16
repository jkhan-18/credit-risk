# Lauki Finance: Credit Risk Modelling

A Streamlit web app that predicts the credit default probability of a loan applicant and outputs a credit score (300–900) with a rating.

## Features

The model uses the following inputs:

| Input | Description |
|---|---|
| Age | Applicant's age |
| Income | Annual income |
| Loan Amount | Requested loan amount |
| Loan Tenure | Loan duration in months |
| Avg DPD | Average days past due per delinquency |
| Delinquency Ratio | % of months with a missed payment |
| Credit Utilization Ratio | % of credit currently in use |
| Open Loan Accounts | Number of active loan accounts |
| Residence Type | Owned / Rented / Mortgage |
| Loan Purpose | Education / Home / Auto / Personal |
| Loan Type | Secured / Unsecured |

## Output

- **Default Probability** — likelihood the applicant will default
- **Credit Score** — scaled 300–900 (higher is better)
- **Rating** — Poor / Average / Good / Excellent

## Run Locally

```bash
# Clone the repo
git clone <your-repo-url>
cd Credit_Card_Risk

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

## Project Structure

```
Credit_Card_Risk/
├── app/
│   ├── main.py                  # Streamlit UI
│   ├── prediction_helper.py     # Model loading & inference
│   └── artifacts/
│       └── model_data.joblib    # Trained model + scaler
├── requirements.txt
├── .gitignore
└── credit_risk_model_codebasics.ipynb  # Training notebook
```

## Model

- **Algorithm**: Logistic Regression (tuned with Optuna)
- **Class imbalance**: handled with SMOTETomek
- **AUC**: 0.98 | **Gini Coefficient**: 0.96
- **Feature selection**: VIF (multicollinearity) + Information Value (IV > 0.02)
