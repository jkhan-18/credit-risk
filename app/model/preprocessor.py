"""
preprocessor.py — Transform raw applicant inputs into a model-ready DataFrame.

The prepare_input() function mirrors the exact feature-engineering and
preprocessing pipeline applied to the training data in the notebook:

  Step 1 – Compute the loan_to_income engineered feature.
  Step 2 – Build a dict with 13 model features + 11 placeholder columns.
  Step 3 – Apply the fitted MinMaxScaler across all 18 numeric columns.
  Step 4 – Subset to the 13 final model features and return a one-row DataFrame.

Why placeholder columns?
  The MinMaxScaler was fitted on 18 numeric columns during training.  After
  scaling, VIF and IV-based feature selection reduced the model inputs to 13
  columns.  At inference time we must still feed the scaler all 18 columns so
  that the transform is consistent with training.  The 11 dropped columns are
  filled with 1 (a neutral value within the [0, 1] scaled range) and their
  scaled output is then discarded.
"""

import pandas as pd

from model.loader import scaler, features, cols_to_scale


def prepare_input(
    age: int,
    income: float,
    loan_amount: float,
    loan_tenure_months: int,
    avg_dpd_per_delinquency: float,
    delinquency_ratio: float,
    credit_utilization_ratio: float,
    num_open_accounts: int,
    residence_type: str,
    loan_purpose: str,
    loan_type: str,
) -> pd.DataFrame:
    """Transform raw applicant values into a scaled, encoded DataFrame for inference.

    Args:
        age: Applicant age in years (18–100).
        income: Annual gross income in ₹.
        loan_amount: Requested loan amount in ₹.
        loan_tenure_months: Repayment period in months.
        avg_dpd_per_delinquency: Average days past due per delinquent month.
        delinquency_ratio: Percentage of loan months with a missed payment (0–100).
        credit_utilization_ratio: Percentage of available credit in use (0–100).
        num_open_accounts: Number of currently active loan accounts.
        residence_type: One of 'Owned', 'Rented', 'Mortgage'.
        loan_purpose: One of 'Education', 'Home', 'Auto', 'Personal'.
        loan_type: One of 'Secured', 'Unsecured'.

    Returns:
        A single-row DataFrame with 13 columns in the same order and scale as
        the training data, ready to be passed to the model.
    """

    # ── Step 1: Engineered Features ───────────────────────────────────────────
    # loan_to_income is derived during training as loan_amount / income.
    # Guard against division by zero in case income is 0 (should not happen in
    # normal usage, but defensive programming avoids a runtime ZeroDivisionError).
    loan_to_income = loan_amount / income if income > 0 else 0.0

    # ── Step 2: Assemble Full Input Row ──────────────────────────────────────
    input_data = {
        # 7 numeric features that survive into the final model
        "age":                      age,
        "loan_tenure_months":       loan_tenure_months,
        "number_of_open_accounts":  num_open_accounts,
        "credit_utilization_ratio": credit_utilization_ratio,
        "loan_to_income":           loan_to_income,
        "delinquency_ratio":        delinquency_ratio,
        "avg_dpd_per_delinquency":  avg_dpd_per_delinquency,

        # 6 one-hot encoded categorical columns (drop_first=True was used in training).
        # For each categorical field, exactly one indicator is set to 1 if the
        # value matches; all others default to 0.  The dropped reference category
        # is represented by all indicators being 0 (e.g., 'Mortgage' residence).
        "residence_type_Owned":    1 if residence_type == "Owned"    else 0,
        "residence_type_Rented":   1 if residence_type == "Rented"   else 0,
        "loan_purpose_Education":  1 if loan_purpose  == "Education" else 0,
        "loan_purpose_Home":       1 if loan_purpose  == "Home"      else 0,
        "loan_purpose_Personal":   1 if loan_purpose  == "Personal"  else 0,
        "loan_type_Unsecured":     1 if loan_type     == "Unsecured" else 0,

        # 11 placeholder columns required by the fitted MinMaxScaler.
        # These were present when the scaler was fitted in training but were
        # subsequently removed (VIF / IV selection).  We supply a neutral value
        # of 1 so the scaler receives its expected 18-column input; their scaled
        # values are discarded in Step 4.
        "number_of_dependants":       1,
        "years_at_current_address":   1,
        "zipcode":                    1,
        "sanction_amount":            1,
        "processing_fee":             1,
        "gst":                        1,
        "net_disbursement":           1,
        "principal_outstanding":      1,
        "bank_balance_at_application": 1,
        "number_of_closed_accounts":  1,
        "enquiry_count":              1,
    }

    df = pd.DataFrame([input_data])

    # ── Step 3: Scale Numeric Columns ────────────────────────────────────────
    # Apply the MinMaxScaler that was fitted on the training set.
    # cols_to_scale contains all 18 numeric columns the scaler expects.
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # ── Step 4: Select Final Model Features ──────────────────────────────────
    # Retain only the 13 columns the LogisticRegression was trained on.
    # This step implicitly discards the 11 placeholder columns from Step 2.
    df = df[features]

    return df
