"""
predictor.py — Inference functions for single-applicant and batch scoring.

Public API:
    predict()                — Score one applicant; return probability, score,
                               rating, and per-feature SHAP values.
    predict_batch()          — Score a DataFrame of applicants row-by-row.
    calculate_credit_score() — Convert a preprocessed feature row to a credit
                               score and categorical rating.

Credit score formula:
    Credit Score = BASE_SCORE + (1 − P_default) × SCALE_LENGTH
    where BASE_SCORE = 300 and SCALE_LENGTH = 600 (range: 300–900).
"""

import numpy as np
import pandas as pd

from config import BASE_SCORE, SCALE_LENGTH
from model.loader import model, features, SHAP_AVAILABLE, shap_explainer
from model.preprocessor import prepare_input


def calculate_credit_score(input_df: pd.DataFrame) -> tuple:
    """Convert a preprocessed feature row into a default probability, credit score, and rating.

    The logistic regression's decision boundary is computed manually from the
    model's coefficient and intercept so we can reuse the intermediate log-odds
    value.  The result is then mapped to the 300–900 credit score scale.

    Args:
        input_df: A single-row DataFrame produced by prepare_input().

    Returns:
        A tuple of:
            - default_probability (float): Estimated probability of loan default.
            - credit_score (int):          Score mapped to the 300–900 range.
            - rating (str):                Categorical risk rating.
    """
    # Compute the log-odds (linear score) manually.
    # np.dot handles the weighted sum; model.intercept_ adds the bias term.
    log_odds = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Apply the sigmoid (logistic) function to convert log-odds → probability.
    default_probability = float(1.0 / (1.0 + np.exp(-log_odds)).flatten()[0])

    # Map the non-default probability to the credit score range.
    # Higher non-default probability → higher (better) credit score.
    non_default_probability = 1.0 - default_probability
    credit_score = int(BASE_SCORE + non_default_probability * SCALE_LENGTH)

    # Assign a human-readable rating based on score thresholds.
    rating = _score_to_rating(credit_score)

    return default_probability, credit_score, rating


def _score_to_rating(score: int) -> str:
    """Map a numeric credit score to a categorical rating label.

    Thresholds (intentionally mirrors common bureau segmentation):
        750–900  → Excellent
        650–749  → Good
        500–649  → Average
        300–499  → Poor

    Args:
        score: Integer credit score (valid range: 300–900).

    Returns:
        One of 'Poor', 'Average', 'Good', 'Excellent'.
    """
    if score >= 750:
        return "Excellent"
    elif score >= 650:
        return "Good"
    elif score >= 500:
        return "Average"
    else:
        return "Poor"


def predict(
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
) -> tuple:
    """Score a single applicant and return a full prediction result with explanations.

    Pipeline:
        1. prepare_input()          → scales and encodes raw inputs.
        2. calculate_credit_score() → probability, score, rating from the model.
        3. SHAP LinearExplainer     → per-feature contribution values.

    Args:
        age: Applicant age in years.
        income: Annual gross income in ₹.
        loan_amount: Requested loan amount in ₹.
        loan_tenure_months: Repayment period in months.
        avg_dpd_per_delinquency: Average days past due per delinquent month.
        delinquency_ratio: Percentage of loan months with a missed payment.
        credit_utilization_ratio: Percentage of available credit in use.
        num_open_accounts: Number of currently active loan accounts.
        residence_type: One of 'Owned', 'Rented', 'Mortgage'.
        loan_purpose: One of 'Education', 'Home', 'Auto', 'Personal'.
        loan_type: One of 'Secured', 'Unsecured'.

    Returns:
        A tuple of:
            - default_probability (float):  Estimated probability of loan default.
            - credit_score (int):           Score in the 300–900 range.
            - rating (str):                 Categorical risk rating.
            - shap_values (pd.Series):      Per-feature SHAP contributions, indexed
                                            by feature name.
    """
    # Preprocess raw inputs into a scaled, encoded DataFrame.
    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type,
    )

    # Run the model to get probability, credit score, and rating.
    default_probability, credit_score, rating = calculate_credit_score(input_df)

    # Compute feature-level SHAP values for the explainability panel.
    if SHAP_AVAILABLE and shap_explainer is not None:
        # shap_explainer.shap_values() returns shape (n_rows, n_features).
        # We take [0] to get the single-row array.
        raw_shap = shap_explainer.shap_values(input_df)[0]
    else:
        # Fallback: approximate contribution as coefficient × scaled feature value.
        # This is directionally correct (positive = risk-increasing) but less
        # accurate than true SHAP values.  Sufficient when shap is unavailable.
        raw_shap = input_df.values[0] * model.coef_[0]

    shap_series = pd.Series(raw_shap, index=features)

    return default_probability, credit_score, rating, shap_series


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score a batch of applicants from a DataFrame.

    Calls predict() for each row individually.  Errors on individual rows are
    caught and surfaced in the 'rating' column as an error message so that one
    malformed row does not abort the entire batch.

    Args:
        df: DataFrame with exactly the columns listed in config.BATCH_COLUMNS.
            Numeric columns may arrive as strings (from CSV upload) — they are
            coerced with int()/float() inside this function.

    Returns:
        A DataFrame with three columns per input row:
            'credit_score'        – Integer score (300–900), or None on error.
            'default_probability' – Probability × 100, rounded to 2 dp, or None.
            'rating'              – Rating string, or an error message.
    """
    results = []

    for _, row in df.iterrows():
        try:
            prob, score, rating, _ = predict(
                age=int(row["age"]),
                income=float(row["income"]),
                loan_amount=float(row["loan_amount"]),
                loan_tenure_months=int(row["loan_tenure_months"]),
                avg_dpd_per_delinquency=float(row["avg_dpd_per_delinquency"]),
                delinquency_ratio=float(row["delinquency_ratio"]),
                credit_utilization_ratio=float(row["credit_utilization_ratio"]),
                num_open_accounts=int(row["num_open_accounts"]),
                residence_type=str(row["residence_type"]),
                loan_purpose=str(row["loan_purpose"]),
                loan_type=str(row["loan_type"]),
            )
            results.append({
                "credit_score":        score,
                "default_probability": round(prob * 100, 2),
                "rating":              rating,
            })

        except Exception as exc:
            # Surface per-row errors without halting the entire batch run.
            results.append({
                "credit_score":        None,
                "default_probability": None,
                "rating":              f"Error: {exc}",
            })

    return pd.DataFrame(results)
