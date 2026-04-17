"""
config.py — Application-wide constants and configuration.

All tunable values (rating thresholds, colour palettes, feature labels, model paths,
and credit-score mapping parameters) live here so that changes only need to be made
in one place rather than scattered across multiple modules.

Import individual constants:
    from config import RATING_CONFIG, MODEL_PATH, BATCH_COLUMNS
"""

from pathlib import Path

# ── Model Artifact ────────────────────────────────────────────────────────────
# Absolute path to the serialised model bundle saved by the training notebook.
# The bundle contains: model (LogisticRegression), scaler (MinMaxScaler),
# features (final 13-column Index), and cols_to_scale (18-column list).
MODEL_PATH: Path = Path(__file__).parent / "artifacts" / "model_data.joblib"

# ── Credit Score Mapping ──────────────────────────────────────────────────────
# The logistic regression's default probability is converted to a human-readable
# score on the 300–900 scale (mirrors the CIBIL bureau scale used in India):
#
#     Credit Score = BASE_SCORE + (1 − P_default) × SCALE_LENGTH
#
# A "certain default" (P = 1) maps to 300; a "certain non-default" (P = 0)
# maps to 900.  Real applicants fall somewhere in between.
BASE_SCORE: int = 300
SCALE_LENGTH: int = 600

# ── Rating Display Configuration ─────────────────────────────────────────────
# Maps each categorical rating to the hex colours and emoji used across the UI
# (gauges, rating cards, batch table highlights) and in the PDF report.
RATING_CONFIG: dict = {
    "Poor":      {"color": "#c62828", "bg": "#ffebee", "emoji": "🔴"},
    "Average":   {"color": "#e65100", "bg": "#fff3e0", "emoji": "🟠"},
    "Good":      {"color": "#558b2f", "bg": "#f1f8e9", "emoji": "🟢"},
    "Excellent": {"color": "#1565c0", "bg": "#e3f2fd", "emoji": "🌟"},
}

# RGB equivalents of the rating colours used by fpdf2, which requires integer
# (R, G, B) tuples instead of hex strings.
RATING_COLORS_RGB: dict = {
    "Poor":      (198,  40,  40),
    "Average":   (230,  81,   0),
    "Good":      ( 85, 139,  47),
    "Excellent": ( 21, 101, 192),
}

# ── Feature Labels ────────────────────────────────────────────────────────────
# Maps internal model feature column names to human-readable display strings
# used in the SHAP bar chart tooltips and in the PDF risk-factor section.
FEATURE_LABELS: dict = {
    "age":                      "Age",
    "loan_tenure_months":       "Loan Tenure",
    "number_of_open_accounts":  "Open Accounts",
    "credit_utilization_ratio": "Credit Utilization",
    "loan_to_income":           "Loan-to-Income Ratio",
    "delinquency_ratio":        "Delinquency Ratio",
    "avg_dpd_per_delinquency":  "Avg DPD",
    "residence_type_Owned":     "Residence: Owned",
    "residence_type_Rented":    "Residence: Rented",
    "loan_purpose_Education":   "Purpose: Education",
    "loan_purpose_Home":        "Purpose: Home",
    "loan_purpose_Personal":    "Purpose: Personal",
    "loan_type_Unsecured":      "Loan Type: Unsecured",
}

# ── Batch Scoring Columns ─────────────────────────────────────────────────────
# Exact column names expected in a CSV uploaded for batch scoring.
# The order matches the keyword arguments of the predict() function.
BATCH_COLUMNS: list = [
    "age",
    "income",
    "loan_amount",
    "loan_tenure_months",
    "avg_dpd_per_delinquency",
    "delinquency_ratio",
    "credit_utilization_ratio",
    "num_open_accounts",
    "residence_type",
    "loan_purpose",
    "loan_type",
]
