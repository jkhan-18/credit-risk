"""
loader.py — Load the trained model bundle from disk exactly once at import time.

Exposes module-level globals (model, scaler, features, cols_to_scale) so that
every import of this module shares the same in-memory objects.  Loading once at
startup avoids repeated disk I/O on every prediction request.

Also initialises the SHAP LinearExplainer if the optional `shap` package is
installed.  If `shap` is absent, SHAP_AVAILABLE is set to False and the
predictor falls back to a coefficient-based approximation.
"""

import numpy as np
import joblib

from config import MODEL_PATH

# ── Load Model Bundle ─────────────────────────────────────────────────────────
# The .joblib file was saved by the training notebook as a dict with four keys:
#   'model'         – fitted LogisticRegression instance
#   'scaler'        – fitted MinMaxScaler instance
#   'features'      – pandas Index of the 13 final model columns
#   'cols_to_scale' – list of the 18 numeric columns the scaler was fitted on
_bundle = joblib.load(MODEL_PATH)

model: object         = _bundle["model"]           # Logistic Regression classifier
scaler: object        = _bundle["scaler"]          # MinMaxScaler fitted on training data
features: object      = _bundle["features"]        # Final 13-column feature Index
cols_to_scale: list   = _bundle["cols_to_scale"]   # 18 numeric columns for the scaler

# ── SHAP Explainer Initialisation ─────────────────────────────────────────────
# LinearExplainer is the correct choice for logistic regression — it is both
# fast (O(features)) and exact (no approximation needed for linear models).
#
# Background dataset: a single all-zeros row is used so that a SHAP value of
# +X means "this feature raised the log-odds of default by X compared to a
# completely neutral applicant baseline".  Using zeros keeps the explanation
# intuitive for risk officers rather than using a training-set mean background.
#
# 'interventional' perturbation removes feature correlations from the
# explanation, which is generally preferable for regulatory-facing outputs.
try:
    import shap as _shap_lib

    _background = np.zeros((1, len(features)))
    shap_explainer: object = _shap_lib.LinearExplainer(
        model, _background, feature_perturbation="interventional"
    )
    SHAP_AVAILABLE: bool = True

except Exception:
    # shap is not installed or failed to initialise — predictor.py will use
    # the coefficient × feature-value fallback instead.
    shap_explainer = None
    SHAP_AVAILABLE = False
