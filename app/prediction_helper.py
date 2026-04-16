import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Path to the saved model and its components
MODEL_PATH = Path(__file__).parent / 'artifacts' / 'model_data.joblib'

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']

try:
    import shap as _shap_lib
    _shap_background = np.zeros((1, len(features)))
    _shap_explainer = _shap_lib.LinearExplainer(model, _shap_background, feature_perturbation='interventional')
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

_BATCH_COLUMNS = [
    'age', 'income', 'loan_amount', 'loan_tenure_months', 'avg_dpd_per_delinquency',
    'delinquency_ratio', 'credit_utilization_ratio', 'num_open_accounts',
    'residence_type', 'loan_purpose', 'loan_type'
]


def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                  loan_purpose, loan_type):
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        # additional dummy fields just for scaling purpose
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }
    df = pd.DataFrame([input_data])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df[features]
    return df


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                             delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                             residence_type, loan_purpose, loan_type)
    probability, credit_score, rating = calculate_credit_score(input_df)

    if _SHAP_AVAILABLE:
        shap_values = pd.Series(_shap_explainer.shap_values(input_df)[0], index=features)
    else:
        shap_values = pd.Series(input_df.values[0] * model.coef_[0], index=features)

    return probability, credit_score, rating, shap_values


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_
    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score[0])
    return default_probability.flatten()[0], int(credit_score[0]), rating


def predict_batch(df):
    results = []
    for _, row in df.iterrows():
        try:
            prob, score, rating, _ = predict(
                age=int(row['age']),
                income=float(row['income']),
                loan_amount=float(row['loan_amount']),
                loan_tenure_months=int(row['loan_tenure_months']),
                avg_dpd_per_delinquency=float(row['avg_dpd_per_delinquency']),
                delinquency_ratio=float(row['delinquency_ratio']),
                credit_utilization_ratio=float(row['credit_utilization_ratio']),
                num_open_accounts=int(row['num_open_accounts']),
                residence_type=str(row['residence_type']),
                loan_purpose=str(row['loan_purpose']),
                loan_type=str(row['loan_type'])
            )
            results.append({
                'credit_score': score,
                'default_probability': round(prob * 100, 2),
                'rating': rating
            })
        except Exception as e:
            results.append({'credit_score': None, 'default_probability': None, 'rating': f'Error: {e}'})
    return pd.DataFrame(results)
