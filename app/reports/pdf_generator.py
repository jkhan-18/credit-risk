"""
pdf_generator.py — One-page PDF credit assessment report generator.

Exports:
    generate_pdf()  — Builds a formatted PDF from a prediction result and
                      returns it as bytes for a Streamlit download button.

The report layout (top → bottom):
    1. Blue header banner   — company name + report title + date.
    2. Assessment result    — coloured score box, rating label, default probability.
    3. Applicant details    — two-column table of all input values.
    4. Top SHAP factors     — top-5 risk drivers with direction indicators.
    5. Footer               — model performance metadata + confidentiality notice.

Layout is managed manually (no auto page break) to guarantee the content fits
on exactly one A4 page.
"""

from datetime import date

import pandas as pd
from fpdf import FPDF

from config import FEATURE_LABELS, RATING_COLORS_RGB


def generate_pdf(
    inputs: dict,
    probability: float,
    credit_score: int,
    rating: str,
    shap_values: pd.Series,
) -> bytes:
    """Generate a one-page PDF credit risk assessment report.

    Args:
        inputs:       Raw form inputs dict with keys: age, income, loan_amount,
                      loan_tenure_months, avg_dpd_per_delinquency, delinquency_ratio,
                      credit_utilization_ratio, num_open_accounts, residence_type,
                      loan_purpose, loan_type.
        probability:  Model's estimated default probability (0.0–1.0).
        credit_score: Credit score in the 300–900 range.
        rating:       Categorical rating ('Poor', 'Average', 'Good', 'Excellent').
        shap_values:  Per-feature SHAP values as a pandas Series indexed by
                      feature name (matching config.FEATURE_LABELS keys).

    Returns:
        PDF content as bytes, suitable for passing to st.download_button(data=...).
    """
    # Resolve the RGB colour tuple for this rating (used throughout the PDF).
    rc = RATING_COLORS_RGB.get(rating, (51, 51, 51))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)  # We control layout manually on one page.
    pdf.add_page()

    # ── Section 1: Header Banner ───────────────────────────────────────────────
    # Deep blue filled rectangle spanning the full page width.
    pdf.set_fill_color(21, 101, 192)
    pdf.rect(0, 0, 210, 24, "F")

    pdf.set_xy(10, 7)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "QUANTUM FINANCE  -  CREDIT RISK ASSESSMENT REPORT")

    # Generation date beneath the banner.
    pdf.set_xy(10, 28)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Generated: {date.today().strftime('%B %d, %Y')}")

    # ── Section 2: Assessment Result ──────────────────────────────────────────
    pdf.set_xy(10, 40)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "ASSESSMENT RESULT")

    # Coloured rectangle containing the numeric credit score.
    pdf.set_xy(10, 50)
    pdf.set_fill_color(*rc)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 26)
    pdf.cell(55, 18, str(credit_score), align="C", fill=True)

    # Rating label and default probability to the right of the score box.
    pdf.set_xy(70, 50)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*rc)
    pdf.cell(0, 8, rating)

    pdf.set_xy(70, 60)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Default Probability: {probability:.1%}")

    # Horizontal rule dividing the result block from the details section.
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, 74, 200, 74)

    # ── Section 3: Applicant Details ──────────────────────────────────────────
    pdf.set_xy(10, 78)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "APPLICANT DETAILS")

    # Guard against division by zero when formatting the LTI ratio.
    lti = inputs["loan_amount"] / inputs["income"] if inputs["income"] else None

    # List of (label, formatted value) pairs for the two-column details grid.
    details = [
        ("Age",                str(inputs["age"])),
        ("Income",             f"Rs {inputs['income']:,}"),
        ("Loan Amount",        f"Rs {inputs['loan_amount']:,}"),
        ("Loan Tenure",        f"{inputs['loan_tenure_months']} months"),
        ("Loan-to-Income",     f"{lti:.2f}" if lti is not None else "N/A"),
        ("Loan Purpose",       inputs["loan_purpose"]),
        ("Loan Type",          inputs["loan_type"]),
        ("Residence Type",     inputs["residence_type"]),
        ("Open Accounts",      str(inputs["num_open_accounts"])),
        ("Credit Utilization", f"{inputs['credit_utilization_ratio']}%"),
        ("Delinquency Ratio",  f"{inputs['delinquency_ratio']}%"),
        ("Avg DPD",            str(inputs["avg_dpd_per_delinquency"])),
    ]

    y_start = 89
    # Two-column grid: left column at x=10, right column at x=105.
    for i, (label, value) in enumerate(details):
        col = i % 2          # 0 = left column, 1 = right column
        row = i // 2
        pdf.set_xy(10 + col * 95, y_start + row * 8)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(42, 6, label + ":")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(50, 6, value)

    # Divider below the details grid.
    # The grid height depends on the number of rows (ceil(len(details) / 2)).
    y_div = y_start + (len(details) // 2 + 1) * 8 + 4
    pdf.line(10, y_div, 200, y_div)

    # ── Section 4: Top SHAP Risk Factors ──────────────────────────────────────
    pdf.set_xy(10, y_div + 4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "TOP RISK FACTORS  (SHAP - SHapley Additive exPlanations)")

    # Display the 5 most influential features sorted by absolute SHAP value.
    top_factors = shap_values.abs().nlargest(5)
    for i, feat in enumerate(top_factors.index):
        val = shap_values[feat]
        label     = FEATURE_LABELS.get(feat, feat)
        direction = "Increases default risk" if val > 0 else "Reduces default risk"
        indicator = "[+]" if val > 0 else "[-]"
        # Red for risk-increasing, green for risk-reducing factors.
        color = (198, 40, 40) if val > 0 else (46, 125, 50)

        pdf.set_xy(10, y_div + 14 + i * 8)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*color)
        pdf.cell(0, 6, f"  {i + 1}. {label}  {indicator}  {direction}")

    # ── Section 5: Footer ─────────────────────────────────────────────────────
    # Anchored to y=286 (1 mm from the bottom of A4) so it never overlaps content.
    pdf.set_xy(10, 287)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, 286, 200, 286)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(
        0, 6,
        "Quantum Finance Credit Risk Model  |  AUC: 0.98  |  Gini: 0.96  |  For internal use only",
    )

    # pdf.output() returns a bytearray; cast to bytes for Streamlit compatibility.
    return bytes(pdf.output())
