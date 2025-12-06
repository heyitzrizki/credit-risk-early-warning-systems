import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import os

# ----------------------------------------------------------
# PATH HELPER
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def p(path): return os.path.join(BASE_DIR, path)

# ----------------------------------------------------------
# LOAD PIPELINE MODELS (FINAL)
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        pd_model = joblib.load(p("PD_LGBM_pipeline.pkl"))
        lgd_model = joblib.load(p("LGD_LGBM_pipeline.pkl"))
        shap_explainer = joblib.load(p("pd_shap_explainer.pkl"))
        return pd_model, lgd_model, shap_explainer
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

pd_model, lgd_model, shap_explainer = load_models()

# ----------------------------------------------------------
# HUMAN-FRIENDLY NAICS LABELS
# ----------------------------------------------------------
NAICS_LABELS = {
    "11": "Agriculture, Forestry, Fishing & Hunting",
    "21": "Mining, Quarrying, Oil & Gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation",
    "49": "Warehousing",
    "51": "Information Services",
    "52": "Finance & Insurance",
    "53": "Real Estate & Leasing",
    "54": "Professional, Scientific & Technical Services",
    "55": "Management of Companies",
    "56": "Administrative & Support Services",
    "61": "Educational Services",
    "62": "Healthcare & Social Assistance",
    "71": "Arts & Recreation",
    "72": "Accommodation & Food Services",
    "81": "Other Services",
    "92": "Public Administration"
}

# ----------------------------------------------------------
# RISK GAUGE
# ----------------------------------------------------------
def risk_grade(pd_val):
    if pd_val < 0.02:
        return "Very Low Risk", "#2ecc71"
    elif pd_val < 0.10:
        return "Low Risk", "#7ed957"
    elif pd_val < 0.30:
        return "Moderate Risk", "#f1c40f"
    elif pd_val < 0.60:
        return "High Risk", "#e67e22"
    else:
        return "Critical Risk", "#e74c3c"

def gauge(pd_value):
    label, color = risk_grade(pd_value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_value,
        number={'valueformat': '.2f'},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 0.02], "color": "#2ecc71"},
                {"range": [0.02, 0.10], "color": "#7ed957"},
                {"range": [0.10, 0.30], "color": "#f1c40f"},
                {"range": [0.30, 0.60], "color": "#e67e22"},
                {"range": [0.60, 1.00], "color": "#e74c3c"},
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10))
    return fig, label, color

# ----------------------------------------------------------
# STRESS SCENARIOS
# ----------------------------------------------------------
SCENARIOS = {
    "Base Case": 1.0,
    "Mild Stress (Economic Slowdown)": 1.2,
    "Moderate Stress (Credit Tightening)": 1.5,
    "Severe Stress (Recession)": 2.0,
    "Extreme Stress (Black Swan Event)": 3.0
}

# ----------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------
st.set_page_config(page_title="Corporate Credit Risk Dashboard", layout="wide")
st.title("Enterprise Credit Risk Early Warning System")
st.write("Corporate Credit Dashboard — PD, LGD, Expected Loss")

# ----------------------------------------------------------
# USER INPUT
# ----------------------------------------------------------
left, right = st.columns(2)

with left:
    Term = st.number_input("Loan Term (Months)", min_value=1, value=60)
    NoEmp = st.number_input("Number of Employees", min_value=0, value=5)
    loan_amt = st.number_input("Loan Amount (USD)", min_value=0.0, value=10000.0)
    log_loan_amt = float(np.log1p(loan_amt))

with right:
    new_business = 1 if st.selectbox("New Business?", ["No", "Yes"]) == "Yes" else 0
    low_doc = 1 if st.selectbox("Low Documentation Loan?", ["No", "Yes"]) == "Yes" else 0
    urban_flag = 1 if st.selectbox("Urban Area?", ["No", "Yes"]) == "Yes" else 0

    naics_key = st.selectbox(
        "Industry Sector (NAICS)",
        list(NAICS_LABELS.keys()),
        format_func=lambda x: f"{x} — {NAICS_LABELS[x]}"
    )
    ApprovalFY = st.selectbox("Loan Approval Year", list(range(1990, 2025)))

scenario_choice = st.sidebar.selectbox("Stress Scenario", list(SCENARIOS.keys()))
stress_mult = SCENARIOS[scenario_choice]

# ----------------------------------------------------------
# RUN PREDICTION
# ----------------------------------------------------------
if st.button("Run Prediction"):

    # Create DataFrame for pipeline input
    X_input = pd.DataFrame([{
        "Term": Term,
        "NoEmp": NoEmp,
        "log_loan_amt": log_loan_amt,
        "new_business": new_business,
        "low_doc": low_doc,
        "urban_flag": urban_flag,
        "NAICS_2": naics_key,
        "ApprovalFY": ApprovalFY
    }])

    # Predict
    pd_val = float(pd_model.predict_proba(X_input)[0][1])
    lgd_val = float(np.clip(lgd_model.predict(X_input)[0], 0, 1))
    el_val = pd_val * lgd_val * loan_amt

    # Stress scenario
    stressed_pd = float(np.clip(pd_val * stress_mult, 0, 1))
    stressed_el = stressed_pd * lgd_val * loan_amt

    col1, col2 = st.columns([1, 2])

    # ------------------------------------------------------
    # PD GAUGE
    # ------------------------------------------------------
    with col1:
        fig, rating, color = gauge(pd_val)
        st.subheader("Risk Level")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h4 style='color:{color}'>{rating}</h4>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # RESULTS TABLE
    # ------------------------------------------------------
    with col2:
        st.subheader("Risk Assessment Results")
        st.write(f"**Probability of Default (PD):** {pd_val:.2%}")
        st.write(f"**Loss Given Default (LGD):** {lgd_val:.2f}")
        st.write(f"**Expected Loss (EL):** ${el_val:,.2f}")
        st.markdown("---")
        st.write(f"**Stress Scenario:** {scenario_choice}")
        st.write(f"**Stressed PD:** {stressed_pd:.2%}")
        st.write(f"**Stressed Expected Loss:** ${stressed_el:,.2f}")

    # ------------------------------------------------------
    # SHAP EXPLAINABILITY
    # ------------------------------------------------------
    st.subheader("Explainability — SHAP Risk Drivers")
    shap_vals = shap_explainer.shap_values(X_input)
    st.pyplot(shap.summary_plot(shap_vals, X_input))
