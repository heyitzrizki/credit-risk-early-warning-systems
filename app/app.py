import streamlit as st
import joblib
import json
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
# LOAD MODELS (USE PIPELINE FILES!)
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        pd_model = joblib.load(p("PD_pipeline.pkl"))          # UPDATED
        lgd_model = joblib.load(p("LGD_pipeline.pkl"))        # UPDATED
        shap_explainer = joblib.load(p("pd_shap_explainer.pkl"))
        return pd_model, lgd_model, shap_explainer
    except Exception as e:
        st.error(f"❌ Model Loading Error: {e}")
        return None, None, None

# ----------------------------------------------------------
# LOAD PREPROCESSOR META
# ----------------------------------------------------------
@st.cache_resource
def load_preprocessor_meta():
    try:
        with open(p("preprocessor_meta.json"), "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Preprocessor Metadata Error: {e}")
        return None

pd_model, lgd_model, shap_explainer = load_models()
meta = load_preprocessor_meta()

if meta:
    NUMERIC = meta["numeric_features"]
    BINARY = meta["binary_features"]
    CATEG = meta["categorical_features"]
    SCALER_MEAN = np.array(meta["scaler_mean"])
    SCALER_SCALE = np.array(meta["scaler_scale"])
    OHE_CATEGORIES = meta["ohe_categories"]
else:
    NUMERIC = BINARY = CATEG = []
    SCALER_MEAN = SCALER_SCALE = []
    OHE_CATEGORIES = {}

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
    "92": "Public Administration",
    "Un": "Unknown"
}

# ----------------------------------------------------------
# MANUAL PREPROCESSING
# ----------------------------------------------------------
def preprocess_single(input_dict):
    df = pd.DataFrame([input_dict])

    for col in NUMERIC + BINARY + CATEG:
        if col not in df.columns:
            df[col] = 0

    X_num = df[NUMERIC].astype(float).values
    X_num = (X_num - SCALER_MEAN) / SCALER_SCALE

    X_bin = df[BINARY].astype(float).values

    ohe_arrays = []
    for col in CATEG:
        categories = OHE_CATEGORIES[col]
        mapping = {str(v): i for i, v in enumerate(categories)}
        encoded = np.zeros((1, len(categories)))
        val = str(df[col].iloc[0])
        if val in mapping:
            encoded[0, mapping[val]] = 1
        ohe_arrays.append(encoded)

    X_cat = np.concatenate(ohe_arrays, axis=1)

    return np.concatenate([X_num, X_bin, X_cat], axis=1)

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
# UI START
# ----------------------------------------------------------
st.set_page_config(page_title="Corporate Credit Risk Dashboard", layout="wide")

st.title("Enterprise Credit Risk Early Warning System")
st.write("Corporate Credit Dashboard — PD, LGD, Expected Loss")

# INPUT FORM
left, right = st.columns(2)

with left:
    Term = st.number_input("Loan Term (Months)", min_value=1, value=60)
    NoEmp = st.number_input("Number of Employees", min_value=0, value=5)
    loan_amt = st.number_input("Loan Amount (USD)", min_value=0.0, value=10000.0)
    log_loan_amt = float(np.log1p(loan_amt))

with right:
    new_business = st.selectbox("New Business?", ["No", "Yes"])
    low_doc = st.selectbox("Low Documentation Loan?", ["No", "Yes"])
    urban_flag = st.selectbox("Urban Area?", ["No", "Yes"])

new_business = 1 if new_business == "Yes" else 0
low_doc = 1 if low_doc == "Yes" else 0
urban_flag = 1 if urban_flag == "Yes" else 0

# Friendly NAICS
naics_display = [f"{k} — {NAICS_LABELS[k]}" for k in OHE_CATEGORIES["NAICS_2"]]
sel = st.selectbox("Industry Sector (NAICS)", naics_display)
NAICS_2 = sel.split(" — ")[0]

ApprovalFY = st.selectbox("Loan Approval Year", OHE_CATEGORIES["ApprovalFY"])

scenario_choice = st.sidebar.selectbox("Stress Scenario", list(SCENARIOS.keys()))
stress_mult = SCENARIOS[scenario_choice]

# ----------------------------------------------------------
# RUN PREDICTION
# ----------------------------------------------------------
if st.button("Run Prediction"):
    X = preprocess_single({
        "Term": Term,
        "NoEmp": NoEmp,
        "log_loan_amt": log_loan_amt,
        "new_business": new_business,
        "low_doc": low_doc,
        "urban_flag": urban_flag,
        "NAICS_2": NAICS_2,
        "ApprovalFY": ApprovalFY
    })

    pd_val = float(pd_model.predict_proba(X)[0][1])
    lgd_val = float(np.clip(lgd_model.predict(X)[0], 0, 1))
    el_val = pd_val * lgd_val * loan_amt

    stressed_pd = float(np.clip(pd_val * stress_mult, 0, 1))
    stressed_el = stressed_pd * lgd_val * loan_amt

    col1, col2 = st.columns([1, 2])

    with col1:
        fig, rating, color = gauge(pd_val)
        st.subheader("Risk Level")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h4 style='color:{color}'>{rating}</h4>", unsafe_allow_html=True)

    with col2:
        st.subheader("Risk Assessment Results")
        st.write(f"**Probability of Default (PD):** {pd_val:.2%}")
        st.write(f"**Loss Given Default (LGD):** {lgd_val:.2f}")
        st.write(f"**Expected Loss (EL):** ${el_val:,.2f}")
        st.markdown("---")
        st.write(f"**Stress Scenario:** {scenario_choice}")
        st.write(f"**Stressed PD:** {stressed_pd:.2%}")
        st.write(f"**Stressed Expected Loss:** ${stressed_el:,.2f}")

    st.subheader("Explainability — SHAP Risk Drivers")
    shap_vals = shap_explainer.shap_values(X)
    st.pyplot(shap.summary_plot(shap_vals, X))
