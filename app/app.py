import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def p(path): return os.path.join(BASE_DIR, path)

@st.cache_resource
def load_models():
    try:
        pd_model = joblib.load(p("PD_model.pkl"))
        lgd_model = joblib.load(p("LGD_model.pkl"))
        shap_explainer = joblib.load(p("pd_shap_explainer.pkl"))
        return pd_model, lgd_model, shap_explainer
    except:
        return None, None, None

@st.cache_resource
def load_preprocessor_meta():
    try:
        with open(p("preprocessor_meta.json"), "r") as f:
            return json.load(f)
    except:
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
        v = str(df[col].iloc[0])
        if v in mapping:
            encoded[0, mapping[v]] = 1
        ohe_arrays.append(encoded)

    X_cat = np.concatenate(ohe_arrays, axis=1)
    return np.concatenate([X_num, X_bin, X_cat], axis=1)

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
    risk_label, color = risk_grade(pd_value)
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
    return fig, risk_label, color

SCENARIOS = {
    "Base Case": 1.0,
    "Mild Stress (Economic Slowdown)": 1.2,
    "Moderate Stress (Credit Tightening)": 1.5,
    "Severe Stress (Recession)": 2.0,
    "Extreme Stress (Black Swan Event)": 3.0
}

LANG = {
    "EN": {
        "title": "Enterprise Credit Risk Early Warning System",
        "subtitle": "Corporate Credit Risk Dashboard — PD, LGD, Expected Loss",
        "inputs": "Borrower Information",
        "predict": "Run Prediction",
        "stress": "Stress Scenario",
        "results": "Risk Assessment Results",
        "shap": "Risk Drivers (Explainability)"
    }
}

T = LANG["EN"]

st.set_page_config(page_title="Corporate Credit Risk EWS", layout="wide")

st.title(T["title"])
st.write(T["subtitle"])

left, right = st.columns([1, 1])

with left:
    Term = st.number_input("Loan Term (Months)", min_value=1, value=60)
    NoEmp = st.number_input("Number of Employees", min_value=0, value=10)
    loan_amt = st.number_input("Loan Amount (USD)", min_value=0.0, value=10000.0)
    log_loan_amt = float(np.log1p(loan_amt))

with right:
    new_business = st.selectbox("Is Borrower Newly Established?", ["No", "Yes"])
    low_doc = st.selectbox("Low Documentation Loan?", ["No", "Yes"])
    urban_flag = st.selectbox("Urban Area?", ["No", "Yes"])

new_business = 1 if new_business == "Yes" else 0
low_doc = 1 if low_doc == "Yes" else 0
urban_flag = 1 if urban_flag == "Yes" else 0

naics_display = [f"{k} — {v}" for k, v in NAICS_LABELS.items() if k in OHE_CATEGORIES["NAICS_2"]]
sel_naics = st.selectbox("Industry Sector (NAICS)", naics_display)
NAICS_2 = sel_naics.split(" — ")[0]

ApprovalFY = st.selectbox("Loan Approval Year", OHE_CATEGORIES["ApprovalFY"])

scenario_choice = st.sidebar.selectbox(T["stress"], list(SCENARIOS.keys()))
stress_mult = SCENARIOS[scenario_choice]

if st.button(T["predict"]):
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

    rcol1, rcol2 = st.columns([1, 2])

    with rcol1:
        fig, rating, color = gauge(pd_val)
        st.subheader("Risk Level")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h4 style='color:{color};'>{rating}</h4>", unsafe_allow_html=True)

    with rcol2:
        st.subheader(T["results"])
        st.write(f"**Probability of Default (PD):** {pd_val:.2%}")
        st.write(f"**Loss Given Default (LGD):** {lgd_val:.2f}")
        st.write(f"**Expected Loss (EL):** ${el_val:,.2f}")
        st.markdown("---")
        st.write(f"**Stress Scenario:** {scenario_choice}")
        st.write(f"**Stressed PD:** {stressed_pd:.2%}")
        st.write(f"**Stressed Expected Loss:** ${stressed_el:,.2f}")

    st.subheader(T["shap"])
    shap_vals = shap_explainer.shap_values(X)
    st.pyplot(shap.summary_plot(shap_vals, X))
