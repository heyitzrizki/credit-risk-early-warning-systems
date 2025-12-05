import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
import os

# =========================================================
# FIX PATH HANDLING
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def p(path):
    return os.path.join(BASE_DIR, path)

# =========================================================
# LOAD MODELS & PREPROCESSOR
# =========================================================
@st.cache_resource
def load_models():
    try:
        pd_model = joblib.load(p("PD_model.pkl"))
        lgd_model = joblib.load(p("LGD_model.pkl"))
        shap_explainer = joblib.load(p("pd_shap_explainer.pkl"))
        return pd_model, lgd_model, shap_explainer
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None, None, None

@st.cache_resource
def load_preprocessor_meta():
    try:
        with open(p("preprocessor_meta.json"), "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Preprocessor Metadata Error: {e}")
        return None

pd_model, lgd_model, shap_explainer = load_models()
meta = load_preprocessor_meta()

# =========================================================
# EXTRACT PREPROCESSOR DETAILS
# =========================================================
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

# =========================================================
# FULL FEATURE NAME MAPPING for SHAP
# =========================================================
FULL_FEATURE_NAMES = (
    NUMERIC +
    BINARY +
    [f"NAICS Sector: {cat}" for cat in OHE_CATEGORIES["NAICS_2"]] +
    [f"Approval Year: {cat}" for cat in OHE_CATEGORIES["ApprovalFY"]]
)

# =========================================================
# MANUAL PREPROCESSING (SINGLE INPUT)
# =========================================================
def preprocess_single(input_dict):
    df = pd.DataFrame([input_dict])

    for col in NUMERIC + BINARY + CATEG:
        if col not in df.columns:
            df[col] = 0

    # Numeric scaling
    X_num = df[NUMERIC].astype(float).values
    X_num = (X_num - SCALER_MEAN) / SCALER_SCALE

    # Binary passthrough
    X_bin = df[BINARY].astype(float).values

    # OHE
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

# =========================================================
# STRESS TEST
# =========================================================
def apply_stress(pd_pred, mult):
    return float(np.clip(pd_pred * mult, 0, 1))

# =========================================================
# LANGUAGE PACK
# =========================================================
LANG = {
    "EN": {
        "title": "Enterprise Credit Risk Early Warning System",
        "subtitle": "Predict PD, LGD, Expected Loss with SHAP Explainability & Stress Test",
        "inputs": "Borrower Information",
        "predict_btn": "Run Prediction",
        "stress_label": "Stress Test Multiplier",
        "pd_result": "Probability of Default (PD)",
        "lgd_result": "Loss Given Default (LGD)",
        "el_result": "Expected Loss (EL)",
        "stress_section": "Stress Scenarios",
        "shap_title": "SHAP Explainability (PD Model)",
        "term": "Loan Term (Months)",
        "emp": "Number of Employees",
        "loan_amt": "Loan Amount (USD)",
        "newbiz": "Is this a new business?",
        "lowdoc": "Low documentation loan?",
        "urban": "Urban area?",
        "naics": "Industry Sector (NAICS)",
        "fy": "Loan Approval Year"
    }
}

T = LANG["EN"]

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Credit Risk EWS", layout="wide")
st.title(T["title"])
st.write(T["subtitle"])
st.write("")

# Stress slider
stress_mult = st.sidebar.slider(T["stress_label"], 0.5, 3.0, 1.0, 0.1)

# =========================================================
# USER-FRIENDLY INPUTS
# =========================================================
st.subheader(T["inputs"])

col1, col2 = st.columns(2)

with col1:
    Term = st.number_input(T["term"], min_value=1, value=60)
    NoEmp = st.number_input(T["emp"], min_value=0, value=10)
    loan_amt = st.number_input(T["loan_amt"], min_value=1000.0, value=50000.0)

with col2:
    new_business = st.selectbox(T["newbiz"], ["Yes", "No"])
    low_doc = st.selectbox(T["lowdoc"], ["Yes", "No"])
    urban_flag = st.selectbox(T["urban"], ["Yes", "No"])

NAICS_2 = st.selectbox(T["naics"], OHE_CATEGORIES["NAICS_2"])
ApprovalFY = st.selectbox(T["fy"], OHE_CATEGORIES["ApprovalFY"])

# Convert to model-friendly values
new_business = 1 if new_business == "Yes" else 0
low_doc = 1 if low_doc == "Yes" else 0
urban_flag = 1 if urban_flag == "Yes" else 0
log_loan_amt = np.log(loan_amt)

# =========================================================
# PREDICT BUTTON
# =========================================================
if st.button(T["predict_btn"]):

    input_data = {
        "Term": Term,
        "NoEmp": NoEmp,
        "log_loan_amt": log_loan_amt,
        "new_business": new_business,
        "low_doc": low_doc,
        "urban_flag": urban_flag,
        "NAICS_2": NAICS_2,
        "ApprovalFY": ApprovalFY
    }

    X = preprocess_single(input_data)

    pd_pred = float(pd_model.predict_proba(X)[0][1])
    lgd_pred = float(lgd_model.predict(X)[0])
    el_pred = pd_pred * lgd_pred

    stressed_pd = apply_stress(pd_pred, stress_mult)
    stressed_el = stressed_pd * lgd_pred

    # =====================================================
    # DISPLAY RESULTS (USER-FRIENDLY)
    # =====================================================
    st.subheader(T["pd_result"])
    st.success(f"{pd_pred*100:.2f}%")

    st.subheader(T["lgd_result"])
    st.success(f"{lgd_pred:.4f}")

    st.subheader(T["el_result"])
    st.success(f"{el_pred:.4f}")

    st.subheader(T["stress_section"])
    st.info(f"Stressed PD: {stressed_pd*100:.2f}%")
    st.info(f"Stressed Expected Loss: {stressed_el:.4f}")

    # =====================================================
    # SHAP EXPLAINABILITY
    # =====================================================
    st.subheader(T["shap_title"])
    shap_values = shap_explainer.shap_values(X)

    # Summary Plot
    shap.summary_plot(shap_values, X, feature_names=FULL_FEATURE_NAMES, plot_type="bar", show=False)
    st.pyplot()

    # Top Features
    top_df = pd.DataFrame({
        "Feature": FULL_FEATURE_NAMES,
        "Impact": np.abs(shap_values).flatten()
    }).sort_values("Impact", ascending=False).head(10)

    st.write("### Top 10 Most Influential Features")
    st.dataframe(top_df.reset_index(drop=True))
