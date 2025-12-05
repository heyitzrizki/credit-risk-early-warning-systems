import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
import os

# ============================================================
# FIX PATH HANDLER FOR STREAMLIT CLOUD
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def p(path):
    return os.path.join(BASE_DIR, path)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    try:
        pd_model = joblib.load(p("PD_pipeline.pkl"))
        lgd_model = joblib.load(p("LGD_pipeline.pkl"))
        shap_explainer = joblib.load(p("pd_shap_explainer.pkl"))
        return pd_model, lgd_model, shap_explainer
    except Exception as e:
        st.error(f"❌ Model Loading Error: {e}")
        return None, None, None

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

# ============================================================
# EXTRACT METADATA
# ============================================================
if meta:
    NUMERIC = meta["numeric_features"]
    BINARY = meta["binary_features"]
    CATEG = meta["categorical_features"]
    SCALER_MEAN = np.array(meta["scaler_mean"])
    SCALER_SCALE = np.array(meta["scaler_scale"])
    OHE_CATEGORIES = meta["ohe_categories"]
else:
    NUMERIC = []
    BINARY = []
    CATEG = []
    SCALER_MEAN = []
    SCALER_SCALE = []
    OHE_CATEGORIES = {}

# ============================================================
# MANUAL PREPROCESSOR
# ============================================================
def preprocess_single(input_dict):
    df = pd.DataFrame([input_dict])

    # Ensure all required fields exist
    for col in NUMERIC + BINARY + CATEG:
        if col not in df.columns:
            df[col] = 0

    # Numeric scaling
    X_num = df[NUMERIC].astype(float).values
    X_num = (X_num - SCALER_MEAN) / SCALER_SCALE

    # Binary pass-through
    X_bin = df[BINARY].astype(float).values

    # OHE categorical
    ohe_out = []
    for col in CATEG:
        cats = OHE_CATEGORIES[col]
        mapping = {str(v): i for i, v in enumerate(cats)}

        onehot = np.zeros((1, len(cats)))
        val = str(df[col].iloc[0])

        if val in mapping:
            onehot[0, mapping[val]] = 1

        ohe_out.append(onehot)

    X_cat = np.concatenate(ohe_out, axis=1)

    return np.concatenate([X_num, X_bin, X_cat], axis=1)

# ============================================================
# RISK LEVEL HELPER
# ============================================================
def risk_label(pd_value):
    if pd_value < 0.20:
        return "🟢 Low Risk"
    elif pd_value < 0.50:
        return "🟡 Moderate Risk"
    elif pd_value < 0.80:
        return "🟠 High Risk"
    else:
        return "🔴 Severe Risk"

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Credit Risk EWS", layout="wide")

st.title("🏦 Enterprise Credit Risk Early Warning System")
st.caption("Predict Probability of Default, LGD, Expected Loss, and visualize risk levels.")

st.header("Borrower Information")

col1, col2 = st.columns(2)

with col1:
    Term = st.number_input("📆 Loan Term (Months)", min_value=1, value=60)
    NoEmp = st.number_input("👥 Number of Employees", min_value=0, value=10)
    loan_amt = st.number_input("💵 Loan Amount (USD)", min_value=1.0, value=5000.0)
    log_loan_amt = np.log(loan_amt)

with col2:
    new_business = st.selectbox("Is Borrower Newly Established?", ["No", "Yes"])
    low_doc = st.selectbox("Low Documentation Loan?", ["No", "Yes"])
    urban_flag = st.selectbox("Urban Area?", ["No", "Yes"])

# Convert Yes/No → 1/0
new_business = 1 if new_business == "Yes" else 0
low_doc = 1 if low_doc == "Yes" else 0
urban_flag = 1 if urban_flag == "Yes" else 0

# NAICS dropdown (friendly)
naics_options = [f"{code} — Industry Sector" for code in OHE_CATEGORIES.get("NAICS_2", [])]
selected_naics = st.selectbox("🏭 Industry Sector (NAICS)", naics_options)
NAICS_2 = selected_naics.split(" — ")[0]

# Approval Year
ApprovalFY = st.selectbox("📅 Loan Approval Year", OHE_CATEGORIES.get("ApprovalFY", []))

# ============================================================
# PREDICTION
# ============================================================
if st.button("🔎 Run Prediction"):
    if pd_model is None:
        st.error("Models failed to load.")
    else:
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

        st.subheader("Risk Assessment Results")

        st.metric("Probability of Default (PD)", f"{pd_pred:.2%}", help=risk_label(pd_pred))
        st.metric("Loss Given Default (LGD)", f"{lgd_pred:.4f}")
        st.metric("Expected Loss (EL)", f"${el_pred:.2f}")

        st.success(risk_label(pd_pred))

