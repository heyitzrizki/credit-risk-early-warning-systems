import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap

# =========================================================
# PATH ARTIFACT (Root = folder app/ di Streamlit Cloud)
# =========================================================
MODEL_PD_PATH = "PD_model.pkl"
MODEL_LGD_PATH = "LGD_model.pkl"
SHAP_PATH = "pd_shap_explainer.pkl"
PREPROCESSOR_JSON = "preprocessor_meta.json"

# =========================================================
# LOAD ARTIFACTS
# =========================================================
@st.cache_resource
def load_models():
    try:
        pd_model = joblib.load(MODEL_PD_PATH)
        lgd_model = joblib.load(MODEL_LGD_PATH)
        shap_explainer = joblib.load(SHAP_PATH)
        return pd_model, lgd_model, shap_explainer
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None, None, None

@st.cache_resource
def load_preprocessor():
    try:
        with open(PREPROCESSOR_JSON, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Preprocessor Metadata Error: {e}")
        return None

pd_model, lgd_model, shap_explainer = load_models()
meta = load_preprocessor()

# Extract metadata
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
# MANUAL PREPROCESSING
# =========================================================
def preprocess_single(input_dict):
    df = pd.DataFrame([input_dict])

    # Ensure all required cols exist
    for col in NUMERIC + BINARY + CATEG:
        if col not in df.columns:
            df[col] = 0

    # Numeric scaling
    X_num = df[NUMERIC].astype(float).values
    X_num = (X_num - SCALER_MEAN) / SCALER_SCALE

    # Binary pass-through
    X_bin = df[BINARY].astype(float).values

    # OHE categorical
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
# STRESS TEST FUNCTION
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
        "inputs": "Borrower Input Features",
        "predict_btn": "Run Prediction",
        "stress_label": "Stress Test Multiplier",
        "pd_result": "Predicted Probability of Default (PD)",
        "lgd_result": "Predicted Loss Given Default (LGD)",
        "el_result": "Expected Loss (EL)",
        "stress_section": "Stressed PD & Expected Loss",
        "shap_title": "SHAP Explainability (PD Model)"
    },
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Prediksi PD, LGD, Expected Loss dengan SHAP & Stress Test",
        "inputs": "Masukkan Fitur Peminjam",
        "predict_btn": "Jalankan Prediksi",
        "stress_label": "Multiplier Stress Test",
        "pd_result": "Probabilitas Gagal Bayar (PD)",
        "lgd_result": "Loss Given Default (LGD)",
        "el_result": "Expected Loss (EL)",
        "stress_section": "PD & Expected Loss setelah Stress",
        "shap_title": "Penjelasan SHAP (Model PD)"
    },
    "KR": {
        "title": "기업 신용위험 조기경보 시스템",
        "subtitle": "PD, LGD, Expected Loss 예측 및 SHAP 설명 · 스트레스 테스트",
        "inputs": "대출자 입력 정보",
        "predict_btn": "예측 실행",
        "stress_label": "스트레스 테스트 배수",
        "pd_result": "부도확률 (PD)",
        "lgd_result": "손실률 (LGD)",
        "el_result": "예상손실 (EL)",
        "stress_section": "스트레스 적용 후 PD & Expected Loss",
        "shap_title": "SHAP 설명 (PD 모델)"
    },
}

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Credit Risk EWS", layout="wide")

lang_choice = st.sidebar.selectbox("Language / Bahasa / 언어", ["EN", "ID", "KR"])
T = LANG[lang_choice]

st.title(T["title"])
st.write(T["subtitle"])
st.write("")

# Stress slider
stress_mult = st.sidebar.slider(T["stress_label"], 0.5, 3.0, 1.0, 0.1)

# ================================
# Manual Input Section
# ================================
st.subheader(T["inputs"])

col1, col2 = st.columns(2)

with col1:
    Term = st.number_input("Term (months)", min_value=1, value=60)
    NoEmp = st.number_input("Number of Employees", min_value=0, value=5)
    log_loan_amt = st.number_input("log(Loan Amount)", value=10.5)

with col2:
    new_business = st.selectbox("New Business (0/1)", [0,1])
    low_doc = st.selectbox("Low Documentation (0/1)", [0,1])
    urban_flag = st.selectbox("Urban Area (0/1)", [0,1])

NAICS_2 = st.selectbox("NAICS Sector", OHE_CATEGORIES["NAICS_2"])
ApprovalFY = st.selectbox("Fiscal Year", OHE_CATEGORIES["ApprovalFY"])

# ================================
# Predict Button
# ================================
if st.button(T["predict_btn"]):
    if pd_model is None:
        st.error("Model failed to load.")
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

        # Baseline predictions
        pd_pred = float(pd_model.predict_proba(X)[0][1])
        lgd_pred = float(lgd_model.predict(X)[0])
        el_pred = pd_pred * lgd_pred

        # Stress
        stressed_pd = apply_stress(pd_pred, stress_mult)
        stressed_el = stressed_pd * lgd_pred

        # Results
        st.subheader(T["pd_result"])
        st.success(round(pd_pred, 4))

        st.subheader(T["lgd_result"])
        st.success(round(lgd_pred, 4))

        st.subheader(T["el_result"])
        st.success(round(el_pred, 4))

        st.subheader(T["stress_section"])
        st.info(f"Stressed PD: {round(stressed_pd,4)}")
        st.info(f"Stressed Expected Loss: {round(stressed_el,4)}")

        # SHAP plot
        st.subheader(T["shap_title"])
        shap_values = shap_explainer.shap_values(X)
        st.set_option("deprecation.showPyplotGlobalUse", False)
        st.pyplot(shap.summary_plot(shap_values, X))

