import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
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

def gauge_color(pd_value):
    if pd_value < 0.20:
        return "#2ecc71"
    elif pd_value < 0.50:
        return "#f1c40f"
    else:
        return "#e74c3c"

def apply_stress(pd_pred, mult):
    return float(np.clip(pd_pred * mult, 0, 1))

LANG = {
    "EN": {
        "title": "Enterprise Credit Risk Early Warning System",
        "subtitle": "Predict PD, LGD, Expected Loss with SHAP Explainability & Stress Test",
        "inputs": "Borrower Information",
        "predict_btn": "Run Prediction",
        "stress_label": "Stress Scenario",
        "pd_result": "Probability of Default (PD)",
        "lgd_result": "Loss Given Default (LGD)",
        "el_result": "Expected Loss (EL)",
        "stress_section": "Stress Scenarios",
        "shap_title": "SHAP Explainability (PD Model)"
    },
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Prediksi PD, LGD, Expected Loss dengan SHAP & Stress Test",
        "inputs": "Informasi Peminjam",
        "predict_btn": "Jalankan Prediksi",
        "stress_label": "Skenario Stress",
        "pd_result": "Probabilitas Gagal Bayar (PD)",
        "lgd_result": "Loss Given Default (LGD)",
        "el_result": "Expected Loss (EL)",
        "stress_section": "Skenario Stress",
        "shap_title": "Penjelasan SHAP (Model PD)"
    },
    "KR": {
        "title": "기업 신용위험 조기경보 시스템",
        "subtitle": "PD, LGD, Expected Loss 예측 및 SHAP 설명 · 스트레스 테스트 제공",
        "inputs": "대출자 정보",
        "predict_btn": "예측 실행",
        "stress_label": "스트레스 시나리오",
        "pd_result": "부도확률 (PD)",
        "lgd_result": "손실률 (LGD)",
        "el_result": "예상손실 (EL)",
        "stress_section": "스트레스 결과",
        "shap_title": "SHAP 설명 (PD 모델)"
    }
}

st.set_page_config(page_title="Credit Risk EWS", layout="wide")

lang_choice = st.sidebar.selectbox("Language / Bahasa / 언어", ["EN", "ID", "KR"])
T = LANG[lang_choice]

st.title(T["title"])
st.write(T["subtitle"])
st.write("")

scenario = st.sidebar.selectbox(
    T["stress_label"],
    ["Base Case", "Mild Stress (Economic Slowdown)", "Moderate Stress (Credit Tightening)", 
     "Severe Stress (Recession)", "Extreme Stress (Black Swan Event)"]
)

scenario_map = {
    "Base Case": 1.0,
    "Mild Stress (Economic Slowdown)": 1.2,
    "Moderate Stress (Credit Tightening)": 1.5,
    "Severe Stress (Recession)": 2.0,
    "Extreme Stress (Black Swan Event)": 3.0
}

stress_mult = scenario_map[scenario]

st.sidebar.info(f"Stress Multiplier Applied: ×{stress_mult}")

st.subheader(T["inputs"])

col1, col2 = st.columns(2)

with col1:
    Term = st.number_input("Loan Term (Months)", min_value=1, value=60)
    NoEmp = st.number_input("Number of Employees", min_value=0, value=10)
    loan_amt = st.number_input("Loan Amount (USD)", min_value=0.0, value=5000.0)
    log_loan_amt = float(np.log1p(loan_amt))

with col2:
    new_business = st.selectbox("Is this a new business?", ["No", "Yes"])
    low_doc = st.selectbox("Low documentation loan?", ["No", "Yes"])
    urban_flag = st.selectbox("Urban area?", ["No", "Yes"])

new_business = 1 if new_business == "Yes" else 0
low_doc = 1 if low_doc == "Yes" else 0
urban_flag = 1 if urban_flag == "Yes" else 0

NAICS_map = {code: f"{code} — Industry Sector" for code in OHE_CATEGORIES["NAICS_2"]}
Approval_map = {yr: str(yr) for yr in OHE_CATEGORIES["ApprovalFY"]}

NAICS_2 = st.selectbox("Industry Sector (NAICS)", list(NAICS_map.keys()), format_func=lambda x: NAICS_map[x])
ApprovalFY = st.selectbox("Loan Approval Year", list(Approval_map.keys()))

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

        pd_pred = float(pd_model.predict_proba(X)[0][1])
        lgd_pred = float(lgd_model.predict(X)[0])
        el_pred = pd_pred * lgd_pred

        stressed_pd = apply_stress(pd_pred, stress_mult)
        stressed_el = stressed_pd * lgd_pred

        color = gauge_color(pd_pred)
        st.markdown(
            f"""
            <div style='padding:20px; background-color:{color}; border-radius:10px; color:white; font-size:22px; text-align:center;'>
                PD Risk Level: {round(pd_pred,4)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader(T["pd_result"])
        st.success(round(pd_pred, 4))

        st.subheader(T["lgd_result"])
        st.success(round(lgd_pred, 4))

        st.subheader(T["el_result"])
        st.success(round(el_pred, 4))

        st.subheader(T["stress_section"])
        st.info(f"Stressed PD: {round(stressed_pd,4)}")
        st.info(f"Stressed Expected Loss: {round(stressed_el,4)}")

        st.subheader(T["shap_title"])
        shap_values = shap_explainer.shap_values(X)
        st.pyplot(shap.summary_plot(shap_values, X))
