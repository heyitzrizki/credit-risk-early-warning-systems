import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

# ======================================================================
# Path Loader
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PD_MODEL_PATH = os.path.join(BASE_DIR, "PD_model_pipeline.pkl")
LGD_MODEL_PATH = os.path.join(BASE_DIR, "LGD_model_pipeline.pkl")
SHAP_PATH = os.path.join(BASE_DIR, "pd_shap_explainer.pkl")

# ======================================================================
# Load Artifacts
# ======================================================================
@st.cache_resource
def load_models():
    pd_model = joblib.load(PD_MODEL_PATH)
    lgd_model = joblib.load(LGD_MODEL_PATH)
    explainer = joblib.load(SHAP_PATH)
    return pd_model, lgd_model, explainer

try:
    pd_model, lgd_model, shap_explainer = load_models()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# ======================================================================
# Language Packs
# ======================================================================
LANG = {
    "EN": {
        "title": "Enterprise Credit Risk Early Warning System",
        "sub": "Predict PD, LGD, Expected Loss, with SHAP Explainability & Stress Test",
        "input_section": "Borrower Information Input",
        "run_btn": "Run Risk Analysis",
        "stress_label": "Stress Scenario Multiplier (Default: 1.2)",
        "shap_title": "Feature Contribution (SHAP)",
        "result_title": "Risk Result",
        "pd_base": "Baseline Probability of Default",
        "pd_stress": "Stressed Probability of Default",
        "lgd_label": "Loss Given Default Prediction",
        "el_label": "Expected Loss",
    },
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "sub": "Prediksi PD, LGD, Expected Loss, dengan SHAP & Stress Testing",
        "input_section": "Input Informasi Peminjam",
        "run_btn": "Jalankan Analisis Risiko",
        "stress_label": "Multiplier Stress Scenario (Default: 1.2)",
        "shap_title": "Kontribusi Fitur (SHAP)",
        "result_title": "Hasil Risiko",
        "pd_base": "Probabilitas Gagal Bayar (Baseline)",
        "pd_stress": "Probabilitas Gagal Bayar (Stres)",
        "lgd_label": "Prediksi LGD",
        "el_label": "Expected Loss",
    },
    "KR": {
        "title": "기업 신용위험 조기경보 시스템",
        "sub": "PD, LGD, Expected Loss 예측 및 SHAP 설명가능성 + 스트레스 테스트",
        "input_section": "차입자 정보 입력",
        "run_btn": "리스크 분석 실행",
        "stress_label": "스트레스 시나리오 배수 (기본값: 1.2)",
        "shap_title": "특성 기여도 (SHAP)",
        "result_title": "리스크 결과",
        "pd_base": "기본 부도확률",
        "pd_stress": "스트레스 부도확률",
        "lgd_label": "LGD 예측",
        "el_label": "Expected Loss",
    }
}

# ======================================================================
# UI
# ======================================================================
st.set_page_config(page_title="Credit Risk EWS", layout="wide")

st.sidebar.title("🌐 Language / Bahasa / 언어")
lang_choice = st.sidebar.selectbox("Select Language", ["EN", "ID", "KR"])
T = LANG[lang_choice]

st.title(T["title"])
st.write(T["sub"])

st.markdown("### " + T["input_section"])

# ======================================================================
# Input Fields (adjust these to match your training features!)
# ======================================================================
col1, col2, col3 = st.columns(3)

with col1:
    term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=36)
    no_emp = st.number_input("Number of Employees", min_value=0, value=10)

with col2:
    loan_amt = st.number_input("Loan Amount", min_value=1000, value=50000)
    new_business = st.selectbox("New Business?", ["Yes", "No"])

with col3:
    low_doc = st.selectbox("Low Documentation?", ["Yes", "No"])
    urban = st.selectbox("Urban Area?", ["Yes", "No"])

stress_mult = st.slider(T["stress_label"], 1.0, 2.0, 1.2, 0.05)

# Build dataframe
input_df = pd.DataFrame([{
    "Term": term,
    "NoEmp": no_emp,
    "log_loan_amt": np.log1p(loan_amt),
    "new_business": 1 if new_business == "Yes" else 0,
    "low_doc": 1 if low_doc == "Yes" else 0,
    "urban_flag": 1 if urban == "Yes" else 0
}])

# ======================================================================
# RUN ANALYSIS
# ======================================================================
if st.button(T["run_btn"]):

    # --------------------
    # Predictions
    # --------------------
    pd_base = float(pd_model.predict_proba(input_df)[0][1])
    lgd_pred = float(lgd_model.predict(input_df)[0])
    el_value = pd_base * lgd_pred * loan_amt

    # --------------------
    # Stress Testing
    # --------------------
    pd_stress = min(pd_base * stress_mult, 1.0)

    # --------------------
    # SHAP Explainability
    # --------------------
    shap_vals = shap_explainer(input_df)

    # --------------------
    # Output
    # --------------------
    st.subheader(T["result_title"])

    st.metric(T["pd_base"], f"{pd_base:.3f}")
    st.metric(T["pd_stress"], f"{pd_stress:.3f}")
    st.metric(T["lgd_label"], f"{lgd_pred:.3f}")
    st.metric(T["el_label"], f"${el_value:,.0f}")

    # SHAP plot
    st.subheader(T["shap_title"])
    shap_fig = shap.plots.waterfall(shap_vals[0], show=False)
    st.pyplot(shap_fig)
