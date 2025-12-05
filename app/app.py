import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import plotly.graph_objects as go
import plotly.express as px

# =====================================
# 1. PAGE CONFIG & STYLE
# =====================================
st.set_page_config(page_title="Enterprise Credit Risk EWS", layout="wide", page_icon="💼")

FINTECH_BLUE = "#0A2647"
LIGHT_BLUE = "#144272"
ACCENT_BLUE = "#205295"
BG_GRAY = "#f5f6fa"

st.markdown(f"""
    <style>
        .main {{ background-color: {BG_GRAY}; }}
        .title-text {{ color: {FINTECH_BLUE}; font-size: 36px; font-weight: 700; }}
        .sub-text {{ color: {ACCENT_BLUE}; font-size: 18px; }}
    </style>
""", unsafe_allow_html=True)

# =====================================
# 2. MULTILINGUAL DICTIONARY
# =====================================
LANG = {
    "EN": {
        "title": "Enterprise Credit Risk Early Warning System",
        "subtitle": "Predict PD, LGD, Expected Loss, with SHAP Explainability & Stress Test",
        "input_header": "Borrower Information",
        "industry": "Industry Sector",
        "loan_amt": "Loan Amount (USD)",
        "employees": "Number of Employees",
        "new_business": "New Business?",
        "yes": "Yes",
        "no": "No",
        "predict": "Run Credit Risk Analysis",
        "stress_label": "Market Stress (VIX Index)",
        "pd_base": "Baseline PD",
        "pd_stress": "Stressed PD",
        "lgd": "Loss Given Default",
        "el": "Expected Loss",
        "explain": "SHAP Local Explanation",
    },
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Prediksi PD, LGD, Expected Loss, dengan SHAP & Stress Test",
        "input_header": "Informasi Debitur",
        "industry": "Sektor Industri",
        "loan_amt": "Jumlah Pinjaman (USD)",
        "employees": "Jumlah Karyawan",
        "new_business": "Bisnis Baru?",
        "yes": "Ya",
        "no": "Tidak",
        "predict": "Jalankan Analisis Risiko Kredit",
        "stress_label": "Stress Pasar (VIX Index)",
        "pd_base": "PD Normal",
        "pd_stress": "PD Stres",
        "lgd": "Loss Given Default",
        "el": "Expected Loss",
        "explain": "Penjelasan SHAP",
    },
    "KR": {
        "title": "기업 신용위험 조기경보 시스템",
        "subtitle": "PD, LGD, Expected Loss 예측 및 SHAP 설명 · 스트레스 테스트 제공",
        "input_header": "차입자 정보",
        "industry": "산업 분야",
        "loan_amt": "대출 금액 (USD)",
        "employees": "직원 수",
        "new_business": "신규 사업 여부",
        "yes": "예",
        "no": "아니요",
        "predict": "신용위험 분석 실행",
        "stress_label": "시장 스트레스 (VIX 지수)",
        "pd_base": "기본 PD",
        "pd_stress": "스트레스 PD",
        "lgd": "LGD",
        "el": "기대손실",
        "explain": "SHAP 설명",
    }
}

# =====================================
# 3. INDUSTRY → NAICS MAPPING
# =====================================
INDUSTRY_TO_NAICS = {
    "Manufacturing": "31",
    "Retail": "44",
    "Services": "54",
    "Finance": "52",
    "Technology": "51",
    "Healthcare": "62",
    "Real Estate": "53",
    "Construction": "23",
    "Transportation": "48",
    "Agriculture": "11",
}

# =====================================
# 4. LOAD MODELS & EXPLAINER
# =====================================
try:
    PD_PIPE = joblib.load("PD_model_pipeline.pkl")
    LGD_PIPE = joblib.load("LGD_model_pipeline.pkl")
    SHAP_EXPLAINER = joblib.load("pd_shap_explainer.pkl")
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    st.error(f"Model Loading Error: {e}")

# =====================================
# 5. STRESS TEST FUNCTION
# =====================================
def apply_stress(pd_base, vix):
    baseline = 20
    if vix <= baseline:
        mult = 1.0
    else:
        mult = 1.0 + ((vix - baseline) / 100) * 1.5
    return min(pd_base * mult, 1.0), mult

# =====================================
# 6. UI LANGUAGE SELECTION
# =====================================
st.sidebar.header("🌐 Language / Bahasa / 언어")
lang_choice = st.sidebar.selectbox("Select Language", ["EN", "ID", "KR"])
TXT = LANG[lang_choice]

# =====================================
# 7. MAIN TITLE
# =====================================
st.markdown(f"<div class='title-text'>{TXT['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-text'>{TXT['subtitle']}</div>", unsafe_allow_html=True)

# =====================================
# 8. INPUT FORM (NON-TECH FRIENDLY)
# =====================================
st.write("---")
st.subheader(TXT["input_header"])

col1, col2 = st.columns(2)

industry = col1.selectbox(TXT["industry"], list(INDUSTRY_TO_NAICS.keys()))
loan_amt = col1.number_input(TXT["loan_amt"], min_value=1000.0, value=50000.0)
employees = col2.number_input(TXT["employees"], min_value=1, value=20)
new_business = col2.selectbox(TXT["new_business"], [TXT["yes"], TXT["no"]])

newbiz_flag = 1 if new_business == TXT["yes"] else 0
naics_val = INDUSTRY_TO_NAICS[industry]

input_df = pd.DataFrame({
    "NAICS": [naics_val],
    "DisbursementGross": [loan_amt],
    "NoEmp": [employees],
    "new_business": [newbiz_flag],
    "low_doc": [0],
    "urban_flag": [1],
    "log_loan_amt": [np.log1p(loan_amt)],
    "Term": [12],
})

# =====================================
# 9. RUN ANALYSIS BUTTON
# =====================================
run_btn = st.button(TXT["predict"])

if run_btn:

    if not MODEL_READY:
        st.error("Model not loaded.")
        st.stop()

    # ---------- PREDICT PD ----------
    pd_val = PD_PIPE.predict_proba(input_df)[0][1]

    # ---------- PREDICT LGD ----------
    lgd_val = LGD_PIPE.predict(input_df)[0]

    # ---------- EXPECTED LOSS ----------
    el_base = pd_val * lgd_val * loan_amt

    st.success("Prediction completed.")

    # Display Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(TXT["pd_base"], f"{pd_val:.2%}")
    m2.metric(TXT["lgd"], f"{lgd_val:.2%}")
    m3.metric(TXT["el"], f"${el_base:,.0f}")

    # ---------- STRESS TEST SECTION ----------
    st.write("---")
    st.subheader(TXT["stress_label"])

    vix_level = st.slider("VIX", 10, 80, 20)

    pd_stress, mult = apply_stress(pd_val, vix_level)
    el_stress = pd_stress * lgd_val * loan_amt

    s1, s2 = st.columns(2)
    s1.metric(TXT["pd_stress"], f"{pd_stress:.2%}")
    s2.metric(TXT["el"], f"${el_stress:,.0f}")

    st.info(f"Multiplier: {mult:.2f}x | Difference: {el_stress - el_base:,.0f}")

    # ---------- SHAP EXPLANATION ----------
    st.write("---")
    st.subheader(TXT["explain"])

    transformed_X = PD_PIPE.named_steps["preprocess"].transform(input_df)
    shap_vals = SHAP_EXPLAINER.shap_values(transformed_X)[0]
    feature_names = PD_PIPE.named_steps["preprocess"].get_feature_names_out()

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "importance": shap_vals
    }).sort_values(by="importance", key=abs, ascending=False).head(10)

    fig = px.bar(shap_df, x="importance", y="feature", orientation="h", title="Top SHAP Factors")
    st.plotly_chart(fig, use_container_width=True)
