import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ====================================================================================
# FIX PATH — ensure Streamlit loads files relative to folder containing app.py
# ====================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def p(path):
    return os.path.join(BASE_DIR, path)

# ====================================================================================
# LOAD ARTIFACTS
# ====================================================================================

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

# ====================================================================================
# CUSTOM FINTECH MINIMALIST UI
# ====================================================================================
st.set_page_config(page_title="Credit Risk EWS", layout="wide")

st.markdown("""
<style>

    .main {
        background-color: #0f172a;
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
        color: white;
    }

    .result-card {
        background-color: #1e293b;
        padding: 18px;
        border-radius: 12px;
        margin-top: 12px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.25);
    }

    h2, h3, h4 {
        color: #60a5fa;
    }

    .stSlider > div > div > div > div {
        background: #3b82f6;
    }

</style>
""", unsafe_allow_html=True)

# ====================================================================================
# MANUAL PREPROCESSING
# ====================================================================================
def preprocess_single(input_dict):
    df = pd.DataFrame([input_dict])

    for col in NUMERIC + BINARY + CATEG:
        if col not in df.columns:
            df[col] = 0

    # Numeric
    X_num = df[NUMERIC].astype(float).values
    X_num = (X_num - SCALER_MEAN) / SCALER_SCALE

    # Binary
    X_bin = df[BINARY].astype(float).values

    # OHE categories
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

# ====================================================================================
# STRESS TEST FUNCTION
# ====================================================================================
def apply_stress(pd_pred, mult):
    return float(np.clip(pd_pred * mult, 0, 1))

# ====================================================================================
# LANGUAGE PACK
# ====================================================================================
LANG = {
    "EN": {
        "title": "Enterprise Credit Risk Early Warning System",
        "subtitle": "Predict PD, LGD, Expected Loss with SHAP Explainability & Stress Test",

        "inputs": "Borrower Information",
        "predict_btn": "Run Prediction",

        "term": "Loan Term (months)",
        "noemp": "Number of Employees",
        "loan": "Loan Amount (approx.)",

        "newbiz": "Is this a new business?",
        "lowdoc": "Documentation Quality",
        "urban": "Business Location",
        "naics": "Business Sector (NAICS)",
        "fiscal": "Fiscal Year",

        "stress_label": "Stress Test Multiplier",
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

        "term": "Tenor Pinjaman (bulan)",
        "noemp": "Jumlah Karyawan",
        "loan": "Jumlah Pinjaman (perkiraan)",

        "newbiz": "Apakah bisnis baru?",
        "lowdoc": "Kualitas Dokumen",
        "urban": "Lokasi Bisnis",
        "naics": "Sektor Bisnis (NAICS)",
        "fiscal": "Tahun Fiskal",

        "stress_label": "Multiplier Stress Test",
        "pd_result": "Probabilitas Gagal Bayar (PD)",
        "lgd_result": "Loss Given Default (LGD)",
        "el_result": "Expected Loss (EL)",
        "stress_section": "Skenario Stress",
        "shap_title": "Penjelasan SHAP (Model PD)"
    },

    "KR": {
        "title": "기업 신용위험 조기경보 시스템",
        "subtitle": "PD, LGD, Expected Loss 예측 · SHAP 설명 · 스트레스 테스트",

        "inputs": "차입자 정보 입력",
        "predict_btn": "예측 실행",

        "term": "대출 기간 (개월)",
        "noemp": "직원 수",
        "loan": "대출 금액 (대략적)",

        "newbiz": "신규 사업체 여부",
        "lowdoc": "서류 품질",
        "urban": "지역 구분",
        "naics": "산업 분야 (NAICS)",
        "fiscal": "회계연도",

        "stress_label": "스트레스 배수",
        "pd_result": "부도확률 (PD)",
        "lgd_result": "손실률 (LGD)",
        "el_result": "예상손실 (EL)",
        "stress_section": "스트레스 시나리오",
        "shap_title": "SHAP 설명 (PD 모델)"
    }
}

# ====================================================================================
# STREAMLIT UI
# ====================================================================================

lang_choice = st.sidebar.selectbox("Language / Bahasa / 언어", ["EN", "ID", "KR"])
T = LANG[lang_choice]

st.title(T["title"])
st.write(T["subtitle"])
st.write("")

# Stress slider
stress_mult = st.sidebar.slider(T["stress_label"], 0.5, 3.0, 1.0, 0.1)

# ================================
# MANUAL INPUT SECTION
# ================================
st.markdown("<h3>📌 " + T["inputs"] + "</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    Term = st.number_input(T["term"], min_value=1, value=60)
    NoEmp = st.number_input(T["noemp"], min_value=0, value=5)
    loan_amount = st.number_input(T["loan"], min_value=1000, max_value=5000000, value=200000)
    log_loan_amt = np.log(loan_amount)

with col2:
    new_business = st.selectbox(T["newbiz"], ["Yes", "No"])
    low_doc_quality = st.selectbox(T["lowdoc"], ["High Quality", "Medium", "Low"])
    urban_area = st.selectbox(T["urban"], ["Urban", "Suburban", "Rural"])

# Convert friendly labels → model values
newbiz_val = 1 if new_business == "Yes" else 0
lowdoc_val = 1 if low_doc_quality == "Low" else 0
urban_val = 1 if urban_area == "Urban" else 0

NAICS_2 = st.selectbox(T["naics"], OHE_CATEGORIES["NAICS_2"])
ApprovalFY = st.selectbox(T["fiscal"], OHE_CATEGORIES["ApprovalFY"])

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
            "new_business": newbiz_val,
            "low_doc": lowdoc_val,
            "urban_flag": urban_val,
            "NAICS_2": NAICS_2,
            "ApprovalFY": ApprovalFY
        }

        X = preprocess_single(input_data)

        pd_pred = float(pd_model.predict_proba(X)[0][1])
        lgd_pred = float(lgd_model.predict(X)[0])
        el_pred = pd_pred * lgd_pred

        stressed_pd = apply_stress(pd_pred, stress_mult)
        stressed_el = stressed_pd * lgd_pred

        # OUTPUT CARDS
        st.markdown(f"""
        <div class="result-card">
            <h4>{T["pd_result"]}</h4>
            <p style='font-size:24px; font-weight:700; color:#93c5fd;'>{round(pd_pred,4)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <h4>{T["lgd_result"]}</h4>
            <p style='font-size:24px; font-weight:700; color:#93c5fd;'>{round(lgd_pred,4)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <h4>{T["el_result"]}</h4>
            <p style='font-size:24px; font-weight:700; color:#93c5fd;'>{round(el_pred,4)}</p>
        </div>
        """, unsafe_allow_html=True)

        # Stress Results
        st.markdown(f"""
        <div class="result-card">
            <h4>{T["stress_section"]}</h4>
            <p>Stressed PD: <b>{round(stressed_pd,4)}</b></p>
            <p>Stressed Expected Loss: <b>{round(stressed_el,4)}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # SHAP Explainability
        st.markdown("<h3>🔍 " + T["shap_title"] + "</h3>", unsafe_allow_html=True)

        shap_values = shap_explainer.shap_values(X)

        tab1, tab2 = st.tabs(["Summary Plot", "Top Features"])

        with tab1:
            fig = shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig)

        with tab2:
            shap_df = pd.DataFrame({
                "Feature": NUMERIC + BINARY + CATEG,
                "Impact": np.abs(shap_values).mean(axis=0)
            }).sort_values("Impact", ascending=False)

            st.dataframe(shap_df.head(10))
