import streamlit as st
import joblib
import json
import numpy as np
import shap

# ====================================================================================
# FILE PATHS
# ====================================================================================

MODEL_PD_PATH = "app/PD_model.pkl"
MODEL_LGD_PATH = "app/LGD_model.pkl"
SHAP_PATH = "app/pd_shap_explainer.pkl"
PREPROCESSOR_JSON = "app/preprocessor_meta.json"


# ====================================================================================
# LOAD ARTIFACTS
# ====================================================================================

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
def load_preprocessor_meta():
    try:
        with open(PREPROCESSOR_JSON, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Preprocessor Metadata Error: {e}")
        return None


pd_model, lgd_model, shap_explainer = load_models()
meta = load_preprocessor_meta()

# Extract metadata
NUMERIC = meta["numeric_features"]
BINARY = meta["binary_features"]
CATEG = meta["categorical_features"]

SCALER_MEAN = np.array(meta["scaler_mean"])
SCALER_SCALE = np.array(meta["scaler_scale"])
OHE_CATEGORIES = meta["ohe_categories"]


# ====================================================================================
# PREPROCESSING (MANUAL)
# ====================================================================================

def preprocess_single(input_dict):
    """
    input_dict = {
       "Term": value,
       "NoEmp": value,
       ...
    }
    """

    # Convert to array shaped like training data
    numeric_vals = np.array([input_dict[col] for col in NUMERIC], dtype=float)
    numeric_scaled = (numeric_vals - SCALER_MEAN) / SCALER_SCALE

    binary_vals = np.array([input_dict[col] for col in BINARY], dtype=float)

    # Manual One-Hot Encoding
    ohe_list = []
    for cat_col in CATEG:
        categories = OHE_CATEGORIES[cat_col]
        encoding = np.zeros(len(categories))

        raw_val = str(input_dict[cat_col])
        if raw_val in categories:
            idx = categories.index(raw_val)
            encoding[idx] = 1

        ohe_list.append(encoding)

    X = np.concatenate([numeric_scaled, binary_vals] + ohe_list)
    return X.reshape(1, -1)


# ====================================================================================
# STRESS TESTING
# ====================================================================================

def apply_stress(pd_pred, mult):
    return np.clip(pd_pred * mult, 0, 1)


# ====================================================================================
# LANGUAGE PACK
# ====================================================================================

LANG = {
    "EN": {
        "title": "Enterprise Credit Risk Early Warning System",
        "subtitle": "Predict PD, LGD, Expected Loss with SHAP Explainability & Stress Test",

        "stress_label": "Stress Test Multiplier",
        "form_title": "Borrower Information Input",

        "btn_predict": "Run Prediction",
        "pd_label": "Predicted Probability of Default (PD)",
        "lgd_label": "Predicted Loss Given Default (LGD)",
        "el_label": "Expected Loss (EL)",
        "shap_title": "SHAP Explainability (PD Model)",

        "term": "Loan Term (months)",
        "noemp": "Number of Employees",
        "loanamt": "Loan Amount",
        "newbiz": "New Business (0/1)",
        "lowdoc": "Low Documentation (0/1)",
        "urban": "Urban Flag (0/1)",
        "fy": "Approval Fiscal Year",
        "naics": "NAICS 2-Digit Code",
    },
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Prediksi PD, LGD, Expected Loss dengan SHAP & Stress Test",

        "stress_label": "Multiplier Stress Test",
        "form_title": "Input Informasi Peminjam",

        "btn_predict": "Jalankan Prediksi",
        "pd_label": "Probabilitas Gagal Bayar (PD)",
        "lgd_label": "Loss Given Default (LGD)",
        "el_label": "Expected Loss (EL)",
        "shap_title": "Penjelasan SHAP (Model PD)",

        "term": "Jangka Waktu Kredit (bulan)",
        "noemp": "Jumlah Karyawan",
        "loanamt": "Jumlah Pinjaman",
        "newbiz": "Bisnis Baru (0/1)",
        "lowdoc": "Dokumen Minimum (0/1)",
        "urban": "Daerah Urban (0/1)",
        "fy": "Tahun Persetujuan Kredit",
        "naics": "Kode NAICS 2-Digit",
    },
    "KR": {
        "title": "기업 신용위험 조기경보 시스템",
        "subtitle": "PD, LGD, Expected Loss 예측 · SHAP 설명 · 스트레스 테스트",

        "stress_label": "스트레스 테스트 배수",
        "form_title": "차입자 정보 입력",

        "btn_predict": "예측 실행",
        "pd_label": "부도확률 (PD)",
        "lgd_label": "손실률 (LGD)",
        "el_label": "예상손실 (EL)",
        "shap_title": "SHAP 설명 (PD 모델)",

        "term": "대출 기간 (개월)",
        "noemp": "직원 수",
        "loanamt": "대출 금액",
        "newbiz": "신규 사업 여부 (0/1)",
        "lowdoc": "간편 서류 여부 (0/1)",
        "urban": "도시 지역 여부 (0/1)",
        "fy": "승인 연도",
        "naics": "NAICS 2자리 코드",
    }
}


# ====================================================================================
# STREAMLIT UI
# ====================================================================================

st.set_page_config(layout="wide", page_title="Credit Risk EWS")

lang_choice = st.sidebar.selectbox("Language / Bahasa / 언어", ["EN", "ID", "KR"])
T = LANG[lang_choice]

st.title(T["title"])
st.write(T["subtitle"])
st.write("")

stress_mult = st.sidebar.slider(T["stress_label"], 0.5, 3.0, 1.0, 0.1)

st.subheader(T["form_title"])

# ====================================================================================
# INPUT FORM
# ====================================================================================

with st.form("borrower_form"):

    Term = st.number_input(T["term"], min_value=1, step=1)
    NoEmp = st.number_input(T["noemp"], min_value=0, step=1)
    LoanAmt = st.number_input(T["loanamt"], min_value=1, step=100)

    new_business = st.selectbox(T["newbiz"], [0, 1])
    low_doc = st.selectbox(T["lowdoc"], [0, 1])
    urban_flag = st.selectbox(T["urban"], [0, 1])

    ApprovalFY = st.selectbox(T["fy"], OHE_CATEGORIES["ApprovalFY"])
    NAICS_2 = st.selectbox(T["naics"], OHE_CATEGORIES["NAICS_2"])

    submitted = st.form_submit_button(T["btn_predict"])

# ====================================================================================
# RUN PREDICTION
# ====================================================================================

if submitted:

    if pd_model is None:
        st.error("Models failed to load.")
        st.stop()

    input_dict = {
        "Term": Term,
        "NoEmp": NoEmp,
        "log_loan_amt": np.log1p(LoanAmt),
        "new_business": new_business,
        "low_doc": low_doc,
        "urban_flag": urban_flag,
        "ApprovalFY": str(ApprovalFY),
        "NAICS_2": str(NAICS_2),
    }

    X = preprocess_single(input_dict)

    pd_pred = pd_model.predict_proba(X)[0, 1]
    lgd_pred = lgd_model.predict(X)[0]
    el_pred = pd_pred * lgd_pred

    stressed_pd = apply_stress(pd_pred, stress_mult)
    stressed_el = stressed_pd * lgd_pred

    st.subheader(T["pd_label"])
    st.write(pd_pred)

    st.subheader(T["lgd_label"])
    st.write(lgd_pred)

    st.subheader(T["el_label"])
    st.write(el_pred)

    st.write("### Stress Test")
    st.write("Stressed PD:", stressed_pd)
    st.write("Stressed Expected Loss:", stressed_el)

    # SHAP plot
    st.subheader(T["shap_title"])
    shap_vals = shap_explainer.shap_values(X)
    st.set_option("deprecation.showPyplotGlobalUse", False)
    st.pyplot(shap.summary_plot(shap_vals, X))
