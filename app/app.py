import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap

# ====================================================================================
# LOAD ARTIFACTS
# ====================================================================================

@st.cache_resource
def load_models():
    try:
        pd_model = joblib.load("PD_model_pipeline.pkl")
        lgd_model = joblib.load("LGD_model_pipeline.pkl")
        shap_explainer = joblib.load("pd_shap_explainer.pkl")
        return pd_model, lgd_model, shap_explainer
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None, None, None

@st.cache_resource
def load_preprocessor_meta():
    try:
        with open("preprocessor_meta.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Preprocessor Metadata Error: {e}")
        return None

pd_model, lgd_model, shap_explainer = load_models()
meta = load_preprocessor_meta()

if meta is not None:
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
# MANUAL PREPROCESSING
# ====================================================================================

def preprocess(df_input: pd.DataFrame) -> np.ndarray:
    df = df_input.copy()

    # Ensure all required columns exist
    for col in NUMERIC + BINARY + CATEG:
        if col not in df.columns:
            df[col] = 0

    # 1) Scale numeric features
    X_num = df[NUMERIC].astype(float).values
    X_num = (X_num - SCALER_MEAN) / SCALER_SCALE

    # 2) Binary passthrough
    X_bin = df[BINARY].astype(float).values

    # 3) Manual OHE
    ohe_arrays = []
    for cat_col in CATEG:
        categories = OHE_CATEGORIES[cat_col]
        mapping = {str(v): i for i, v in enumerate(categories)}

        encoded = np.zeros((len(df), len(categories)))
        for r, raw_val in enumerate(df[cat_col]):
            key = str(raw_val)
            if key in mapping:
                encoded[r, mapping[key]] = 1
        ohe_arrays.append(encoded)

    X_cat = np.concatenate(ohe_arrays, axis=1)
    X_all = np.concatenate([X_num, X_bin, X_cat], axis=1)
    return X_all


# ====================================================================================
# STRESS TESTING FUNCTION
# ====================================================================================

def apply_stress(pd_pred, multiplier):
    stressed_pd = (pd_pred * multiplier).clip(0, 1)
    return stressed_pd


# ====================================================================================
# UI LANGUAGE PACK
# ====================================================================================

LANG = {
    "EN": {
        "title": "Enterprise Credit Risk Early Warning System",
        "subtitle": "Predict PD, LGD, Expected Loss with SHAP Explainability & Stress Test",
        "upload_title": "Upload Borrower Data (CSV)",
        "predict_btn": "Run Prediction",
        "stress_label": "Stress Test Multiplier",
        "pd_result": "Predicted Probability of Default (PD)",
        "lgd_result": "Predicted Loss Given Default (LGD)",
        "el_result": "Expected Loss (EL)",
        "shap_title": "SHAP Explainability (PD Model)"
    },
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Prediksi PD, LGD, Expected Loss dengan SHAP & Stress Test",
        "upload_title": "Unggah Data Peminjam (CSV)",
        "predict_btn": "Jalankan Prediksi",
        "stress_label": "Multiplier Stress Test",
        "pd_result": "Probabilitas Gagal Bayar (PD)",
        "lgd_result": "Loss Given Default (LGD)",
        "el_result": "Expected Loss (EL)",
        "shap_title": "Penjelasan SHAP (Model PD)"
    },
    "KR": {
        "title": "기업 신용위험 조기경보 시스템",
        "subtitle": "PD, LGD, Expected Loss 예측 및 SHAP 설명 · 스트레스 테스트 제공",
        "upload_title": "차입자 데이터 업로드 (CSV)",
        "predict_btn": "예측 실행",
        "stress_label": "스트레스 테스트 배수",
        "pd_result": "부도확률 (PD)",
        "lgd_result": "손실률 (LGD)",
        "el_result": "예상손실 (EL)",
        "shap_title": "SHAP 설명 (PD 모델)"
    }
}

# ====================================================================================
# STREAMLIT UI
# ====================================================================================

st.set_page_config(page_title="Credit Risk EWS", layout="wide")

# Sidebar language
lang_choice = st.sidebar.selectbox("Language / Bahasa / 언어", ["EN", "ID", "KR"])
T = LANG[lang_choice]

st.title(T["title"])
st.write(T["subtitle"])
st.write("")

uploaded = st.file_uploader(T["upload_title"], type=["csv"])

stress_mult = st.sidebar.slider(T["stress_label"], 0.5, 3.0, 1.0, 0.1)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    if st.button(T["predict_btn"]):
        if pd_model is None:
            st.error("Models not loaded.")
        else:
            X = preprocess(df)

            pd_pred = pd_model.predict_proba(X)[:, 1]
            lgd_pred = lgd_model.predict(X)
            el_pred = pd_pred * lgd_pred

            stressed_pd = apply_stress(pd_pred, stress_mult)
            stressed_el = stressed_pd * lgd_pred

            st.subheader(T["pd_result"])
            st.write(pd_pred)

            st.subheader(T["lgd_result"])
            st.write(lgd_pred)

            st.subheader(T["el_result"])
            st.write(el_pred)

            st.write("### Stress Test Result (PD & EL)")
            st.write("Stressed PD:", stressed_pd)
            st.write("Stressed Expected Loss:", stressed_el)

            # SHAP PLOT
            st.subheader(T["shap_title"])
            shap_values = shap_explainer.shap_values(X)
            st.set_option("deprecation.showPyplotGlobalUse", False)
            st.pyplot(shap.summary_plot(shap_values, X))


else:
    st.info("Please upload a CSV file to begin.")

