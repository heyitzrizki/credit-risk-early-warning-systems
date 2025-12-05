import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import os

# ==============================
# LOAD ARTIFACTS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
PD_MODEL_PATH = os.path.join(BASE_DIR, "PD_model.pkl")
LGD_MODEL_PATH = os.path.join(BASE_DIR, "LGD_model.pkl")
SHAP_PD_EXPLAINER_PATH = os.path.join(BASE_DIR, "pd_shap_explainer.pkl")

preprocessor = joblib.load(PREPROCESSOR_PATH)
pd_model = joblib.load(PD_MODEL_PATH)
lgd_model = joblib.load(LGD_MODEL_PATH)
shap_explainer = joblib.load(SHAP_PD_EXPLAINER_PATH)

# ==============================
# UI CONFIG
# ==============================
st.set_page_config(page_title="Credit Risk Early Warning System", layout="wide")

LANG = st.sidebar.selectbox("Language / Bahasa / 언어", ["EN", "ID", "KR"])

# ==============================
# TEXT DICTIONARY
# ==============================
TXT = {
    "EN": {
        "title": "Enterprise Credit Risk Early Warning System",
        "subtitle": "Predict PD, LGD, Expected Loss, with SHAP Explainability & Stress Test",
        "upload": "Upload Portfolio File (CSV/XLSX)",
        "run": "Run Analysis",
        "pd": "Probability of Default (PD)",
        "lgd": "Loss Given Default (LGD)",
        "el": "Expected Loss (EL)",
        "stress": "Stress Test (Market Shock)",
        "shap_title": "SHAP Feature Impact (PD)",
    },
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Prediksi PD, LGD, Expected Loss, dengan SHAP & Stress Test",
        "upload": "Unggah File Portofolio (CSV/XLSX)",
        "run": "Jalankan Analisis",
        "pd": "Probabilitas Gagal Bayar (PD)",
        "lgd": "Loss Given Default (LGD)",
        "el": "Expected Loss (EL)",
        "stress": "Stress Test (Guncangan Pasar)",
        "shap_title": "Dampak Fitur (SHAP) untuk PD",
    },
    "KR": {
        "title": "기업 신용 리스크 조기 경보 시스템",
        "subtitle": "PD, LGD, 예상 손실 예측 및 SHAP 설명과 스트레스 테스트",
        "upload": "포트폴리오 파일 업로드 (CSV/XLSX)",
        "run": "분석 실행",
        "pd": "부도확률 (PD)",
        "lgd": "부도시 손실률 (LGD)",
        "el": "예상 손실 (EL)",
        "stress": "스트레스 테스트 (시장 충격)",
        "shap_title": "SHAP 특성 영향 (PD)",
    }
}
T = TXT[LANG]

# ==============================
# PAGE HEADER
# ==============================
st.title(T["title"])
st.write(f"**{T['subtitle']}**")

# ==============================
# FILE UPLOAD
# ==============================
st.header(T["upload"])
file = st.file_uploader("", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
    st.write("### Preview")
    st.dataframe(df.head())

    if st.button(T["run"]):

        # ==============================
        # PREPROCESS DATA
        # ==============================
        try:
            X = preprocessor.transform(df)
        except Exception as e:
            st.error(f"Preprocessing Error: {e}")
            st.stop()

        # ==============================
        # PREDICT PD & LGD
        # ==============================
        pd_pred = pd_model.predict_proba(X)[:, 1]
        lgd_pred = lgd_model.predict(X)

        # Compute Expected Loss
        # Assume Exposure = loan amount column
        if "DisbursementGross" in df.columns:
            ead = df["DisbursementGross"].astype(float)
        else:
            ead = np.ones(len(df)) * 100000  # default

        el = pd_pred * lgd_pred * ead

        result = df.copy()
        result["PD"] = pd_pred
        result["LGD"] = lgd_pred
        result["EL"] = el

        st.write("### Results")
        st.dataframe(result[["PD", "LGD", "EL"]].head())

        # ==============================
        # STRESS TEST MODULE
        # ==============================
        st.subheader(T["stress"])
        stress_level = st.slider("Market Shock %", 0, 200, 50)
        stressed_pd = np.clip(pd_pred * (1 + stress_level/100), 0, 1)
        stressed_el = stressed_pd * lgd_pred * ead

        st.write("#### Stress Test Result Preview")
        st.dataframe(pd.DataFrame({
            "PD_baseline": pd_pred,
            "PD_stressed": stressed_pd,
            "EL_stressed": stressed_el
        }).head())

        # ==============================
        # SHAP PLOT
        # ==============================
        st.subheader(T["shap_title"])

        shap_values = shap_explainer.shap_values(X)

        st.write("#### Feature Impact Summary Plot")
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(bbox_inches="tight")

        # Optional download
        st.download_button(
            label="Download Results CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="credit_risk_results.csv",
            mime="text/csv"
        )
