import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sklearn
import xgboost

# ======================================================================================
# 0. DIRECTORIES
# ======================================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================================================
# 1. PAGE SETUP
# ======================================================================================
st.set_page_config(
    page_title="Enterprise Credit Risk EWS",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================================================
# 2. LANGUAGE DICTIONARY
# ======================================================================================
LANG = {
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Analisis PD & LGD • Stress Testing · Pipeline AI",
        "upload_header": "Unggah Data Portofolio",
        "upload_label": "Unggah CSV atau Excel",
        "sidebar_model": "Status Model",
        "model_missing": "❌ Model tidak ditemukan. Pastikan file .pkl tersedia.",
        "success_load": "✅ Model berhasil dimuat!",
        "run_analysis": "🚀 Jalankan Analisis AI",
        "demo_data": "Buat Data Demo",
        "tab1": "📊 Laporan Portofolio",
        "tab2": "⚡ Stress Testing & Inspektor",
        "col_pd": "Probabilitas Gagal Bayar (PD)",
        "col_lgd": "Loss Given Default (LGD)",
        "col_el": "Expected Loss (EL)",
        "stress_vix": "Skenario Krisis (VIX Index)",
        "stress_desc": "Geser slider untuk mensimulasikan volatilitas pasar.",
        "diagnostic": "🩺 Mode Diagnostik"
    },
    "EN": {
        "title": "Enterprise Credit Risk EWS",
        "subtitle": "PD & LGD Analysis • Stress Testing · AI Pipeline",
        "upload_header": "Upload Portfolio Data",
        "upload_label": "Upload CSV or Excel",
        "sidebar_model": "Model Status",
        "model_missing": "❌ Model not found. Ensure .pkl files exist.",
        "success_load": "✅ Models loaded successfully!",
        "run_analysis": "🚀 Run AI Analysis",
        "demo_data": "Generate Demo Data",
        "tab1": "📊 Portfolio Report",
        "tab2": "⚡ Stress Testing & Inspector",
        "col_pd": "Probability of Default (PD)",
        "col_lgd": "Loss Given Default (LGD)",
        "col_el": "Expected Loss (EL)",
        "stress_vix": "Crisis Scenario (VIX Index)",
        "stress_desc": "Slide to simulate market volatility.",
        "diagnostic": "🩺 Diagnostic Mode"
    }
}

# ======================================================================================
# 3. SIDEBAR LANGUAGE SELECTOR
# ======================================================================================
with st.sidebar:
    st.header("🌐 Language")
    lang = st.selectbox("Select Language", ["ID", "EN"])
txt = LANG[lang]

# ======================================================================================
# 4. MODEL LOADER (SMART & ROBUST)
# ======================================================================================
MODEL_FILES = {
    "pd": "pd_model_pipeline.pkl",
    "lgd": "lgd_model_pipeline.pkl"
}

@st.cache_resource
def load_pipeline_model(filename):
    """Loads a model safely with full-path handling & error catching."""
    path = os.path.join(APP_DIR, filename)

    if not os.path.exists(path):
        return None, f"❌ File '{filename}' tidak ditemukan di {path}"

    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, f"❌ Gagal load model {filename}: {e}"

pd_model, pd_err = load_pipeline_model(MODEL_FILES["pd"])
lgd_model, lgd_err = load_pipeline_model(MODEL_FILES["lgd"])

# ======================================================================================
# 5. SIDEBAR MODEL STATUS + DEPENDENCY CHECK
# ======================================================================================
with st.sidebar:
    st.subheader(txt["sidebar_model"])

    if pd_model is None or lgd_model is None:
        st.error(txt["model_missing"])
        if pd_err: st.write(pd_err)
        if lgd_err: st.write(lgd_err)
    else:
        st.success(txt["success_load"])

    # Dependency check
    st.markdown("### 🔍 Dependency Check")
    st.write(f"• scikit-learn runtime: **{sklearn.__version__}**")
    st.write(f"• xgboost runtime: **{xgboost.__version__}**")

    if sklearn.__version__ != "1.6.1":
        st.warning("⚠️ scikit-learn mismatch! Trained on 1.6.1")
    if xgboost.__version__ != "3.1.2":
        st.warning("⚠️ xgboost mismatch! Trained on 3.1.2")

# ======================================================================================
# 6. DIAGNOSTIC MODE
# ======================================================================================
with st.expander(txt["diagnostic"]):
    st.write("### Model Pipeline Parameters")
    if pd_model:
        st.json(pd_model.get_params(deep=False))
    if lgd_model:
        st.json(lgd_model.get_params(deep=False))

    st.write("### Environment Info")
    st.write({
        "Python": os.sys.version,
        "sklearn version": sklearn.__version__,
        "xgboost version": xgboost.__version__,
        "APP_DIR": APP_DIR,
        "Files in APP_DIR": os.listdir(APP_DIR)
    })

# ======================================================================================
# 7. DEMO DATA GENERATOR
# ======================================================================================
def generate_demo():
    return pd.DataFrame({
        "Name": ["A", "B", "C", "D"],
        "NAICS": ["541512", "331110", "448130", "722511"],
        "ApprovalFY": [2010, 2015, 2018, 2020],
        "Term": [36, 60, 48, 72],
        "NoEmp": [12, 50, 7, 30],
        "NewExist": [1, 2, 1, 1],
        "UrbanRural": [1, 2, 1, 1],
        "LowDoc": ["N", "Y", "N", "N"],
        "DisbursementGross": [150e6, 450e6, 220e6, 300e6]
    })

# ======================================================================================
# 8. MAIN UI
# ======================================================================================
st.title(txt["title"])
st.markdown(f"**{txt['subtitle']}**")
st.header("1. " + txt["upload_header"])

uploaded = st.file_uploader(txt["upload_label"], type=["csv", "xlsx"])

if uploaded:
    try:
        df_raw = (
            pd.read_csv(uploaded)
            if uploaded.name.endswith(".csv")
            else pd.read_excel(uploaded)
        )
        st.session_state["df"] = df_raw
    except Exception as e:
        st.error(f"❌ Error membaca file: {e}")
else:
    if st.button(txt["demo_data"]):
        st.session_state["df"] = generate_demo()

# ======================================================================================
# 9. RUN ANALYSIS
# ======================================================================================
if "df" in st.session_state and pd_model is not None:
    df = st.session_state["df"]
    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button(txt["run_analysis"]):
        try:
            pd_pred = pd_model.predict_proba(df)[:, 1]
            lgd_pred = np.clip(lgd_model.predict(df), 0, 1)

            df_res = df.copy()
            df_res[txt["col_pd"]] = pd_pred
            df_res[txt["col_lgd"]] = lgd_pred
            df_res[txt["col_el"]] = pd_pred * lgd_pred * df["DisbursementGross"]

            st.session_state["res"] = df_res
            st.success("🎉 Analisis selesai!")
        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")
            st.info("Periksa apakah kolom input cocok dengan pipeline training.")

# ======================================================================================
# 10. RESULT VIEW
# ======================================================================================
if "res" in st.session_state:
    res = st.session_state["res"]
    tab1, tab2 = st.tabs([txt["tab1"], txt["tab2"]])

    with tab1:
        st.subheader("Stress Testing")
        vix = st.slider(txt["stress_vix"], 10, 80, 20)

        stressed_pd = np.minimum(res[txt["col_pd"]] * (1 + (vix - 20) / 100 * 1.5), 1)
        stressed_el = stressed_pd * res[txt["col_lgd"]] * res["DisbursementGross"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline EL", f"{res[txt['col_el']].sum():,.0f}")
        col2.metric("Stressed EL", f"{stressed_el.sum():,.0f}")
        col3.metric("Impact", f"{stressed_el.sum() - res[txt['col_el']].sum():,.0f}")

        st.dataframe(res)

    with tab2:
        st.subheader("Debtor Inspector")
        name = st.selectbox("Select Debtor", res["Name"].unique())
        d = res[res["Name"] == name].iloc[0]

        colA, colB = st.columns(2)
        colA.metric("PD", f"{d[txt['col_pd']]:.2%}")
        colB.metric("LGD", f"{d[txt['col_lgd']]:.2%}")
