import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ===================================================================
# 1. PAGE SETUP
# ===================================================================
st.set_page_config(
    page_title="Enterprise Credit Risk EWS",
    page_icon="🛡️",
    layout="wide"
)

# ===================================================================
# 2. LANGUAGE DICTIONARY
# ===================================================================
LANG_DICT = {
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Analisis Probabilitas Gagal Bayar (PD) & Stress Testing Ekonomi Makro",
        "upload_header": "Unggah Data Portofolio",
        "upload_label": "Unggah file CSV atau Excel (Data Peminjam)",
        "sidebar_model": "Status Model",
        "model_missing": "Model tidak ditemukan. Pastikan file .pkl tersedia.",
        "success_load": "Model berhasil dimuat!",
        "run_analysis": "🚀 Jalankan Analisis AI",
        "demo_data": "Buat Data Demo (Simulasi)",
        "tab1": "📋 Laporan Portofolio",
        "tab2": "⚡ Inspektor & Stress Test",
        "col_pd": "Probabilitas Gagal Bayar (PD)",
        "col_lgd": "Loss Given Default (LGD)",
        "col_el": "Expected Loss (EL)",
        "stress_vix": "Skenario Krisis (VIX Index)",
        "stress_desc": "Geser slider untuk mensimulasikan volatilitas pasar.",
        "portfolio_summary": "Ringkasan Portofolio",
        "total_el": "Total Expected Loss (Baseline)",
        "avg_pd": "Rata-rata PD",
        "avg_lgd": "Rata-rata LGD",
        "total_el_stressed": "Total Expected Loss (Stressed)",
        "el_impact": "Dampak Krisis (EL naik)",
        "select_debtor": "Pilih Debitur",
        "debtor_pd_lgd": "PD & LGD Debitur"
    },
    "EN": {
        "title": "Enterprise Credit Risk EWS",
        "subtitle": "Probability of Default (PD) & Macro Stress Testing",
        "upload_header": "Upload Portfolio Data",
        "upload_label": "Upload CSV or Excel",
        "sidebar_model": "Model Status",
        "model_missing": "Model not found. Make sure .pkl files exist.",
        "success_load": "Models loaded successfully!",
        "run_analysis": "🚀 Run AI Analysis",
        "demo_data": "Generate Demo Data",
        "tab1": "📋 Portfolio Report",
        "tab2": "⚡ Inspector & Stress Test",
        "col_pd": "Probability of Default (PD)",
        "col_lgd": "Loss Given Default (LGD)",
        "col_el": "Expected Loss (EL)",
        "stress_vix": "Crisis Scenario (VIX Index)",
        "stress_desc": "Slide to simulate market volatility.",
        "portfolio_summary": "Portfolio Summary",
        "total_el": "Total Expected Loss (Baseline)",
        "avg_pd": "Average PD",
        "avg_lgd": "Average LGD",
        "total_el_stressed": "Total Expected Loss (Stressed)",
        "el_impact": "Crisis Impact (EL Increase)",
        "select_debtor": "Select Debtor",
        "debtor_pd_lgd": "Debtor PD & LGD"
    }
}

# ===================================================================
# 3. SIDEBAR & LANGUAGE SELECTOR
# ===================================================================
with st.sidebar:
    st.header("🌐 Language / Bahasa")
    lang_opt = st.selectbox("Select Language", ["ID", "EN"])
    txt = LANG_DICT[lang_opt]
    st.markdown("---")

# ===================================================================
# 4. LOAD PIPELINE MODELS
# ===================================================================
@st.cache_resource
def load_models():
    try:
        pd_model = joblib.load("pd_model_pipeline.pkl")
        lgd_model = joblib.load("lgd_model_pipeline.pkl")
        return pd_model, lgd_model
    except Exception as e:
        st.error(f"{txt['model_missing']} | Detail: {e}")
        return None, None

pd_model, lgd_model = load_models()

with st.sidebar:
    st.subheader(txt["sidebar_model"])
    if pd_model is None:
        st.error(txt["model_missing"])
    else:
        st.success(txt["success_load"])

# ===================================================================
# 5. GENERATE DEMO DATA
# ===================================================================
def generate_demo_data():
    return pd.DataFrame({
        "Name": ["Debitur A", "Debitur B", "Debitur C", "Debitur D"],
        "NAICS": ["541512", "331110", "448130", "722511"],
        "ApprovalFY": [2010, 2015, 2018, 2020],
        "Term": [36, 60, 48, 72],
        "NoEmp": [12, 50, 7, 30],
        "NewExist": [1, 2, 1, 1],
        "UrbanRural": [1, 2, 1, 1],
        "LowDoc": ["N", "Y", "N", "N"],
        "DisbursementGross": [150_000_000, 450_000_000, 220_000_000, 300_000_000]
    })

# ===================================================================
# 6. PAGE TITLE
# ===================================================================
st.title(txt["title"])
st.markdown(f"**{txt['subtitle']}**")

st.header("1. " + txt["upload_header"])

# ===================================================================
# 7. FILE UPLOADER
# ===================================================================
uploaded_file = st.file_uploader(txt["upload_label"], type=["csv", "xlsx"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.session_state["df_raw"] = df_raw

else:
    if st.button(txt["demo_data"]):
        st.session_state["df_raw"] = generate_demo_data()

# ===================================================================
# 8. RUN ANALYSIS
# ===================================================================
if "df_raw" in st.session_state and pd_model is not None:

    df_raw = st.session_state["df_raw"]
    st.subheader("Preview Data Input")
    st.dataframe(df_raw.head())

    if st.button(txt["run_analysis"]):

        try:
            # Pipeline already handles preprocessing
            pd_pred = pd_model.predict_proba(df_raw)[:, 1]
            lgd_pred = lgd_model.predict(df_raw)
            lgd_pred = np.clip(lgd_pred, 0, 1)

            df_results = df_raw.copy()
            df_results[txt["col_pd"]] = pd_pred
            df_results[txt["col_lgd"]] = lgd_pred
            df_results[txt["col_el"]] = (
                df_results[txt["col_pd"]] * df_results[txt["col_lgd"]] * df_results["DisbursementGross"]
            )

            st.session_state["results"] = df_results
            st.success("Analisis selesai!")

        except Exception as e:
            st.error(f"Kesalahan Prediksi Model: {e}")
            st.warning("Pastikan format kolom input sesuai training model.")

# ===================================================================
# 9. DISPLAY RESULTS
# ===================================================================
if "results" in st.session_state:

    results = st.session_state["results"]

    tab1, tab2 = st.tabs([txt["tab1"], txt["tab2"]])

    # -------------------------
    # TAB 1 — Portfolio Summary
    # -------------------------
    with tab1:

        st.subheader(txt["stress_vix"])
        vix = st.slider(txt["stress_vix"], 10, 80, 20)
        st.markdown(f"*{txt['stress_desc']}*")

        # Stress PD
        stressed_pd = np.minimum(results[txt["col_pd"]] * (1 + (vix - 20) / 100 * 1.5), 1)
        stressed_el = stressed_pd * results[txt["col_lgd"]] * results["DisbursementGross"]

        total_el = results[txt["col_el"]].sum()
        total_el_stressed = stressed_el.sum()

        st.subheader(txt["portfolio_summary"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(txt["total_el"], f"{total_el:,.0f}")
        col2.metric(txt["avg_pd"], f"{results[txt['col_pd']].mean():.2%}")
        col3.metric(txt["total_el_stressed"], f"{total_el_stressed:,.0f}")
        col4.metric(txt["el_impact"], f"{(total_el_stressed - total_el):,.0f}")

        st.dataframe(results)

    # -------------------------
    # TAB 2 — Inspector & Stress Test
    # -------------------------
    with tab2:

        st.subheader(txt["select_debtor"])
        name = st.selectbox("Debitur", results["Name"].unique())

        d = results[results["Name"] == name].iloc[0]

        st.write("**PD & LGD**")
        colA, colB = st.columns(2)
        colA.metric("PD", f"{d[txt['col_pd']]:.2%}")
        colB.metric("LGD", f"{d[txt['col_lgd']]:.2%}")

