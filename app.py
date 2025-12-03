import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

st.set_page_config(page_title="Enterprise Credit Risk EWS", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

lang_dict = {
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Analisis Probabilitas Gagal Bayar & Stress Testing Ekonomi Makro",
        "upload_header": "Unggah Data Portofolio",
        "upload_label": "Unggah file CSV atau Excel (Data Peminjam)",
        "sidebar_settings": "Pengaturan",
        "sidebar_model": "Status Model",
        "tab1": "📋 Laporan Portofolio",
        "tab2": "⚡ Inspektor & Stress Test",
        "col_pd": "Probabilitas Default (PD)",
        "col_lgd": "Loss Given Default (LGD)",
        "col_el": "Expected Loss (EL)",
        "stress_vix": "Skenario Krisis (VIX Index)",
        "stress_desc": "Geser untuk mensimulasikan volatilitas pasar. VIX > 30 menandakan ketidakpastian tinggi.",
        "download_btn": "Unduh Hasil Analisis",
        "model_missing": "Model tidak ditemukan. Pastikan file .pkl ada di folder yang sama.",
        "success_load": "Model berhasil dimuat!",
        "risk_profile": "Profil Risiko Debitur",
        "base_vs_stress": "Perbandingan: Normal vs Krisis",
    },
    "EN": {
        "title": "Enterprise Credit Risk EWS",
        "subtitle": "Probability of Default Analysis & Macro Stress Testing",
        "upload_header": "Upload Portfolio Data",
        "upload_label": "Upload CSV or Excel file (Borrower Data)",
        "sidebar_settings": "Settings",
        "sidebar_model": "Model Status",
        "tab1": "📋 Portfolio Report",
        "tab2": "⚡ Inspector & Stress Test",
        "col_pd": "Probability of Default (PD)",
        "col_lgd": "Loss Given Default (LGD)",
        "col_el": "Expected Loss (EL)",
        "stress_vix": "Crisis Scenario (VIX Index)",
        "stress_desc": "Slide to simulate market volatility. VIX > 30 indicates high uncertainty.",
        "download_btn": "Download Analysis Results",
        "model_missing": "Models not found. Ensure .pkl files exist.",
        "success_load": "Models loaded successfully!",
        "risk_profile": "Debtor Risk Profile",
        "base_vs_stress": "Comparison: Baseline vs Stressed",
    }
}

with st.sidebar:
    st.header("🌐 Language / Bahasa")
    lang_opt = st.selectbox("Select Language", ["ID", "EN"])
    txt = lang_dict[lang_opt]

@st.cache_resource
def load_models_smart():
    models = {}
    base_dir = os.path.dirname(__file__)
    pd_path = os.path.join(base_dir, "PD_model_tuned_pipeline.pkl")
    lgd_path = os.path.join(base_dir, "LGD_model_pipeline.pkl")

    try:
        if os.path.exists(pd_path):
            models['PD'] = joblib.load(pd_path)
            models['PD_Name'] = pd_path
        if os.path.exists(lgd_path):
            models['LGD'] = joblib.load(lgd_path)
            models['LGD_Name'] = lgd_path
        if 'PD' in models and 'LGD' in models:
            return models
        return None
    except Exception:
        return None

st.sidebar.subheader(f"🛠️ {txt['sidebar_model']}")
models = load_models_smart()

if not models:
    st.sidebar.error(txt['model_missing'])
uploaded_pd = st.sidebar.file_uploader("Upload PD Model (.pkl)", type="pkl")
uploaded_lgd = st.sidebar.file_uploader("Upload LGD Model (.pkl)", type="pkl")
if uploaded_pd and uploaded_lgd:
    models = {}
    models['PD'] = joblib.load(uploaded_pd)
    models['LGD'] = joblib.load(uploaded_lgd)
    st.sidebar.success(txt['success_load'])

st.title(txt['title'])
st.markdown(f"**{txt['subtitle']}**")
st.header(f"1. {txt['upload_header']}")

uploaded_file = st.file_uploader(txt['upload_label'], type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.session_state['df'] = df
else:
    if st.button("Generate Demo Data (Simulasi)"):
        st.session_state['df'] = pd.DataFrame({
            'Name': ['ABC', 'DEF'],
            'NAICS': ['33', '44'],
            'Term': [12, 24],
            'NoEmp': [10, 20],
            'NewExist': [1, 2],
            'UrbanRural': [1, 2],
            'LowDoc': ['N', 'Y'],
            'DisbursementGross': [50000, 90000]
        })

if 'df' in st.session_state and models:
    df = st.session_state['df']
    if st.button("🚀 Run AI Analysis"):
        X = df.copy()
        X['NAICS'] = X['NAICS'].astype(str).str[:2]
        X['log_loan_amt'] = np.log1p(X['DisbursementGross'])
        X['new_business'] = X['NewExist'].apply(lambda x: 1 if x == 2 else 0)
        X['low_doc'] = X['LowDoc'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
        X['urban_flag'] = X['UrbanRural'].apply(lambda x: 1 if x > 0 else 0)

        try:
            df['PD_Predicted'] = models['PD'].predict_proba(X)[:, 1]
            df['LGD_Predicted'] = models['LGD'].predict(X)
            df['EL_Amount'] = df['PD_Predicted'] * df['LGD_Predicted'] * df['DisbursementGross']
            st.session_state['results'] = df
            st.success("Analysis Complete!")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if 'results' in st.session_state:
    results = st.session_state['results']
    tab1, tab2 = st.tabs([txt['tab1'], txt['tab2']])

    with tab1:
        st.dataframe(results)

    with tab2:
        debtor = st.selectbox("Select Debtor", results['Name'])
        d = results[results['Name'] == debtor].iloc[0]
        vix = st.slider("VIX", 10, 80, 20)
        baseline = d['PD_Predicted']
        multiplier = 1 + ((vix - 20) / 100) * 1.5 if vix > 20 else 1.0
        stressed_pd = min(baseline * multiplier, 1)
        stressed_el = stressed_pd * d['LGD_Predicted'] * d['DisbursementGross']

        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Baseline', 'Stressed'], y=[d['EL_Amount'], stressed_el]))
        st.plotly_chart(fig, use_container_width=True)
