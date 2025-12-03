import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Enterprise Credit Risk EWS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# --- LANGUAGE DICTIONARY ---
lang_dict = {
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Analisis Probabilitas Gagal Bayar & Stress Testing Ekonomi Makro",
        "upload_header": "Unggah Data Portofolio",
        "upload_label": "Unggah file CSV atau Excel (Data Peminjam)",
        "sidebar_model": "Status Model",
        "tab1": "📋 Laporan Portofolio",
        "tab2": "⚡ Inspektor & Stress Test",
        "download_btn": "Unduh Hasil Analisis",
        "model_missing": "Model tidak ditemukan. Pastikan file .pkl ada di folder yang sama.",
        "success_load": "Model berhasil dimuat!",
        "risk_profile": "Profil Risiko Debitur",
    },
    "EN": {
        "title": "Enterprise Credit Risk EWS",
        "subtitle": "Probability of Default Analysis & Macro Stress Testing",
        "upload_header": "Upload Portfolio Data",
        "upload_label": "Upload CSV or Excel file (Borrower Data)",
        "sidebar_model": "Model Status",
        "tab1": "📋 Portfolio Report",
        "tab2": "⚡ Inspector & Stress Test",
        "download_btn": "Download Analysis Results",
        "model_missing": "Models not found. Ensure .pkl files are in the same folder.",
        "success_load": "Models loaded successfully!",
        "risk_profile": "Debtor Risk Profile",
    }
}

with st.sidebar:
    st.header("🌐 Language / Bahasa")
    lang_opt = st.selectbox("Select Language", ["ID", "EN"])
    txt = lang_dict[lang_opt]

@st.cache_resource
def load_models_smart():
    models = {}

    # Always load from the same folder as THIS app.py
    base_dir = os.path.dirname(__file__)  
    pd_path = os.path.join(base_dir, "PD_model_tuned_pipeline.pkl")
    lgd_path = os.path.join(base_dir, "LGD_model_pipeline.pkl")

    st.sidebar.info(f"Looking for models in: {base_dir}")

    try:
        if os.path.exists(pd_path):
            models['PD'] = joblib.load(pd_path)
            st.sidebar.success(f"Loaded PD model: {pd_path}")

        if os.path.exists(lgd_path):
            models['LGD'] = joblib.load(lgd_path)
            st.sidebar.success(f"Loaded LGD model: {lgd_path}")

        if 'PD' in models and 'LGD' in models:
            return models

        return None

    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None


# --- FUNGSI PREPROCESSING ---
def preprocess_input(df):
    df_processed = df.copy()
    
    # Cleaning Currency
    currency_cols = ['DisbursementGross', 'GrAppv', 'SBA_Appv']
    for col in currency_cols:
        if col in df_processed.columns and df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        elif col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed.fillna(0, inplace=True)

    if 'NAICS' in df_processed.columns:
        df_processed['NAICS'] = df_processed['NAICS'].astype(str).str[:2]
    
    if 'DisbursementGross' in df_processed.columns:
        df_processed['log_loan_amt'] = np.log1p(df_processed['DisbursementGross'])
    
    # Feature Engineering Sederhana
    if 'NewExist' in df_processed.columns:
        df_processed['new_business'] = df_processed['NewExist'].apply(lambda x: 1 if str(x) in ['2', '2.0'] else 0)
    else:
        df_processed['new_business'] = 0

    if 'LowDoc' in df_processed.columns:
        df_processed['low_doc'] = df_processed['LowDoc'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
    else:
        df_processed['low_doc'] = 0

    if 'UrbanRural' in df_processed.columns:
        df_processed['urban_flag'] = df_processed['UrbanRural'].apply(lambda x: 1 if int(x) > 0 else 0)
    else:
        df_processed['urban_flag'] = 0

    # Kolom wajib dummy jika tidak ada
    for col in ['Term', 'NoEmp']:
        if col not in df_processed.columns:
            df_processed[col] = 0

    return df_processed

def calculate_expected_loss(pd_val, lgd_val, ead_val):
    return pd_val * lgd_val * ead_val

def apply_stress_test(pd_val, vix_index):
    baseline_vix = 20
    if vix_index <= baseline_vix:
        multiplier = 1.0
    else:
        multiplier = 1.0 + ((vix_index - baseline_vix) / 100) * 1.5
    stressed_pd = np.minimum(pd_val * multiplier, 1.0)
    return stressed_pd, multiplier

# --- MAIN APP LOGIC ---

st.sidebar.markdown("---")
st.sidebar.subheader(f"🛠️ {txt['sidebar_model']}")

models = load_models_smart()

if models:
    st.sidebar.success(f"✅ {txt['success_load']}")
    st.sidebar.caption(f"Files: {models.get('PD_Name')} & {models.get('LGD_Name')}")
else:
    st.sidebar.warning(txt['model_missing'])

st.title(txt['title'])
st.markdown(f"**{txt['subtitle']}**")
st.write("---")

# Header & Upload
st.header(f"1. {txt['upload_header']}")
uploaded_file = st.file_uploader(txt['upload_label'], type=["csv", "xlsx"])

# Demo Data Generation
if not uploaded_file:
    if st.button("Generate Demo Data (Simulasi)"):
        data = {
            'LoanNr_ChkDgt': [1001, 1002, 1003, 1004, 1005],
            'Name': ['ABC Corp', 'Delta Mfg', 'Warung Sejahtera', 'Tech Indo', 'Mega Retail'],
            'City': ['Jakarta', 'Surabaya', 'Bandung', 'Jogja', 'Medan'],
            'State': ['JK', 'JI', 'JB', 'YO', 'SU'],
            'NAICS': ['33', '44', '54', '72', '81'],
            'Term': [36, 60, 12, 120, 24],
            'NoEmp': [50, 10, 5, 100, 15],
            'NewExist': ['1', '2', '1', '1', '2'],
            'UrbanRural': ['1', '1', '2', '1', '2'],
            'RevLineCr': ['N', 'Y', 'N', 'Y', 'N'],
            'LowDoc': ['N', 'N', 'Y', 'N', 'Y'],
            'DisbursementGross': ['$500,000', '$150,000', '$50,000', '$1,200,000', '$75,000'],
            'GrAppv': ['$500,000', '$150,000', '$50,000', '$1,200,000', '$75,000'],
            'SBA_Appv': ['$400,000', '$75,000', '$40,000', '$900,000', '$60,000']
        }
        st.session_state['df'] = pd.DataFrame(data)
        st.info("Demo data generated! Proceed to analysis.")
else:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state['df'] = pd.read_csv(uploaded_file)
        else:
            st.session_state['df'] = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Process & Visualize
if 'df' in st.session_state and models:
    df = st.session_state['df']
    
    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())
    
    if st.button("🚀 Run AI Analysis"):
        with st.spinner('Calculating Risk Metrics...'):
            try:
                X_processed = preprocess_input(df)
                
                # Predict PD
                if hasattr(models['PD'], 'predict_proba'):
                    pd_values = models['PD'].predict_proba(X_processed)[:, 1]
                else:
                    pd_values = models['PD'].predict(X_processed)
                
                # Predict LGD
                lgd_values = models['LGD'].predict(X_processed)
                
                df['PD_Predicted'] = pd_values
                df['LGD_Predicted'] = lgd_values
                
                # Calculate EL
                if df['DisbursementGross'].dtype == 'object':
                    loan_amt = df['DisbursementGross'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
                else:
                    loan_amt = df['DisbursementGross']
                
                df['EL_Amount'] = calculate_expected_loss(df['PD_Predicted'], df['LGD_Predicted'], loan_amt)
                
                # Risk Grading
                df['Risk_Grade'] = pd.cut(df['PD_Predicted'], 
                                          bins=[-0.1, 0.05, 0.20, 1.0], 
                                          labels=['Low Risk', 'Medium Risk', 'High Risk'])
                
                st.session_state['results'] = df
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.warning("Ensure input columns match model requirements.")
                st.code(str(e)) # Show detailed error for debugging

    # Results Display
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        tab1, tab2 = st.tabs([f"📊 {txt['tab1']}", f"⚡ {txt['tab2']}"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Portfolio PD", f"{results['PD_Predicted'].mean():.2%}")
            col2.metric("Total Expected Loss", f"${results['EL_Amount'].sum():,.0f}")
            col3.metric("High Risk Companies", f"{results[results['Risk_Grade'] == 'High Risk'].shape[0]}")
            
            st.dataframe(results, use_container_width=True)
            
            st.download_button(
                label=f"📥 {txt['download_btn']}",
                data=results.to_csv(index=False).encode('utf-8'),
                file_name='credit_risk_analysis.csv',
                mime='text/csv',
            )

        with tab2:
            st.subheader(txt['risk_profile'])
            if 'Name' in results.columns:
                selected_debtor = st.selectbox("Select Debtor", results['Name'].unique())
                debtor_data = results[results['Name'] == selected_debtor].iloc[0]
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.info(f"**NAICS:** {debtor_data.get('NAICS', 'N/A')}")
                    st.write(f"**PD:** {debtor_data['PD_Predicted']:.2%}")
                    st.write(f"**LGD:** {debtor_data['LGD_Predicted']:.2%}")
                    grade = debtor_data.get('Risk_Grade', 'Unknown')
                    color = "green" if grade == 'Low Risk' else "orange" if grade == 'Medium Risk' else "red"
                    st.markdown(f"Grade: <span style='color:{color}; font-weight:bold'>{grade}</span>", unsafe_allow_html=True)

                with c2:
                    st.markdown(f"#### 📉 {txt['stress_vix']}")
                    vix = st.slider("VIX Index (Market Volatility)", 10, 80, 20)
                    
                    base_el = debtor_data['EL_Amount']
                    stressed_pd, _ = apply_stress_test(debtor_data['PD_Predicted'], vix)
                    
                    try:
                        amt = float(str(debtor_data['DisbursementGross']).replace('$','').replace(',',''))
                    except:
                        amt = 0
                    
                    stressed_el = calculate_expected_loss(stressed_pd, debtor_data['LGD_Predicted'], amt)
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Base', x=['Expected Loss'], y=[base_el], marker_color='#2ecc71'),
                        go.Bar(name='Stressed', x=['Expected Loss'], y=[stressed_el], marker_color='#e74c3c')
                    ])
                    fig.update_layout(barmode='group', height=300, title="Impact of Market Crisis on Loss")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Upload data with 'Name' column to use Inspector.")
