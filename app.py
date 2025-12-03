import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Enterprise Credit Risk EWS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-med { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Language dictionary
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
        "model_missing": "Model tidak ditemukan otomatis. Silakan unggah manual di bawah ini.",
        "success_load": "Model berhasil dimuat dari sistem lokal!",
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
        "model_missing": "Models not found automatically. Please upload manually below.",
        "success_load": "Models loaded successfully from local system!",
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
    """Load PD and LGD models from local or deployment paths"""
    models = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_dirs = [
        base_dir,
        os.path.join(base_dir, "app"),
        os.path.join(base_dir, "..", "app"),
    ]
    
    pd_filename = "PD_model_tuned_pipeline.pkl"
    pd_path = None
    
    for d in search_dirs:
        full_path = os.path.join(d, pd_filename)
        if os.path.exists(full_path):
            pd_path = full_path
            break
    
    lgd_filename = "LGD_model_pipeline.pkl"
    lgd_path = None
    
    for d in search_dirs:
        full_path = os.path.join(d, lgd_filename)
        if os.path.exists(full_path):
            lgd_path = full_path
            break
    
    if not pd_path or not lgd_path:
        st.sidebar.error("🔍 Model files not found. Searched in:")
        for d in search_dirs:
            st.sidebar.caption(f"📁 {d}")
        st.sidebar.caption(f"Looking for: `{pd_filename}` and `{lgd_filename}`")
        st.sidebar.info(f"Current working directory: `{os.getcwd()}`")
    
    try:
        if pd_path:
            with open(pd_path, "rb") as f:
                models['PD'] = pickle.load(f)
                models['PD_Name'] = os.path.basename(pd_path)
        
        if lgd_path:
            with open(lgd_path, "rb") as f:
                models['LGD'] = pickle.load(f)
                models['LGD_Name'] = os.path.basename(lgd_path)
        
        return models if 'PD' in models and 'LGD' in models else None
    
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")
        return None

def preprocess_input(df):
    """Clean and transform input data for model prediction"""
    df_processed = df.copy()
    
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
    
    if 'NewExist' in df_processed.columns:
        df_processed['new_business'] = df_processed['NewExist'].apply(lambda x: 1 if str(x) == '2' or x == 2 else 0)
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

    if 'Term' not in df_processed.columns:
        df_processed['Term'] = 0
    if 'NoEmp' not in df_processed.columns:
        df_processed['NoEmp'] = 0

    return df_processed

def calculate_expected_loss(pd_val, lgd_val, ead_val):
    return pd_val * lgd_val * ead_val

def apply_stress_test(pd_val, vix_index):
    """Apply stress multiplier based on VIX volatility index"""
    baseline_vix = 20
    if vix_index <= baseline_vix:
        multiplier = 1.0
    else:
        multiplier = 1.0 + ((vix_index - baseline_vix) / 100) * 1.5
    
    stressed_pd = np.minimum(pd_val * multiplier, 1.0)
    return stressed_pd, multiplier

st.sidebar.markdown("---")
st.sidebar.subheader(f"🛠️ {txt['sidebar_model']}")

models = load_models_smart()

if models:
    st.sidebar.success(f"✅ {txt['success_load']}")
    st.sidebar.caption(f"PD Model: `{models.get('PD_Name')}`")
    st.sidebar.caption(f"LGD Model: `{models.get('LGD_Name')}`")
else:
    st.sidebar.warning(txt['model_missing'])
    uploaded_pd = st.sidebar.file_uploader("Upload PD Model (.pkl)", type="pkl")
    uploaded_lgd = st.sidebar.file_uploader("Upload LGD Model (.pkl)", type="pkl")
    
    if uploaded_pd and uploaded_lgd:
        models = {}
        try:
            models['PD'] = pickle.load(uploaded_pd)
            models['LGD'] = pickle.load(uploaded_lgd)
            st.sidebar.success("Models loaded from upload!")
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded files: {e}")

st.title(txt['title'])
st.markdown(f"**{txt['subtitle']}**")

st.write("---")
st.header(f"1. {txt['upload_header']}")

uploaded_file = st.file_uploader(txt['upload_label'], type=["csv", "xlsx"])

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
        df = pd.DataFrame(data)
        st.session_state['df'] = df
        st.info("Demo data generated! Proceed to analysis.")
else:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Error reading file: {e}")

if 'df' in st.session_state and models:
    df = st.session_state['df']
    
    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())
    
    if st.button("🚀 Run AI Analysis"):
        
        with st.spinner('Preprocessing data...'):
            try:
                X_processed = preprocess_input(df)
            except Exception as e:
                st.error(f"Preprocessing Error: {e}")
                st.stop()
        
        with st.spinner('Calculating Probability of Default & LGD...'):
            try:
                if hasattr(models['PD'], 'predict_proba'):
                    pd_values = models['PD'].predict_proba(X_processed)[:, 1]
                else:
                    pd_values = models['PD'].predict(X_processed)
                
                lgd_values = models['LGD'].predict(X_processed)
                
                df['PD_Predicted'] = pd_values
                df['LGD_Predicted'] = lgd_values
                
                if df['DisbursementGross'].dtype == 'object':
                    loan_amt = df['DisbursementGross'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
                else:
                    loan_amt = df['DisbursementGross']
                    
                df['EL_Amount'] = calculate_expected_loss(df['PD_Predicted'], df['LGD_Predicted'], loan_amt)
                
                df['Risk_Grade'] = pd.cut(df['PD_Predicted'], 
                                          bins=[-0.1, 0.05, 0.20, 1.0], 
                                          labels=['Low Risk', 'Medium Risk', 'High Risk'])
                
                st.session_state['results'] = df
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.warning("Make sure input columns match the model requirements (Term, NoEmp, NewExist, LowDoc, etc).")
                st.stop()

    if 'results' in st.session_state:
        results = st.session_state['results']
        
        st.write("---")
        tab1, tab2 = st.tabs([f"📊 {txt['tab1']}", f"⚡ {txt['tab2']}"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            avg_pd = results['PD_Predicted'].mean()
            total_el = results['EL_Amount'].sum()
            high_risk_count = results[results['Risk_Grade'] == 'High Risk'].shape[0]
            
            col1.metric("Average Portfolio PD", f"{avg_pd:.2%}")
            col2.metric("Total Expected Loss", f"${total_el:,.0f}")
            col3.metric("High Risk Debtors", f"{high_risk_count} Companies")
            
            st.subheader("Detail Data")
            
            display_df = results.copy()
            try:
                display_df['PD_Predicted'] = display_df['PD_Predicted'].map('{:.2%}'.format)
                display_df['LGD_Predicted'] = display_df['LGD_Predicted'].map('{:.2%}'.format)
                display_df['EL_Amount'] = display_df['EL_Amount'].map('${:,.2f}'.format)
            except:
                pass
            
            cols_to_show = ['Name', 'City', 'NAICS', 'DisbursementGross', 'PD_Predicted', 'LGD_Predicted', 'Risk_Grade', 'EL_Amount']
            final_cols = [c for c in cols_to_show if c in display_df.columns]
            
            st.dataframe(display_df[final_cols], use_container_width=True)
            
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 {txt['download_btn']}",
                data=csv,
                file_name='credit_risk_analysis_results.csv',
                mime='text/csv',
            )

        with tab2:
            st.subheader(txt['risk_profile'])
            
            if 'Name' in results.columns:
                debtor_list = results['Name'].unique()
                selected_debtor = st.selectbox("Select Debtor / Pilih Debitur", debtor_list)
                
                debtor_data = results[results['Name'] == selected_debtor].iloc[0]
                
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.markdown("#### Current Status (Baseline)")
                    st.info(f"**Industry (NAICS):** {debtor_data.get('NAICS', 'N/A')}")
                    st.write(f"**Loan Amount:** {debtor_data.get('DisbursementGross', 0)}")
                    st.write(f"**Original PD:** {debtor_data['PD_Predicted']:.2%}")
                    st.write(f"**Original LGD:** {debtor_data['LGD_Predicted']:.2%}")
                    
                    risk_grade = debtor_data.get('Risk_Grade', 'Unknown')
                    risk_color = "green" if risk_grade == 'Low Risk' else "orange" if risk_grade == 'Medium Risk' else "red"
                    st.markdown(f"Risk Grade: <span style='color:{risk_color}; font-size:1.2em; font-weight:bold'>{risk_grade}</span>", unsafe_allow_html=True)

                with c2:
                    st.markdown(f"#### 📉 {txt['stress_vix']}")
                    st.write(txt['stress_desc'])
                    
                    vix = st.slider("VIX Index Level", min_value=10, max_value=80, value=20, step=5)
                    
                    base_pd = debtor_data['PD_Predicted']
                    stressed_pd, multiplier = apply_stress_test(base_pd, vix)
                    
                    try:
                        raw_loan = str(debtor_data['DisbursementGross']).replace('$','').replace(',','')
                        loan_val = float(raw_loan)
                    except:
                        loan_val = 0.0
                        
                    base_el = debtor_data['EL_Amount']
                    stressed_el = calculate_expected_loss(stressed_pd, debtor_data['LGD_Predicted'], loan_val)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Baseline', 'Stressed'],
                        y=[base_el, stressed_el],
                        marker_color=['#2ecc71', '#e74c3c'],
                        text=[f"${base_el:,.0f}", f"${stressed_el:,.0f}"],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title=f"Expected Loss Impact (Multiplier: {multiplier:.2f}x)",
                        yaxis_title="Expected Loss ($)",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    diff = stressed_el - base_el
                    if diff > 0:
                        st.error(f"⚠️ Potential Loss Increase: +${diff:,.0f} under this scenario.")
                    else:
                        st.success("Portfolio remains stable under this scenario.")
            else:
                st.warning("Data does not have 'Name' column to select debtor.")

elif not models:
    st.info("👋 Welcome! Please upload your Model PKL files in the sidebar to begin.")
