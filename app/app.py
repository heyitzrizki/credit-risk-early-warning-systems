import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Enterprise Credit Risk EWS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
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

# ==========================================
# 2. LOCALIZATION (MANAJEMEN BAHASA)
# ==========================================
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
        "model_missing": "Model tidak ditemukan. Silakan unggah manual di sidebar.",
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
        "model_missing": "Model not found. Please upload manually in sidebar.",
        "success_load": "Models loaded successfully!",
        "risk_profile": "Debtor Risk Profile",
        "base_vs_stress": "Comparison: Baseline vs Stressed",
    }
}

# Language Selector in Sidebar
with st.sidebar:
    st.header("🌐 Language / Bahasa")
    lang_opt = st.selectbox("Select Language", ["ID", "EN"])
    txt = lang_dict[lang_opt]

# ==========================================
# 3. BACKEND LOGIC & FUNCTIONS
# ==========================================

@st.cache_resource
def load_models(pd_path, lgd_path):
    """
    Mencoba memuat model dari path lokal.
    Mengembalikan dictionary model atau None jika gagal.
    """
    models = {}
    try:
        with open(pd_path, 'rb') as f:
            models['PD'] = pickle.load(f)
        with open(lgd_path, 'rb') as f:
            models['LGD'] = pickle.load(f)
        return models
    except Exception as e:
        return None

def preprocess_input(df):
    """
    Pipeline pembersihan data & Feature Engineering.
    Menyesuaikan data mentah agar cocok dengan model Pipeline.
    """
    df_processed = df.copy()
    
    # 1. Cleaning Currency Columns (Menghapus '$' dan ',')
    currency_cols = ['DisbursementGross', 'GrAppv', 'SBA_Appv']
    for col in currency_cols:
        if col in df_processed.columns and df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

    # 2. Handling Missing Values (Simple Imputation)
    df_processed.fillna(0, inplace=True)

    # 3. Feature Engineering: NAICS (Industry Sector) -> 2 Digit
    if 'NAICS' in df_processed.columns:
        # Konversi ke string, ambil 2 digit pertama
        df_processed['NAICS'] = df_processed['NAICS'].astype(str).str[:2]
        # Mapping mapping spesifik jika diperlukan oleh OneHotEncoder pipeline Anda
        # (Disini kita asumsikan Pipeline scikit-learn menghandle 'unknown' category)
    
    # 4. Feature Engineering: Log Loan Amount
    # Model XGBoost biasanya menyukai distribusi normal
    if 'DisbursementGross' in df_processed.columns:
        df_processed['log_loan_amt'] = np.log1p(df_processed['DisbursementGross'])
    
    # 5. Mapping Binary Columns (Contoh umum data kredit)
    # Jika model dilatih dengan 'Y'/'N' -> 1/0, lakukan disini.
    # Jika pipeline memiliki OneHotEncoder untuk 'Y'/'N', biarkan saja.
    # Asumsi: Pipeline menghandle OneHotEncoding, jadi kita pastikan tipe datanya string/category yang konsisten.
    categorical_cols = ['RevLineCr', 'LowDoc', 'UrbanRural', 'NewExist']
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)

    return df_processed

def calculate_expected_loss(pd, lgd, ead):
    """EL = PD * LGD * EAD"""
    return pd * lgd * ead

def apply_stress_test(pd, vix_index):
    """
    Logika Stress Test Sederhana.
    Baseline VIX biasanya sekitar 15-20.
    Jika VIX naik, PD akan dikalikan dengan multiplier.
    """
    baseline_vix = 20
    # Sigmoid-like scaling atau Linear scaling
    if vix_index <= baseline_vix:
        multiplier = 1.0
    else:
        # Setiap kenaikan 10 poin VIX meningkatkan risiko sebesar 15% (contoh heuristik)
        multiplier = 1.0 + ((vix_index - baseline_vix) / 100) * 1.5
    
    stressed_pd = np.minimum(pd * multiplier, 1.0) # Cap at 100%
    return stressed_pd, multiplier

# ==========================================
# 4. SIDEBAR: MODEL MANAGEMENT
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader(f"🛠️ {txt['sidebar_model']}")

# Coba load otomatis
default_pd_path = "PD_model_tuned_pipeline.pkl"
default_lgd_path = "LGD_model_pipeline.pkl"

models = load_models(default_pd_path, default_lgd_path)

if models:
    st.sidebar.success(f"✅ {txt['success_load']}")
else:
    st.sidebar.warning(txt['model_missing'])
    # Fallback: Manual Upload
    uploaded_pd = st.sidebar.file_uploader("Upload PD Model (.pkl)", type="pkl")
    uploaded_lgd = st.sidebar.file_uploader("Upload LGD Model (.pkl)", type="pkl")
    
    if uploaded_pd and uploaded_lgd:
        models = {}
        try:
            models['PD'] = pickle.load(uploaded_pd)
            models['LGD'] = pickle.load(uploaded_lgd)
            st.sidebar.success(txt['success_load'])
        except:
            st.sidebar.error("Error loading uploaded files.")

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title(txt['title'])
st.markdown(f"**{txt['subtitle']}**")

# TAHAP 1: UPLOAD & PROCESSING
st.write("---")
st.header(f"1. {txt['upload_header']}")

uploaded_file = st.file_uploader(txt['upload_label'], type=["csv", "xlsx"])

# Template Data Generator (untuk demo jika user tidak punya file)
if not uploaded_file:
    if st.button("Generate Demo Data (Simulasi)"):
        data = {
            'LoanNr_ChkDgt': [1001, 1002, 1003, 1004, 1005],
            'Name': ['ABC Corp', 'Delta Mfg', 'Warung Sejahtera', 'Tech Indo', 'Mega Retail'],
            'City': ['Jakarta', 'Surabaya', 'Bandung', 'Jogja', 'Medan'],
            'State': ['JK', 'JI', 'JB', 'YO', 'SU'],
            'NAICS': ['33', '44', '54', '72', '81'], # Manufacturing, Retail, Tech, Food, Services
            'NoEmp': [50, 10, 5, 100, 15],
            'NewExist': ['1', '2', '1', '1', '2'],
            'UrbanRural': ['1', '1', '2', '1', '2'], # 1=Urban, 2=Rural
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
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.session_state['df'] = df

# JIKA DATA TERSEDIA
if 'df' in st.session_state and models:
    df = st.session_state['df']
    
    # 1. Preprocessing
    with st.spinner('Preprocessing data...'):
        X_processed = preprocess_input(df)
        
        # Validasi kolom wajib untuk model (disesuaikan dengan kebutuhan model XGBoost umum)
        # Note: Model Pipeline biasanya menghandle seleksi kolom, tapi kita butuh fitur utama
        
    # 2. Prediction Loop
    if st.button("🚀 Run AI Analysis"):
        with st.spinner('Calculating Probability of Default & LGD...'):
            try:
                # Prediksi PD (Ambil probabilitas kelas positif/gagal bayar)
                # Biasanya predict_proba outputnya [prob_0, prob_1]
                if hasattr(models['PD'], 'predict_proba'):
                    pd_values = models['PD'].predict_proba(X_processed)[:, 1]
                else:
                    pd_values = models['PD'].predict(X_processed) # Fallback jika regresi
                
                # Prediksi LGD
                lgd_values = models['LGD'].predict(X_processed)
                
                # Simpan hasil ke DataFrame
                df['PD_Predicted'] = pd_values
                df['LGD_Predicted'] = lgd_values
                
                # Hitung EL
                # Pastikan kolom numerik bersih untuk perhitungan
                loan_amt = df['DisbursementGross'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
                df['EL_Amount'] = calculate_expected_loss(df['PD_Predicted'], df['LGD_Predicted'], loan_amt)
                
                # Risk Grading (Indikator Warna)
                df['Risk_Grade'] = pd.cut(df['PD_Predicted'], 
                                          bins=[-0.1, 0.05, 0.20, 1.0], 
                                          labels=['Low Risk', 'Medium Risk', 'High Risk'])
                
                st.session_state['results'] = df
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.warning("Pastikan nama kolom data input sesuai dengan training data model.")
                st.stop()

    # TAHAP 2: DASHBOARD & REPORTING
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        st.write("---")
        tab1, tab2 = st.tabs([f"📊 {txt['tab1']}", f"⚡ {txt['tab2']}"])
        
        # === TAB 1: DATA REPORTING ===
        with tab1:
            # Metrics Summary
            col1, col2, col3 = st.columns(3)
            avg_pd = results['PD_Predicted'].mean()
            total_el = results['EL_Amount'].sum()
            high_risk_count = results[results['Risk_Grade'] == 'High Risk'].shape[0]
            
            col1.metric("Average Portfolio PD", f"{avg_pd:.2%}")
            col2.metric("Total Expected Loss", f"${total_el:,.0f}")
            col3.metric("High Risk Debtors", f"{high_risk_count} Companies")
            
            # Interactive Table with formatting
            st.subheader("Detail Data")
            
            # Formatting untuk display
            display_df = results.copy()
            display_df['PD_Predicted'] = display_df['PD_Predicted'].map('{:.2%}'.format)
            display_df['LGD_Predicted'] = display_df['LGD_Predicted'].map('{:.2%}'.format)
            display_df['EL_Amount'] = display_df['EL_Amount'].map('${:,.2f}'.format)
            
            st.dataframe(display_df[['Name', 'City', 'NAICS', 'DisbursementGross', 'PD_Predicted', 'LGD_Predicted', 'Risk_Grade', 'EL_Amount']], use_container_width=True)
            
            # Download Button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 {txt['download_btn']}",
                data=csv,
                file_name='credit_risk_analysis_results.csv',
                mime='text/csv',
            )

        # === TAB 2: STRESS TEST SIMULATOR ===
        with tab2:
            st.subheader(txt['risk_profile'])
            
            # Select Debtor
            debtor_list = results['Name'].unique()
            selected_debtor = st.selectbox("Select Debtor / Pilih Debitur", debtor_list)
            
            # Get Debtor Data
            debtor_data = results[results['Name'] == selected_debtor].iloc[0]
            
            # Layout: Profil Kiri, Simulator Kanan
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown("#### Current Status (Baseline)")
                st.info(f"**Industry (NAICS):** {debtor_data['NAICS']}")
                st.write(f"**Loan Amount:** {debtor_data['DisbursementGross']}")
                st.write(f"**Original PD:** {debtor_data['PD_Predicted']:.2%}")
                st.write(f"**Original LGD:** {debtor_data['LGD_Predicted']:.2%}")
                
                # Risk Badge
                risk_color = "green" if debtor_data['Risk_Grade'] == 'Low Risk' else "orange" if debtor_data['Risk_Grade'] == 'Medium Risk' else "red"
                st.markdown(f"Risk Grade: <span style='color:{risk_color}; font-size:1.2em; font-weight:bold'>{debtor_data['Risk_Grade']}</span>", unsafe_allow_html=True)

            with c2:
                st.markdown(f"#### 📉 {txt['stress_vix']}")
                st.write(txt['stress_desc'])
                
                # Slider VIX
                vix = st.slider("VIX Index Level", min_value=10, max_value=80, value=20, step=5)
                
                # Calculate Stress
                base_pd = debtor_data['PD_Predicted']
                stressed_pd, multiplier = apply_stress_test(base_pd, vix)
                
                # Recalculate EL
                loan_val = float(str(debtor_data['DisbursementGross']).replace('$','').replace(',',''))
                base_el = debtor_data['EL_Amount']
                stressed_el = calculate_expected_loss(stressed_pd, debtor_data['LGD_Predicted'], loan_val)
                
                # Visualisasi Perbandingan (Plotly)
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
                
                # Insight Box
                diff = stressed_el - base_el
                if diff > 0:
                    st.error(f"⚠️ Potential Loss Increase: +${diff:,.0f} under this scenario.")
                else:
                    st.success("Portfolio remains stable under this scenario.")

elif not models:
    st.info("👋 Welcome! Please upload your Model PKL files in the sidebar to begin.")
```

### Cara Menjalankan Aplikasi

1.  **Persiapkan File:**
    Pastikan file berikut berada dalam satu folder yang sama:
    * `app.py` (kode di atas)
    * `PD_model_tuned_pipeline.pkl` (File model PD Anda)
    * `LGD_model_pipeline.pkl` (File model LGD Anda)

2.  **Install Library:**
    Buka terminal dan jalankan perintah ini untuk menginstal dependensi:
    ```bash
    pip install streamlit pandas numpy scikit-learn xgboost plotly openpyxl
    ```
    *(Note: `openpyxl` dibutuhkan untuk membaca file Excel, `xgboost` dibutuhkan jika model Anda berbasis XGBoost)*.

3.  **Jalankan Streamlit:**
    Ketik perintah ini di terminal:
    ```bash
    streamlit run app.py
