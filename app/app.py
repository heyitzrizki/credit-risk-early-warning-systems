import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Credit Risk EWS", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; }
    .stAlert { margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_models():
    models = {}
    
    # Mendapatkan lokasi folder tempat app.py berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path file model
    pd_path = os.path.join(current_dir, "PD_model_tuned_pipeline.pkl")
    lgd_path = os.path.join(current_dir, "LGD_model_pipeline.pkl")
    
    # Cek keberadaan file
    if not os.path.exists(pd_path) or not os.path.exists(lgd_path):
        st.sidebar.error("❌ Model files not found!")
        st.sidebar.caption(f"Dicari di: {current_dir}")
        return None

    try:
        # Load menggunakan Joblib (sesuai notebook kamu)
        models['PD'] = joblib.load(pd_path)
        models['LGD'] = joblib.load(lgd_path)
        
        # --- DEBUGGING SECTION (Cek Tipe Data Model) ---
        # Ini akan muncul di sidebar untuk memastikan yang di-load bukan Array
        st.sidebar.markdown("---")
        st.sidebar.caption("🕵️ **Model Diagnostics:**")
        st.sidebar.caption(f"PD Type: `{type(models['PD']).__name__}`")
        st.sidebar.caption(f"LGD Type: `{type(models['LGD']).__name__}`")
        
        # Validasi sederhana
        if isinstance(models['PD'], (np.ndarray, list)):
             st.error("🚨 CRITICAL ERROR: File PD Model berisi Array/List angka, bukan Model Machine Learning!")
             st.stop()
             
        return models

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
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

    # Feature Engineering (Harus sama persis dengan Notebook)
    if 'NAICS' in df_processed.columns:
        df_processed['NAICS'] = df_processed['NAICS'].astype(str)
        df_processed['NAICS_2'] = df_processed['NAICS'].str[:2]
    
    if 'DisbursementGross' in df_processed.columns:
        df_processed['log_loan_amt'] = np.log1p(df_processed['DisbursementGross'])
    
    if 'NewExist' in df_processed.columns:
        df_processed['new_business'] = df_processed['NewExist'].apply(lambda x: 1 if str(x) in ['2', '2.0'] else 0)

    if 'LowDoc' in df_processed.columns:
        df_processed['low_doc'] = df_processed['LowDoc'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)

    if 'UrbanRural' in df_processed.columns:
        df_processed['urban_flag'] = df_processed['UrbanRural'].apply(lambda x: 1 if int(x) == 1 else 0)
    
    # Pastikan ApprovalFY jadi integer
    if 'ApprovalFY' in df_processed.columns:
        # Bersihkan data non-numeric jika ada
        df_processed['ApprovalFY'] = pd.to_numeric(df_processed['ApprovalFY'], errors='coerce').fillna(0).astype(int)

    return df_processed

# --- APLIKASI UTAMA ---
st.title("🏦 Credit Risk Early Warning System")
st.write("Upload data debitur untuk memprediksi probabilitas gagal bayar (PD) dan estimasi kerugian (LGD).")

# Load Model
models = load_models()

# Upload File
uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])

if not uploaded_file:
    if st.button("Gunakan Data Demo"):
        # Data dummy untuk tes
        data = {
            'LoanNr_ChkDgt': [1001, 1002],
            'Name': ['ABC Corp', 'XYZ Ltd'],
            'City': ['New York', 'Los Angeles'],
            'State': ['NY', 'CA'],
            'Zip': [10001, 90001],
            'Bank': ['Bank A', 'Bank B'],
            'BankState': ['NY', 'CA'],
            'NAICS': ['541100', '722511'],
            'ApprovalDate': ['20-Jan-2010', '15-Feb-2011'],
            'ApprovalFY': [2010, 2011],
            'Term': [84, 60],
            'NoEmp': [5, 10],
            'NewExist': [1, 2],
            'CreateJob': [0, 2],
            'RetainedJob': [5, 8],
            'FranchiseCode': [0, 1],
            'UrbanRural': [1, 2],
            'RevLineCr': ['N', 'N'],
            'LowDoc': ['N', 'Y'],
            'DisbursementGross': ['$100,000', '$50,000'],
            'BalanceGross': ['$0', '$0'],
            'ChgOffPrinGr': ['$0', '$0'],
            'GrAppv': ['$100,000', '$50,000'],
            'SBA_Appv': ['$50,000', '$25,000']
        }
        st.session_state['df'] = pd.DataFrame(data)
        st.rerun()
else:
    try:
        st.session_state['df'] = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

if 'df' in st.session_state and models:
    df = st.session_state['df']
    
    with st.expander("🔍 Preview Data Mentah"):
        st.dataframe(df.head())
    
    if st.button("🚀 Jalankan Prediksi"):
        with st.spinner('Sedang memproses data...'):
            try:
                # 1. Preprocessing
                X_processed = preprocess_input(df)
                
                # 2. Prediction
                # Menggunakan predict_proba untuk PD jika model mendukung
                if hasattr(models['PD'], "predict_proba"):
                    pd_values = models['PD'].predict_proba(X_processed)[:, 1]
                else:
                    pd_values = models['PD'].predict(X_processed)
                
                # LGD Prediction
                lgd_values = models['LGD'].predict(X_processed)
                
                # 3. Hasil
                df['PD_Predicted'] = pd_values
                df['LGD_Predicted'] = lgd_values
                
                # Cleaning Loan Amount untuk perhitungan
                if df['DisbursementGross'].dtype == 'object':
                    loan_amt = df['DisbursementGross'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
                else:
                    loan_amt = df['DisbursementGross']
                
                df['Expected_Loss'] = df['PD_Predicted'] * df['LGD_Predicted'] * loan_amt
                
                # Risk Grading
                df['Risk_Grade'] = pd.cut(df['PD_Predicted'], 
                                          bins=[-0.1, 0.05, 0.20, 1.0], 
                                          labels=['Low', 'Medium', 'High'])
                
                st.success("✅ Prediksi Selesai!")
                
                # Tampilkan Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Rata-rata PD", f"{df['PD_Predicted'].mean():.2%}")
                c2.metric("Total Potensi Rugi (EL)", f"${df['Expected_Loss'].sum():,.0f}")
                c3.metric("Debitur High Risk", len(df[df['Risk_Grade']=='High']))
                
                # Tampilkan Tabel Hasil
                st.subheader("📋 Hasil Analisis")
                cols = ['Name', 'DisbursementGross', 'PD_Predicted', 'LGD_Predicted', 'Risk_Grade', 'Expected_Loss']
                final_cols = [c for c in cols if c in df.columns]
                
                st.dataframe(df[final_cols].style.format({
                    'PD_Predicted': '{:.2%}',
                    'LGD_Predicted': '{:.2%}',
                    'Expected_Loss': '${:,.2f}'
                }))
                
            except Exception as e:
                st.error("Terjadi Kesalahan saat Prediksi:")
                st.code(str(e))
                # Debugging bantu user
                st.warning("Pastikan kolom input sesuai dengan fitur yang digunakan saat training model.")
                if 'X_processed' in locals():
                    st.write("Kolom yang diproses:", X_processed.columns.tolist())

elif not models:
    st.info("Menunggu model dimuat...")
