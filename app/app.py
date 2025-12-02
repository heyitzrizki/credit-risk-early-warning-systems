import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import os

# --- 0. CHECK LIBRARIES ---
try:
    import xgboost as xgb
except ImportError:
    st.error("⚠️ Library 'xgboost' not found. Please run: pip install xgboost")
    st.stop()

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Credit Risk EWS",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .risk-safe { color: #2ecc71; font-weight: bold; }
    .risk-warning { color: #f1c40f; font-weight: bold; }
    .risk-danger { color: #e74c3c; font-weight: bold; }
    .stDataFrame { border: 1px solid #ddd; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LANGUAGE DICTIONARY ---
lang_dict = {
    "English": {
        "title": "🏢 Enterprise Credit Risk System",
        "missing_model": "⚠️ Model files not found in folder.",
        "manual_upload": "📥 Please upload .pkl files manually below:",
        "upload_pd": "Upload PD Model (Pipeline)",
        "upload_lgd": "Upload LGD Model (Pipeline)",
        "upload_header": "📂 1. Upload Dataset",
        "upload_desc": "Upload borrower data (CSV/Excel) for batch prediction.",
        "upload_label": "Choose Data File",
        "sim_header": "🔍 2. Borrower Inspector & Stress Test",
        "select_borrower": "Select Borrower ID:",
        "current_vix": "Market Condition (VIX Index):",
        "stress_analysis": "Stress Test Analysis",
        "stress_insight": "Simulating crisis impact on THIS specific company.",
        "metric_pd": "Probability of Default",
        "metric_el": "Expected Loss",
        "tab_data": "📋 Full Data & Predictions",
        "tab_sim": "🔬 Individual Simulator",
        "download_pred": "Download Predictions"
    },
    "Bahasa Indonesia": {
        "title": "🏢 Enterprise Credit Risk System",
        "missing_model": "⚠️ File model tidak ditemukan di folder.",
        "manual_upload": "📥 Silakan upload file .pkl secara manual di bawah:",
        "upload_pd": "Upload Model PD (Pipeline)",
        "upload_lgd": "Upload Model LGD (Pipeline)",
        "upload_header": "📂 1. Upload Dataset",
        "upload_desc": "Upload data peminjam (CSV/Excel) untuk prediksi massal.",
        "upload_label": "Pilih File Data",
        "sim_header": "🔍 2. Inspeksi Debitur & Stress Test",
        "select_borrower": "Pilih ID Peminjam:",
        "current_vix": "Kondisi Pasar (VIX Index):",
        "stress_analysis": "Analisa Stress Test",
        "stress_insight": "Mensimulasikan dampak krisis pada perusahaan INI.",
        "metric_pd": "Peluang Gagal Bayar (PD)",
        "metric_el": "Estimasi Kerugian (EL)",
        "tab_data": "📋 Data Lengkap & Hasil",
        "tab_sim": "🔬 Simulator Individu",
        "download_pred": "Download Hasil Prediksi"
    }
}

# --- 3. FUNCTIONS ---

def try_load_from_disk():
    """Mencoba load file model dari folder lokal secara otomatis"""
    models = {}
    try:
        # Cek PD Model
        pd_names = ['PD_model_calibrated_pipeline (1).pkl', 'PD_model_calibrated_pipeline.pkl']
        for name in pd_names:
            if os.path.exists(name):
                with open(name, 'rb') as f:
                    models['pd'] = pickle.load(f)
                break
        
        # Cek LGD Model
        lgd_names = ['LGD_model_pipeline (1).pkl', 'LGD_model_pipeline.pkl']
        for name in lgd_names:
            if os.path.exists(name):
                with open(name, 'rb') as f:
                    models['lgd'] = pickle.load(f)
                break
        
        if 'pd' in models and 'lgd' in models:
            return models
        return None
    except Exception:
        return None

def preprocess_input(df_raw):
    df = df_raw.copy()
    
    # 1. Standardize Columns
    col_map = {
        'Amount': 'DisbursementGross', 'LoanAmount': 'DisbursementGross',
        'Time': 'Term', 'Duration': 'Term',
        'Employees': 'NoEmp',
        'Sector': 'NAICS',
        'UrbanRural': 'UrbanRural', 'NewExist': 'NewExist', 'LowDoc': 'LowDoc'
    }
    df.rename(columns=col_map, inplace=True)
    
    required_cols = ['DisbursementGross', 'Term', 'NoEmp', 'NAICS', 'NewExist', 'LowDoc', 'UrbanRural']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        return None, f"Missing Columns: {missing}"

    # 2. Feature Engineering
    # Log Loan Amount
    # Handle 0 or negative values safely for log
    df['DisbursementGross'] = pd.to_numeric(df['DisbursementGross'], errors='coerce').fillna(0)
    df['log_loan_amt'] = np.log(df['DisbursementGross'] + 1)
    
    # New Business
    df['new_business'] = df['NewExist'] 
    
    # Low Doc
    df['low_doc'] = df['LowDoc'].apply(lambda x: 1 if str(x).upper() in ['Y', '1', 'TRUE', 'YES'] else 0)
    
    # Urban Flag
    df['urban_flag'] = df['UrbanRural']
    
    # NAICS - Ensure it is string and take first 2 chars
    df['NAICS'] = df['NAICS'].astype(str).apply(lambda x: x.split('.')[0][:2]) 
    
    # Final DataFrame
    X_pred = df[['Term', 'NoEmp', 'log_loan_amt', 'new_business', 'low_doc', 'urban_flag', 'NAICS']]
    
    return df, X_pred

# --- 4. MAIN APP ---

def main():
    # Sidebar Language
    lang = st.sidebar.selectbox("Language / Bahasa", ["Bahasa Indonesia", "English"])
    t = lang_dict[lang]
    
    st.title(t["title"])
    
    # --- MODEL LOADING LOGIC ---
    models = try_load_from_disk()
    
    # Jika gagal load dari disk, minta upload manual di Sidebar
    if models is None:
        st.sidebar.markdown("---")
        st.sidebar.warning(t["missing_model"])
        st.sidebar.info(t["manual_upload"])
        
        up_pd = st.sidebar.file_uploader(t["upload_pd"], type="pkl", key="pd")
        up_lgd = st.sidebar.file_uploader(t["upload_lgd"], type="pkl", key="lgd")
        
        if up_pd and up_lgd:
            try:
                models = {
                    'pd': pickle.load(up_pd),
                    'lgd': pickle.load(up_lgd)
                }
                st.sidebar.success("✅ Models Loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading files: {e}")
                st.stop()
        else:
            st.warning(f"🛑 {t['missing_model']} {t['manual_upload']} (Check Sidebar)")
            st.stop()

    # --- UPLOAD DATA SECTION ---
    st.header(t["upload_header"])
    st.markdown(t["upload_desc"])
    
    data_file = st.file_uploader(t["upload_label"], type=["csv", "xlsx"])
    
    if data_file:
        # Load Data
        try:
            if data_file.name.endswith('.csv'):
                # Coba deteksi separator (comma vs semicolon)
                try:
                    df_raw = pd.read_csv(data_file)
                    if df_raw.shape[1] < 2: # Kemungkinan salah separator
                        data_file.seek(0)
                        df_raw = pd.read_csv(data_file, sep=';')
                except:
                    data_file.seek(0)
                    df_raw = pd.read_csv(data_file, sep=';')
            else:
                df_raw = pd.read_excel(data_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        # Preprocess
        df_processed, X_pred_or_error = preprocess_input(df_raw)
        
        if isinstance(X_pred_or_error, str):
            st.error(f"❌ {X_pred_or_error}")
            st.info("Required columns: DisbursementGross, Term, NoEmp, NAICS, NewExist, LowDoc, UrbanRural")
            st.stop()
        
        X_pred = X_pred_or_error

        # --- PREDICT ---
        with st.spinner("Running AI Models..."):
            try:
                # PD Prediction
                df_processed['PD_Predicted'] = models['pd'].predict_proba(X_pred)[:, 1]
                
                # LGD Prediction
                lgd_preds = models['lgd'].predict(X_pred)
                df_processed['LGD_Predicted'] = np.clip(lgd_preds, 0, 1)
                
                # EL Calculation
                df_processed['Expected_Loss'] = df_processed['PD_Predicted'] * df_processed['LGD_Predicted'] * df_processed['DisbursementGross']
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.error("Pastikan data input sesuai dengan format training model (tipe data, nama kolom, dll).")
                st.stop()

        st.success(f"✅ Success! Processed {len(df_processed)} rows.")

        # --- TABS ---
        tab1, tab2 = st.tabs([t["tab_data"], t["tab_sim"]])
        
        with tab1:
            st.dataframe(df_processed.style.format({
                'DisbursementGross': '${:,.2f}',
                'PD_Predicted': '{:.2%}', 
                'LGD_Predicted': '{:.2%}',
                'Expected_Loss': '${:,.2f}'
            }), use_container_width=True)
            
            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(t["download_pred"], csv, "risk_predictions.csv", "text/csv")

        with tab2:
            st.subheader(t["sim_header"])
            
            # Selector Row
            row_options = df_processed.index.tolist()
            # Format label dropdown
            fmt = lambda i: f"Row {i+1} | {df_processed.iloc[i]['NAICS']} | ${df_processed.iloc[i]['DisbursementGross']:,.0f}"
            
            sel_idx = st.selectbox(t["select_borrower"], row_options, format_func=fmt)
            row = df_processed.loc[sel_idx]

            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.info(f"**Industry:** {row['NAICS']}")
                st.write(f"**Loan:** ${row['DisbursementGross']:,.0f}")
                st.write(f"**Emp:** {row['NoEmp']}")
                st.write(f"**Loc:** {'Urban' if row['urban_flag']==1 else 'Rural'}")
                st.divider()
                st.metric(t["metric_pd"], f"{row['PD_Predicted']:.2%}")
                st.metric(t["metric_el"], f"${row['Expected_Loss']:,.0f}")

            with c2:
                st.markdown(f"### 📉 {t['stress_analysis']}")
                st.caption(t["stress_insight"])
                
                vix = st.slider(t["current_vix"], 10, 80, 15)
                
                # Logic VIX
                mult, status, color = 1.0, "Normal", "green"
                if vix > 35: mult, status, color = 2.0, "Severe Crisis", "red"
                elif vix > 25: mult, status, color = 1.5, "High Stress", "orange"
                elif vix > 15: mult, status, color = 1.2, "Moderate Stress", "#f1c40f"
                
                st.markdown(f"**Status:** <b style='color:{color}'>{status}</b> (x{mult})", unsafe_allow_html=True)
                
                # Recalculate
                pd_stress = min(row['PD_Predicted'] * mult, 1.0)
                el_stress = pd_stress * row['LGD_Predicted'] * row['DisbursementGross']
                
                col_a, col_b = st.columns(2)
                col_a.metric("Stressed PD", f"{pd_stress:.2%}", delta=f"{(pd_stress-row['PD_Predicted'])*100:.2f}% pts", delta_color="inverse")
                col_b.metric("Stressed EL", f"${el_stress:,.0f}", delta=f"${el_stress-row['Expected_Loss']:,.0f}", delta_color="inverse")
                
                # Chart
                fig = go.Figure(data=[
                    go.Bar(name='Baseline', x=['Expected Loss'], y=[row['Expected_Loss']], marker_color='#2ecc71'),
                    go.Bar(name='Stressed', x=['Expected Loss'], y=[el_stress], marker_color='#e74c3c')
                ])
                fig.update_layout(height=250, margin=dict(t=30,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
