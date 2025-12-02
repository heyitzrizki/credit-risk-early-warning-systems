import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# --- 0. SAFETY CHECK LIBRARY ---
try:
    import xgboost as xgb
except ImportError:
    st.error("⚠️ Library 'xgboost' not found. Please run: pip install xgboost")
    st.stop()

# --- 1. SETUP HALAMAN ---
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

# --- 2. LOGIKA UTILS & BAHASA ---
lang_dict = {
    "English": {
        "title": "🏢 Enterprise Credit Risk System",
        "upload_header": "📂 1. Upload Dataset",
        "upload_desc": "Upload your borrower data (CSV). The model will predict risk for ALL rows.",
        "upload_label": "Choose CSV File",
        "col_missing": "⚠️ Missing columns in CSV:",
        "sim_header": "🔍 2. Borrower Inspector & Stress Test",
        "select_borrower": "Select Borrower ID / Row:",
        "current_vix": "Current Market VIX:",
        "stress_analysis": "Stress Test Analysis",
        "stress_insight": "Simulating crisis impact on THIS specific company.",
        "metric_pd": "Probability of Default",
        "metric_el": "Expected Loss",
        "tab_data": "📋 Full Data & Predictions",
        "tab_sim": "🔬 Individual Simulator",
        "download_pred": "Download Predictions",
        "manual_load": "⚠️ Auto-load failed. Please upload .pkl files manually in sidebar."
    },
    "Bahasa Indonesia": {
        "title": "🏢 Enterprise Credit Risk System",
        "upload_header": "📂 1. Upload Dataset",
        "upload_desc": "Upload data peminjam (CSV). Model akan memprediksi risiko untuk SEMUA baris.",
        "upload_label": "Pilih File CSV",
        "col_missing": "⚠️ Kolom hilang di CSV:",
        "sim_header": "🔍 2. Inspeksi Debitur & Stress Test",
        "select_borrower": "Pilih ID Peminjam / Baris:",
        "current_vix": "Kondisi Pasar (VIX Index):",
        "stress_analysis": "Analisa Stress Test",
        "stress_insight": "Mensimulasikan dampak krisis pada perusahaan INI.",
        "metric_pd": "Peluang Gagal Bayar (PD)",
        "metric_el": "Estimasi Kerugian (EL)",
        "tab_data": "📋 Data Lengkap & Hasil",
        "tab_sim": "🔬 Simulator Individu",
        "download_pred": "Download Hasil Prediksi",
        "manual_load": "⚠️ Gagal memuat otomatis. Silakan upload file .pkl manual di sidebar."
    }
}

# --- 3. FUNGSI LOAD & PREPROCESS ---

@st.cache_resource
def load_models():
    models = {}
    try:
        # UPDATE: Prioritaskan nama file "bersih" sesuai request user
        # List ini akan mencoba nama pertama, jika gagal baru nama kedua
        files = {
            'pd': ['PD_model_calibrated_pipeline.pkl', 'PD_model_calibrated_pipeline (1).pkl'],
            'lgd': ['LGD_model_pipeline.pkl', 'LGD_model_pipeline (1).pkl']
        }
        
        for key, possible_names in files.items():
            model_found = False
            for fname in possible_names:
                try:
                    with open(fname, 'rb') as f:
                        models[key] = pickle.load(f)
                    model_found = True
                    break # Berhenti jika file ditemukan
                except FileNotFoundError:
                    continue
            
            if not model_found:
                return None # Jika tidak ada satupun file yang cocok
                
        return models
    except Exception as e:
        st.error(f"Error Loading Models: {e}")
        return None

def preprocess_input(df_raw):
    """
    Mengubah Raw Data User (CSV) menjadi format yang siap dimakan Model Pipeline.
    """
    df = df_raw.copy()
    
    # 1. Mapping Kolom (Handle variasi nama kolom umum)
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
        return None, f"Kolom wajib tidak ditemukan: {missing}"

    # 2. Feature Engineering
    # Log Loan Amount
    df['DisbursementGross'] = pd.to_numeric(df['DisbursementGross'], errors='coerce').fillna(0)
    df['log_loan_amt'] = np.log(df['DisbursementGross'] + 1)
    
    # New Business (Logic: 2 = New, 1 = Existing)
    df['new_business'] = df['NewExist'] 
    
    # Low Doc (Mapping Y/N/1/0 -> 1/0)
    df['low_doc'] = df['LowDoc'].apply(lambda x: 1 if str(x).upper() in ['Y', '1', 'TRUE', 'YES'] else 0)
    
    # Urban Flag
    df['urban_flag'] = df['UrbanRural']
    
    # NAICS (Ambil 2 digit pertama)
    df['NAICS'] = df['NAICS'].astype(str).str.split('.').str[0].str[:2]
    
    # Final DataFrame untuk Prediksi
    X_pred = df[['Term', 'NoEmp', 'log_loan_amt', 'new_business', 'low_doc', 'urban_flag', 'NAICS']]
    
    return df, X_pred

# --- 4. APLIKASI UTAMA ---

def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    lang = st.sidebar.selectbox("Language / Bahasa", ["Bahasa Indonesia", "English"])
    t = lang_dict[lang]
    
    st.title(t["title"])
    
    # Load Models
    models = load_models()
    
    # Fallback: Jika auto-load gagal, munculkan tombol upload manual
    if not models:
        st.sidebar.warning(t["manual_load"])
        uploaded_pd = st.sidebar.file_uploader("Upload PD Model (.pkl)", type="pkl")
        uploaded_lgd = st.sidebar.file_uploader("Upload LGD Model (.pkl)", type="pkl")
        
        if uploaded_pd and uploaded_lgd:
            try:
                models = {
                    'pd': pickle.load(uploaded_pd),
                    'lgd': pickle.load(uploaded_lgd)
                }
                st.sidebar.success("✅ Models Loaded Manually!")
            except Exception as e:
                st.error(f"Error loading manual files: {e}")
                st.stop()
        else:
            st.warning("⚠️ " + t["manual_load"])
            st.stop()

    # --- BAGIAN 1: UPLOAD & BATCH PROCESS ---
    st.header(t["upload_header"])
    st.markdown(t["upload_desc"])
    
    uploaded_file = st.file_uploader(t["upload_label"], type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Baca File
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_raw = pd.read_csv(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, sep=';')
            else:
                df_raw = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        # Preprocessing & Validasi
        df_processed, X_pred_or_error = preprocess_input(df_raw)
        
        if isinstance(X_pred_or_error, str):
            st.error(X_pred_or_error)
            st.info("💡 Format CSV yang diharapkan (Kolom): DisbursementGross, Term, NoEmp, NAICS, NewExist, LowDoc, UrbanRural")
            return
        
        X_pred = X_pred_or_error
        
        # --- JALANKAN MODEL (BATCH) ---
        with st.spinner("🤖 Running AI Models..."):
            # 1. PD Prediction
            probs = models['pd'].predict_proba(X_pred)[:, 1]
            df_processed['PD_Predicted'] = probs
            
            # 2. LGD Prediction
            lgd_preds = models['lgd'].predict(X_pred)
            lgd_preds = np.clip(lgd_preds, 0, 1) # Clip 0-1
            df_processed['LGD_Predicted'] = lgd_preds
            
            # 3. EL Calculation
            df_processed['Expected_Loss'] = df_processed['PD_Predicted'] * df_processed['LGD_Predicted'] * df_processed['DisbursementGross']
            
        st.success(f"✅ Prediksi selesai untuk {len(df_processed)} baris data!")

        # --- BAGIAN 2: DASHBOARD & SIMULATOR ---
        
        tab1, tab2 = st.tabs([t["tab_data"], t["tab_sim"]])
        
        # TAB 1: DATA TABLE
        with tab1:
            st.dataframe(df_processed, use_container_width=True)
            
            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=t["download_pred"],
                data=csv,
                file_name='credit_risk_predictions.csv',
                mime='text/csv',
            )
            
        # TAB 2: INDIVIDUAL INSPECTOR & STRESS TEST
        with tab2:
            st.subheader(t["sim_header"])
            
            # Selector Row
            row_options = df_processed.index.tolist()
            format_func = lambda x: f"Row {x} | Sektor {df_processed.iloc[x]['NAICS']} | ${df_processed.iloc[x]['DisbursementGross']:,.0f}"
            
            selected_idx = st.selectbox(t["select_borrower"], row_options, format_func=format_func)
            
            # Ambil Data Baris Terpilih
            row_data = df_processed.iloc[selected_idx]
            
            st.markdown("---")
            
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.markdown("### 🏢 Profil Debitur")
                st.info(f"**Sektor Industri:** {row_data['NAICS']}")
                st.write(f"**Pinjaman:** ${row_data['DisbursementGross']:,.2f}")
                st.write(f"**Tenor:** {row_data['Term']} bulan")
                st.write(f"**Karyawan:** {row_data['NoEmp']}")
                st.write(f"**Lokasi:** {'Urban' if row_data['urban_flag']==1 else 'Rural'}")
                st.divider()
                st.metric(t["metric_pd"], f"{row_data['PD_Predicted']:.2%}")
                st.metric(t["metric_el"], f"${row_data['Expected_Loss']:,.2f}")

            with col_right:
                st.markdown(f"### 📉 {t['stress_analysis']}")
                st.write(t["stress_insight"])
                
                # Slider VIX
                vix_val = st.slider(t["current_vix"], 10, 80, 15, key="vix_slider")
                
                # Logic Stress Test
                stress_mult = 1.0
                status = "Normal"
                color = "green"
                
                if vix_val <= 15:
                    stress_mult = 1.0
                elif 15 < vix_val <= 25:
                    stress_mult = 1.2
                    status = "Moderate Stress"
                    color = "orange"
                elif 25 < vix_val <= 35:
                    stress_mult = 1.5
                    status = "High Stress"
                    color = "darkorange"
                else:
                    stress_mult = 2.0
                    status = "Severe Crisis"
                    color = "red"
                
                pd_base = row_data['PD_Predicted']
                pd_stress = min(pd_base * stress_mult, 1.0)
                
                el_base = row_data['Expected_Loss']
                el_stress = pd_stress * row_data['LGD_Predicted'] * row_data['DisbursementGross']
                delta_el = el_stress - el_base
                
                st.markdown(f"**Status Ekonomi:** <span style='color:{color}'>{status}</span> (Multiplier: {stress_mult}x)", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Stressed PD", f"{pd_stress:.2%}", delta=f"{(pd_stress-pd_base)*100:.2f}% pts", delta_color="inverse")
                with c2:
                    st.metric("Stressed EL", f"${el_stress:,.2f}", delta=f"-${delta_el:,.2f}", delta_color="inverse")
                
                fig = go.Figure(data=[
                    go.Bar(name='Normal', x=['Expected Loss'], y=[el_base], marker_color='#2ecc71'),
                    go.Bar(name='Crisis Scenario', x=['Expected Loss'], y=[el_stress], marker_color='#e74c3c')
                ])
                fig.update_layout(title_text=f"Impact of {status} on Selected Borrower", barmode='group', height=300)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
