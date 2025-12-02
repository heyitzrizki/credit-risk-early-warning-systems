import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb

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
        "download_pred": "Download Predictions"
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
        "download_pred": "Download Hasil Prediksi"
    }
}

# --- 3. FUNGSI LOAD & PREPROCESS ---

@st.cache_resource
def load_models():
    models = {}
    try:
        # Load dengan penanganan nama file yang fleksibel
        files = {
            'pd': ['PD_model_calibrated_pipeline (1).pkl', 'PD_model_calibrated_pipeline.pkl'],
            'lgd': ['LGD_model_pipeline (1).pkl', 'LGD_model_pipeline.pkl']
        }
        
        for key, possible_names in files.items():
            for fname in possible_names:
                try:
                    with open(fname, 'rb') as f:
                        models[key] = pickle.load(f)
                    break
                except FileNotFoundError:
                    continue
            if key not in models:
                return None # Jika salah satu model gagal load
                
        return models
    except Exception as e:
        st.error(f"Error Loading Models: {e}")
        return None

def preprocess_input(df_raw):
    """
    Mengubah Raw Data User (CSV) menjadi format yang siap dimakan Model Pipeline.
    Sesuai logic di Notebook: Hitung Log Loan, Mapping NAICS, dll.
    """
    df = df_raw.copy()
    
    # 1. Pastikan Kolom Wajib Ada (Nama kolom sesuai standar SBA Dataset)
    # Mapping nama kolom umum ke nama kolom model jika user salah nama
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

    # 2. Feature Engineering (Sesuai Notebook)
    # Log Loan Amount
    df['log_loan_amt'] = np.log(df['DisbursementGross'] + 1)
    
    # Mapping Model Features
    # Model mengharapkan kolom: ['Term', 'NoEmp', 'log_loan_amt', 'new_business', 'low_doc', 'urban_flag', 'NAICS']
    
    # New Business (Logic: 2 = New, 1 = Existing)
    # Asumsi input user 1/2. Jika user input string, perlu mapping tambahan.
    df['new_business'] = df['NewExist'] 
    
    # Low Doc (Logic: 'Y'=1, 'N'=0 atau input user sudah 1/0)
    df['low_doc'] = df['LowDoc'].apply(lambda x: 1 if str(x).upper() in ['Y', '1', 'TRUE'] else 0)
    
    # Urban Flag (Logic: 1=Urban, 2=Rural, 0=Undefined) -> Model butuh mapping ini
    df['urban_flag'] = df['UrbanRural']
    
    # NAICS
    # Pastikan NAICS string agar OneHotEncoder bekerja
    df['NAICS'] = df['NAICS'].astype(str).str[:2] # Ambil 2 digit pertama saja sesuai notebook
    
    # Final DataFrame untuk Prediksi
    X_pred = df[['Term', 'NoEmp', 'log_loan_amt', 'new_business', 'low_doc', 'urban_flag', 'NAICS']]
    
    return df, X_pred

# --- 4. APLIKASI UTAMA ---

def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    lang = st.sidebar.selectbox("Language", ["Bahasa Indonesia", "English"])
    t = lang_dict[lang]
    
    st.title(t["title"])
    
    # Load Models
    models = load_models()
    if not models:
        st.error("❌ Model Files (.pkl) Missing! Upload .pkl files to directory.")
        return

    # --- BAGIAN 1: UPLOAD & BATCH PROCESS ---
    st.header(t["upload_header"])
    st.markdown(t["upload_desc"])
    
    uploaded_file = st.file_uploader(t["upload_label"], type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Baca File
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
        except:
            st.error("Error reading file.")
            return

        # Preprocessing & Validasi
        df_processed, X_pred_or_error = preprocess_input(df_raw)
        
        if isinstance(X_pred_or_error, str): # Jika return string artinya error message
            st.error(X_pred_or_error)
            # Tampilkan contoh format yang benar
            st.info("💡 Format CSV yang diharapkan (Kolom): DisbursementGross, Term, NoEmp, NAICS, NewExist, LowDoc, UrbanRural")
            return
        
        X_pred = X_pred_or_error
        
        # --- JALANKAN MODEL (BATCH) ---
        with st.spinner("🤖 Mengjalankan Model ML pada seluruh data..."):
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
            
            # Download Button
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
            # User memilih baris berdasarkan index atau nama (jika ada kolom Name)
            row_options = df_processed.index.tolist()
            # Buat label yang informatif: "Index 0 - Sektor 23 - $50000"
            format_func = lambda x: f"Row {x} | Sektor {df_processed.iloc[x]['NAICS']} | ${df_processed.iloc[x]['DisbursementGross']:,.0f}"
            
            selected_idx = st.selectbox(t["select_borrower"], row_options, format_func=format_func)
            
            # Ambil Data Baris Terpilih
            row_data = df_processed.iloc[selected_idx]
            
            st.markdown("---")
            
            # Layout Kolom: Kiri (Data Debitur), Kanan (Stress Test Simulator)
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.markdown("### 🏢 Profil Debitur")
                st.info(f"**Sektor Industri:** {row_data['NAICS']}")
                st.write(f"**Pinjaman:** ${row_data['DisbursementGross']:,.2f}")
                st.write(f"**Tenor:** {row_data['Term']} bulan")
                st.write(f"**Karyawan:** {row_data['NoEmp']}")
                st.write(f"**Lokasi:** {'Urban' if row_data['urban_flag']==1 else 'Rural'}")
                
                st.markdown("---")
                st.markdown("### 📊 Risiko Baseline (Saat Ini)")
                st.metric(t["metric_pd"], f"{row_data['PD_Predicted']:.2%}")
                st.metric(t["metric_el"], f"${row_data['Expected_Loss']:,.2f}")

            with col_right:
                st.markdown(f"### 📉 {t['stress_analysis']}")
                st.write(t["stress_insight"])
                
                # --- STRESS TEST CONTROLLER ---
                # Slider VIX
                vix_val = st.slider(t["current_vix"], 10, 80, 15, key="vix_slider")
                
                # Logic Stress Test (Sesuai Script User)
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
                
                # Kalkulasi Dampak pada ROW INI
                pd_base = row_data['PD_Predicted']
                pd_stress = min(pd_base * stress_mult, 1.0)
                
                el_base = row_data['Expected_Loss']
                el_stress = pd_stress * row_data['LGD_Predicted'] * row_data['DisbursementGross']
                delta_el = el_stress - el_base
                
                # Tampilkan Hasil Stress
                st.markdown(f"**Status Ekonomi:** <span style='color:{color}'>{status}</span> (Multiplier: {stress_mult}x)", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Stressed PD", f"{pd_stress:.2%}", delta=f"{(pd_stress-pd_base)*100:.2f}% pts", delta_color="inverse")
                with c2:
                    st.metric("Stressed EL", f"${el_stress:,.2f}", delta=f"-${delta_el:,.2f}", delta_color="inverse")
                
                # Grafik Perbandingan
                fig = go.Figure(data=[
                    go.Bar(name='Normal', x=['Expected Loss'], y=[el_base], marker_color='#2ecc71'),
                    go.Bar(name='Crisis Scenario', x=['Expected Loss'], y=[el_stress], marker_color='#e74c3c')
                ])
                fig.update_layout(title_text=f"Impact of {status} on Selected Borrower", barmode='group', height=300)
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Tampilan Awal (Belum Upload) - Beri User Template
        st.info("👋 Silakan upload file CSV untuk memulai.")
        st.markdown("### 📝 Contoh Format Data (CSV)")
        
        # Bikin Dummy Data buat Contoh
        dummy_data = pd.DataFrame({
            'DisbursementGross': [50000, 120000, 30000],
            'Term': [60, 84, 36],
            'NoEmp': [5, 12, 2],
            'NAICS': ['23', '72', '44'],
            'NewExist': [1, 2, 1],
            'LowDoc': ['N', 'Y', 'N'],
            'UrbanRural': [1, 1, 2]
        })
        st.dataframe(dummy_data)
        st.caption("Pastikan nama kolom di file Anda sesuai atau mirip dengan tabel di atas.")

if __name__ == "__main__":
    main()
