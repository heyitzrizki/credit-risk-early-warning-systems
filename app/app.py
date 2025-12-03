import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import json
import io

# --- 1. Konfigurasi Halaman dan Lokalisasi ---
st.set_page_config(page_title="Enterprise Credit Risk EWS", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

LANG_DICT = {
    "ID": {
        "title": "Sistem Peringatan Dini Risiko Kredit",
        "subtitle": "Analisis Probabilitas Gagal Bayar (PD) & Stress Testing Ekonomi Makro",
        "upload_header": "Unggah Data Portofolio",
        "upload_label": "Unggah file CSV atau Excel (Data Peminjam)",
        "sidebar_settings": "Pengaturan",
        "sidebar_model": "Status Model & File",
        "tab1": "📋 Laporan Portofolio",
        "tab2": "⚡ Inspektor & Stress Test",
        "col_pd": "Probabilitas Default (PD)",
        "col_lgd": "Loss Given Default (LGD)",
        "col_el": "Expected Loss (EL)",
        "stress_vix": "Skenario Krisis (VIX Index)",
        "stress_desc": "Geser untuk mensimulasikan volatilitas pasar. VIX > 30 menandakan ketidakpastian tinggi.",
        "download_btn": "Unduh Hasil Analisis",
        "model_missing": "Model/config tidak ditemukan. Pastikan semua file .pkl dan .json ada.",
        "success_load": "Model dan Konfigurasi berhasil dimuat!",
        "risk_profile": "Profil Risiko Debitur",
        "base_vs_stress": "Perbandingan EL: Normal vs Krisis",
        "run_analysis": "🚀 Jalankan Analisis AI",
        "demo_data": "Buat Data Demo (Simulasi)",
        "portfolio_summary": "Ringkasan Portofolio",
        "total_el": "Total Expected Loss Portofolio (Baseline)",
        "avg_pd": "Rata-rata PD Portofolio (Baseline)",
        "avg_lgd": "Rata-rata LGD Portofolio",
        "currency": "IDR", # Asumsi mata uang
        "select_debtor": "Pilih Debitur untuk Inspeksi",
        "risk_classification": "Klasifikasi Risiko PD",
        "debtor_pd_lgd": "PD & LGD Debitur",
        "portfolio_stress_test": "Analisis Stress Test Portofolio", # Tambahan
        "total_el_stressed": "Total Expected Loss (Stressed)", # Tambahan
        "el_impact": "Dampak Krisis (Kenaikan EL)", # Tambahan
    },
    "EN": {
        "title": "Enterprise Credit Risk EWS",
        "subtitle": "Probability of Default (PD) Analysis & Macro Stress Testing",
        "upload_header": "Upload Portfolio Data",
        "upload_label": "Upload CSV or Excel file (Borrower Data)",
        "sidebar_settings": "Settings",
        "sidebar_model": "Model Status & Files",
        "tab1": "📋 Portfolio Report",
        "tab2": "⚡ Inspector & Stress Test",
        "col_pd": "Probability of Default (PD)",
        "col_lgd": "Loss Given Default (LGD)",
        "col_el": "Expected Loss (EL)",
        "stress_vix": "Crisis Scenario (VIX Index)",
        "stress_desc": "Slide to simulate market volatility. VIX > 30 indicates high uncertainty.",
        "download_btn": "Download Analysis Results",
        "model_missing": "Model/config not found. Ensure all .pkl and .json files exist.",
        "success_load": "Models and Configuration loaded successfully!",
        "risk_profile": "Debtor Risk Profile",
        "base_vs_stress": "EL Comparison: Baseline vs Stressed",
        "run_analysis": "🚀 Run AI Analysis",
        "demo_data": "Generate Demo Data (Simulation)",
        "portfolio_summary": "Portfolio Summary",
        "total_el": "Total Portfolio Expected Loss (Baseline)",
        "avg_pd": "Average Portfolio PD (Baseline)",
        "avg_lgd": "Average Portfolio LGD",
        "currency": "USD",
        "select_debtor": "Select Debtor for Inspection",
        "risk_classification": "PD Risk Classification",
        "debtor_pd_lgd": "Debtor PD & LGD",
        "portfolio_stress_test": "Portfolio Stress Test Analysis", # Tambahan
        "total_el_stressed": "Total Expected Loss (Stressed)", # Tambahan
        "el_impact": "Crisis Impact (EL Increase)", # Tambahan
    }
}

# Inisialisasi Sidebar
with st.sidebar:
    st.header("🌐 Language / Bahasa")
    lang_opt = st.selectbox("Select Language", ["ID", "EN"])
    txt = LANG_DICT[lang_opt]
    st.markdown("---")

# --- 2. Pemuatan Model dan Konfigurasi ---
@st.cache_resource
def load_all_artifacts():
    """Memuat semua model dan konfigurasi yang diperlukan.
    Model dicari di direktori yang sama dengan app.py."""
    artifacts = {}
    
    # Path file (gunakan nama file yang diunggah)
    pd_path = "pd_model.pkl"
    lgd_path = "lgd_model.pkl"
    config_path = "feature_config.json"
    
    # Periksa apakah file tersedia 
    try:
        # PD Model
        artifacts['PD'] = joblib.load(pd_path)
        
        # LGD Model
        artifacts['LGD'] = joblib.load(lgd_path)
        
        # Config
        with open(config_path, 'r') as f:
            artifacts['CONFIG'] = json.load(f)

        if 'PD' in artifacts and 'LGD' in artifacts and 'CONFIG' in artifacts:
            return artifacts
        return None
    except Exception as e:
        # Tangani error jika file tidak ditemukan
        st.error(f"Error saat memuat model: Pastikan semua file berada dalam folder yang sama. Detail: {e}")
        return None

# Muat model
artifacts = load_all_artifacts()

# Tampilkan status model di sidebar
st.sidebar.subheader(f"🛠️ {txt['sidebar_model']}")
if not artifacts:
    st.sidebar.error(txt['model_missing'])
else:
    st.sidebar.success(txt['success_load'])

st.sidebar.info("Model PD dan LGD dimuat dari `pd_model.pkl` dan `lgd_model.pkl`.")
st.sidebar.markdown("---")


# --- 3. Fungsi Preprocessing Data ---
def preprocess_input(df_raw, config):
    """
    Membuat fitur turunan yang diperlukan oleh model dari data input mentah.
    Ini harus mencerminkan langkah rekayasa fitur sebelum training model.
    """
    X = df_raw.copy()

    # 1. Pastikan kolom ada dan bersih
    if 'NAICS' in X.columns:
        # NAICS 2-digit: Asumsi bahwa NAICS adalah string, ambil 2 digit pertama
        X['NAICS_2'] = X['NAICS'].astype(str).str[:2].replace('', 'Un')
        # Handle NAICS yang tidak ada di kategori training (misal, set ke 'Un')
        valid_naics = set(config.get('naics_categories', []))
        X['NAICS_2'] = X['NAICS_2'].apply(lambda x: x if x in valid_naics else 'Un')
    else:
        # Buat kolom placeholder jika tidak ada di data demo
        X['NAICS_2'] = 'Un'

    # 2. Fitur Numerik (Log Transform)
    if 'DisbursementGross' in X.columns:
        # Gunakan nama kolom yang sudah ada. Jika 'DisbursementGross' adalah string, konversi ke float terlebih dahulu.
        X['DisbursementGross'] = pd.to_numeric(X['DisbursementGross'], errors='coerce').fillna(0)
        X['log_loan_amt'] = np.log1p(X['DisbursementGross'])
    else:
        X['log_loan_amt'] = 0 # Placeholder

    # 3. Fitur Biner/Kategori
    X['new_business'] = X.get('NewExist', 1).apply(lambda x: 1 if x == 2 else 0)
    X['low_doc'] = X.get('LowDoc', 'N').apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
    # Asumsi UrbanRural = 1 atau 2 (Urban/Rural), 0 (Tidak diketahui)
    X['urban_flag'] = X.get('UrbanRural', 1).apply(lambda x: 1 if x > 0 else 0)
    
    # 4. Fitur Tambahan yang MUNGKIN dibutuhkan model (meskipun tidak dipakai di demo data)
    # Jika model Anda membutuhkan 'ApprovalFY' (contoh dari config JSON), ini perlu ditambahkan.
    if 'ApprovalFY' not in X.columns:
        # Tambahkan nilai default yang valid sesuai kategori training (misalnya, nilai pertama)
        X['ApprovalFY'] = config.get('approval_fy_categories', [2000])[0]

    # Pilih hanya kolom yang dibutuhkan oleh Pipeline
    feature_cols = config.get('all_features', [])
    X_processed = X[feature_cols]

    return X_processed, X

# --- 4. Fungsi Pembuatan Data Demo ---
def generate_demo_data(config):
    """Membuat DataFrame demo yang lebih sesuai dengan fitur model."""
    naics_opts = config.get('naics_categories', ['33', '44', '51'])
    fy_opts = config.get('approval_fy_categories', [2010, 2015, 2020])
    
    return pd.DataFrame({
        'Name': ['PT. Cahaya Abadi', 'CV. Digital Cepat', 'UD. Makmur Jaya', 'Koperasi Sejahtera'],
        'NAICS': [np.random.choice(naics_opts) + str(np.random.randint(10, 99)), '541511', '311812', '448130'],
        'ApprovalFY': np.random.choice(fy_opts, 4),
        'Term': np.random.randint(6, 60, 4),
        'NoEmp': np.random.randint(5, 50, 4),
        'NewExist': [1, 2, 1, 1], # 1: Existing, 2: New Business
        'UrbanRural': [1, 2, 1, 1], # 1: Urban, 2: Rural
        'LowDoc': ['N', 'Y', 'N', 'N'],
        'DisbursementGross': np.random.randint(100000000, 500000000, 4)
    })

# --- FUNGSI BARU: Stress Testing PD ---
def apply_vix_stress(pd_baseline, vix_index):
    """
    Menghitung PD yang distres berdasarkan VIX Index.
    """
    # 20 adalah VIX normal/baseline
    # 1.5 adalah sensitivitas PD terhadap VIX (beta)
    multiplier = 1 + (np.maximum(0, vix_index - 20) / 100) * 1.5
    
    # Aplikasikan multiplier ke PD baseline, batasi maksimum 1.0 (100%)
    pd_stressed = np.minimum(pd_baseline * multiplier, 1.0)
    return pd_stressed

# --- 5. Tampilan Utama Aplikasi ---

st.title(txt['title'])
st.markdown(f"**{txt['subtitle']}**")

st.header(f"1. {txt['upload_header']}")

# Input Data
uploaded_file = st.file_uploader(txt['upload_label'], type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        st.session_state['df_raw'] = df_raw
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        del st.session_state['df_raw']
else:
    # Tombol Generate Demo Data
    if st.button(txt['demo_data']):
        if artifacts:
            st.session_state['df_raw'] = generate_demo_data(artifacts['CONFIG'])
        else:
            st.warning("Tidak bisa membuat data demo tanpa file konfigurasi.")

# --- 6. Jalankan Analisis & Prediksi ---
if 'df_raw' in st.session_state and artifacts:
    df_raw = st.session_state['df_raw']
    st.subheader("Preview Data Input")
    st.dataframe(df_raw.head())

    if st.button(txt['run_analysis']):
        try:
            # 1. Preprocessing/Feature Engineering
            X_model_ready, df_with_features = preprocess_input(df_raw, artifacts['CONFIG'])

            # 2. Prediksi
            # PD: Probabilitas gagal bayar (kelas 1)
            pd_pred = artifacts['PD'].predict_proba(X_model_ready)[:, 1]
            # LGD: Loss Given Default
            lgd_pred = artifacts['LGD'].predict(X_model_ready)
            
            # Pastikan LGD dalam rentang [0, 1] jika model regresi tidak membatasi
            lgd_pred = np.clip(lgd_pred, 0, 1)

            # 3. Hitung Expected Loss (EL = PD * LGD * Exposure)
            df_results = df_raw.copy()
            df_results[txt['col_pd']] = pd_pred
            df_results[txt['col_lgd']] = lgd_pred
            df_results['DisbursementGross'] = df_with_features['DisbursementGross'] # Gunakan DisbursementGross yang sudah dinumerik
            df_results[txt['col_el']] = df_results[txt['col_pd']] * df_results[txt['col_lgd']] * df_results['DisbursementGross']
            
            st.session_state['results'] = df_results
            st.success("✅ Analisis Risiko Kredit Selesai!")
        
        except Exception as e:
            st.error(f"❌ Kesalahan Prediksi Model: {e}")
            st.warning("Pastikan data input mengandung semua kolom yang dibutuhkan model dan tipe datanya benar.")

# --- 7. Tampilkan Hasil ---
if 'results' in st.session_state:
    results = st.session_state['results']
    
    st.header(f"2. Hasil Analisis Risiko")
    tab1, tab2 = st.tabs([txt['tab1'], txt['tab2']])

    # --- TAB 1: Laporan Portofolio ---
    with tab1:
        
        # Slider VIX Index untuk Stress Test diterapkan di sini untuk seluruh portofolio
        st.subheader(txt['portfolio_stress_test'])
        vix_portofolio = st.slider(txt['stress_vix'], min_value=10, max_value=80, value=20, step=1, key='vix_portfolio')
        st.markdown(f"*{txt['stress_desc']}*")
        
        # Hitung Stressed PD dan EL untuk SELURUH Portofolio
        results['PD_Stressed'] = apply_vix_stress(results[txt['col_pd']], vix_portofolio)
        results['EL_Stressed'] = results['PD_Stressed'] * results[txt['col_lgd']] * results['DisbursementGross']
        
        # Hitung Metrik Ringkasan
        total_el_baseline = results[txt['col_el']].sum()
        total_el_stressed = results['EL_Stressed'].sum()
        avg_pd = results[txt['col_pd']].mean()
        avg_lgd = results[txt['col_lgd']].mean()
        el_impact = total_el_stressed - total_el_baseline
        
        st.markdown("---")
        st.subheader(txt['portfolio_summary'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            label=txt['total_el'], 
            value=f"{txt['currency']} {total_el_baseline:,.0f}"
        )
        col2.metric(
            label=txt['avg_pd'], 
            value=f"{avg_pd:.2%}"
        )
        col3.metric(
            label=txt['total_el_stressed'], 
            value=f"{txt['currency']} {total_el_stressed:,.0f}"
        )
        col4.metric(
            label=txt['el_impact'], 
            value=f"{txt['currency']} {el_impact:,.0f}",
            delta=f"{el_impact:,.0f}"
        )
        
        st.markdown("---")
        
        # Plot Perbandingan Total EL
        el_comparison_data = pd.DataFrame({
            'Skenario': ['Baseline', f'Stressed (VIX={vix_portofolio})'], 
            'Total Expected Loss': [total_el_baseline, total_el_stressed]
        })
        
        fig_portfolio_el = px.bar(
            el_comparison_data, 
            x='Skenario', 
            y='Total Expected Loss', 
            text='Total Expected Loss',
            color='Skenario',
            color_discrete_map={'Baseline': '#1f77b4', f'Stressed (VIX={vix_portofolio})': '#ff7f0e'},
            title="Total Expected Loss: Baseline vs. Skenario Krisis"
        )
        fig_portfolio_el.update_traces(
            texttemplate=f"{txt['currency']} %{{y:,.0f}}", 
            textposition='outside'
        )
        fig_portfolio_el.update_layout(yaxis_title=f"Total Expected Loss ({txt['currency']})")
        st.plotly_chart(fig_portfolio_el, use_container_width=True)

        st.subheader(f"Data Hasil (PD, LGD, EL, PD Stressed, EL Stressed)")
        # Tampilkan DataFrame dengan format yang lebih baik
        st.dataframe(
            results[[
                'Name', 'DisbursementGross', 
                txt['col_pd'], 'PD_Stressed', 
                txt['col_lgd'], 
                txt['col_el'], 'EL_Stressed'
            ]].style.format({
                txt['col_pd']: "{:.2%}",
                'PD_Stressed': "{:.2%}",
                txt['col_lgd']: "{:.2%}",
                txt['col_el']: f"{txt['currency']} {{:,.0f}}",
                'EL_Stressed': f"{txt['currency']} {{:,.0f}}",
                'DisbursementGross': f"{txt['currency']} {{:,.0f}}"
            })
        )

        # Download Button
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=txt['download_btn'],
            data=csv,
            file_name='credit_risk_analysis_results.csv',
            mime='text/csv',
        )

    # --- TAB 2: Inspektor & Stress Test ---
    with tab2:
        # Pilihan Debitur (Gunakan data terbaru yang sudah ada kolom PD_Stressed dan EL_Stressed)
        st.subheader(txt['risk_profile'])
        debtor_names = results['Name'].unique().tolist()
        debtor = st.selectbox(txt['select_debtor'], debtor_names, key='debtor_select_tab2')
        
        # Ambil data debitur yang dipilih
        d = results[results['Name'] == debtor].iloc[0]
        base_pd = d[txt['col_pd']]
        base_lgd = d[txt['col_lgd']]
        base_el = d[txt['col_el']]
        exposure = d['DisbursementGross']

        st.markdown("---")
        st.subheader(txt['stress_vix'])
        st.markdown(f"*{txt['stress_desc']}*")
        
        # Slider VIX Index untuk Stress Test Debitur (Gunakan key berbeda)
        # Nilai awal diset ke VIX Portofolio
        vix_debtor = st.slider(txt['stress_vix'], min_value=10, max_value=80, value=vix_portofolio, step=1, key='vix_debtor')
        
        # Hitung Stressed PD dan EL UNTUK DEBITUR INI
        stressed_pd = apply_vix_stress(base_pd, vix_debtor)
        stressed_el = stressed_pd * base_lgd * exposure
        
        # Tampilkan metrik untuk debitur yang dipilih
        col4, col5, col6 = st.columns(3)
        col4.metric(label=f"PD ({vix_debtor=})", value=f"{stressed_pd:.2%}", delta=f"{(stressed_pd-base_pd):.2%}")
        col5.metric(label="LGD (Fixed)", value=f"{base_lgd:.2%}")
        col6.metric(label=f"EL ({vix_debtor=})", value=f"{txt['currency']} {stressed_el:,.0f}", delta=f"{(stressed_el-base_el):,.0f}", delta_color="inverse")

        st.markdown("---")
        st.subheader(txt['base_vs_stress'])

        # Plot Perbandingan EL Debitur
        el_data = pd.DataFrame({
            'Skenario': ['Baseline', f'Stressed (VIX={vix_debtor})'], 
            'Expected Loss': [base_el, stressed_el]
        })
        
        fig_el = px.bar(
            el_data, 
            x='Skenario', 
            y='Expected Loss', 
            text='Expected Loss', 
            title=f"Perbandingan EL untuk {debtor}"
        )
        fig_el.update_traces(
            texttemplate=f"{txt['currency']} %{{y:,.0f}}", 
            textposition='outside'
        )
        fig_el.update_layout(yaxis_title=f"Expected Loss ({txt['currency']})", uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_el, use_container_width=True)

        # Plot PD & LGD
        st.subheader(txt['debtor_pd_lgd'])
        fig_pd_lgd = go.Figure(data=[
            go.Bar(name='PD', x=['PD'], y=[stressed_pd], text=f"{stressed_pd:.2%}", marker_color='#1f77b4'),
            go.Bar(name='LGD', x=['LGD'], y=[base_lgd], text=f"{base_lgd:.2%}", marker_color='#ff7f0e')
        ])
        fig_pd_lgd.update_layout(
            title=f"PD & LGD Debitur",
            yaxis_title="Persentase",
            yaxis_tickformat=".0%",
            barmode='group'
        )
        st.plotly_chart(fig_pd_lgd, use_container_width=True)


# Tampilkan pesan jika model/data belum siap
if not artifacts:
    st.info("Harap pastikan semua file model (.pkl) dan konfigurasi (.json) telah diunggah dan dimuat dengan benar.")
elif 'df_raw' not in st.session_state:
    st.info(f"Silakan unggah data portofolio Anda atau klik '{txt['demo_data']}' untuk memulai analisis.")
