import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# 1. PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Enterprise Credit Risk System",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .risk-safe { color: #2ecc71; font-weight: bold; }
    .risk-warning { color: #f1c40f; font-weight: bold; }
    .risk-danger { color: #e74c3c; font-weight: bold; }
    .stDataFrame { border: 1px solid #ddd; border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 2. LANGUAGE DICTIONARY
# =====================================================
lang_dict = {
    "English": {
        "title": "🏢 Enterprise Credit Risk System",
        "upload_header": "📂 1. Upload Dataset",
        "upload_desc": "Upload your borrower data (CSV). The model will predict risk for ALL rows.",
        "upload_label": "Choose CSV File",
        "col_missing": "⚠️ Missing columns in CSV:",
        "sim_header": "🔍 2. Borrower Inspector & Stress Test",
        "select_borrower": "Select Borrower ID / Row:",
        "current_vix": "Market VIX Index:",
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
        "current_vix": "Indeks VIX Pasar:",
        "stress_analysis": "Analisa Stress Test",
        "stress_insight": "Mensimulasikan dampak krisis pada perusahaan ini.",
        "metric_pd": "Peluang Gagal Bayar (PD)",
        "metric_el": "Ekspektasi Kerugian (EL)",
        "tab_data": "📋 Data Lengkap & Prediksi",
        "tab_sim": "🔬 Simulator Individu",
        "download_pred": "Download Hasil Prediksi"
    }
}

# =====================================================
# 3. LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    model_files = {
        'pd': ["PD_model_calibrated_pipeline.pkl", "PD_model_tuned_pipeline.pkl"],
        'lgd': ["LGD_model_pipeline.pkl"]
    }

    models = {}

    for key, file_list in model_files.items():
        loaded = False
        for fname in file_list:
            try:
                with open(fname, 'rb') as f:
                    models[key] = pickle.load(f)
                loaded = True
                break
            except FileNotFoundError:
                continue
        if not loaded:
            st.error(f"❌ Model file missing: {file_list}")
            return None

    return models

# =====================================================
# 4. PREPROCESSING FUNCTION
# =====================================================
def preprocess_input(df_raw):

    df = df_raw.copy()

    # Column alignment
    col_map = {
        'Amount': 'DisbursementGross',
        'LoanAmount': 'DisbursementGross',
        'Time': 'Term',
        'Duration': 'Term',
        'Employees': 'NoEmp',
        'Sector': 'NAICS'
    }
    df.rename(columns=col_map, inplace=True)

    required_cols = ['DisbursementGross', 'Term', 'NoEmp', 'NAICS', 'NewExist', 'LowDoc', 'UrbanRural']
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        return None, f"Missing required columns: {missing}"

    # Feature engineering
    df['log_loan_amt'] = np.log(df['DisbursementGross'] + 1)
    df['new_business'] = df['NewExist']
    df['low_doc'] = df['LowDoc'].apply(lambda x: 1 if str(x).upper() in ['Y', '1'] else 0)
    df['urban_flag'] = df['UrbanRural']
    df['NAICS'] = df['NAICS'].astype(str).str[:2]

    X_pred = df[['Term', 'NoEmp', 'log_loan_amt', 'new_business', 'low_doc', 'urban_flag', 'NAICS']]

    return df, X_pred

# =====================================================
# 5. STREAMLIT UI
# =====================================================
def main():

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    lang = st.sidebar.selectbox("Language", ["Bahasa Indonesia", "English"])
    t = lang_dict[lang]

    st.title(t["title"])

    # Load models
    models = load_models()
    if not models:
        return

    # ------------------------------------
    # UPLOAD DATA
    # ------------------------------------
    st.header(t["upload_header"])
    st.markdown(t["upload_desc"])

    uploaded_file = st.file_uploader(t["upload_label"], type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        except:
            st.error("File reading error")
            return

        df_processed, X_pred_or_error = preprocess_input(df_raw)

        if isinstance(X_pred_or_error, str):
            st.error(X_pred_or_error)
            return

        X_pred = X_pred_or_error

        with st.spinner("Running model..."):
            df_processed['PD_Predicted'] = models['pd'].predict_proba(X_pred)[:, 1]
            df_processed['LGD_Predicted'] = np.clip(models['lgd'].predict(X_pred), 0, 1)
            df_processed['Expected_Loss'] = df_processed['PD_Predicted'] * df_processed['LGD_Predicted'] * df_processed['DisbursementGross']

        st.success(f"Prediction completed for {len(df_processed)} rows.")

        # =====================================================
        # TABS (DATA + SIMULATOR)
        # =====================================================
        tab1, tab2 = st.tabs([t["tab_data"], t["tab_sim"]])

        # TAB 1: DATA TABLE
        with tab1:
            st.dataframe(df_processed, use_container_width=True)
            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(t["download_pred"], csv, "credit_risk_predictions.csv")

        # TAB 2: SIMULATOR
        with tab2:
            st.subheader(t["sim_header"])

            row_options = df_processed.index.tolist()
            format_func = lambda x: f"Row {x} | Sector {df_processed.iloc[x]['NAICS']} | ${df_processed.iloc[x]['DisbursementGross']:,.0f}"

            selected_idx = st.selectbox(t["select_borrower"], row_options, format_func=format_func)
            row_data = df_processed.iloc[selected_idx]

            col_left, col_right = st.columns([1, 2])

            # LEFT: Borrower Profile
            with col_left:
                st.markdown("### 🏢 Borrower Profile")
                st.write(f"**Sector:** {row_data['NAICS']}")
                st.write(f"**Loan Amount:** ${row_data['DisbursementGross']:,.2f}")
                st.write(f"**Term:** {row_data['Term']} months")
                st.write(f"**Employees:** {row_data['NoEmp']}")
                st.write(f"**Urban/Rural:** {'Urban' if row_data['urban_flag']==1 else 'Rural'}")

                st.markdown("---")
                st.markdown("### 📊 Baseline Risk")
                st.metric(t["metric_pd"], f"{row_data['PD_Predicted']:.2%}")
                st.metric(t["metric_el"], f"${row_data['Expected_Loss']:,.2f}")

            # RIGHT: Stress Test
            with col_right:
                st.markdown(f"### 📉 {t['stress_analysis']}")
                st.write(t["stress_insight"])

                vix = st.slider(t["current_vix"], 10, 80, 15)

                if vix <= 15:
                    mult = 1.0; status = "Normal"; color = "green"
                elif vix <= 25:
                    mult = 1.2; status = "Moderate Stress"; color = "orange"
                elif vix <= 35:
                    mult = 1.5; status = "High Stress"; color = "darkorange"
                else:
                    mult = 2.0; status = "Severe Crisis"; color = "red"

                pd_base = row_data['PD_Predicted']
                pd_stress = min(pd_base * mult, 1.0)
                el_base = row_data['Expected_Loss']
                el_stress = pd_stress * row_data['LGD_Predicted'] * row_data['DisbursementGross']
                delta_el = el_stress - el_base

                st.markdown(f"**Status:** <span style='color:{color}'>
