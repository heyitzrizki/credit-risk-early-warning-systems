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
        "currency": "IDR",
        "select_debtor": "Pilih Debitur untuk Inspeksi",
        "risk_classification": "Klasifikasi Risiko PD",
        "debtor_pd_lgd": "PD & LGD Debitur",
        "portfolio_stress_test": "Analisis Stress Test Portofolio",
        "total_el_stressed": "Total Expected Loss (Stressed)",
        "el_impact": "Dampak Krisis (Kenaikan EL)",
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
        "portfolio_stress_test": "Portfolio Stress Test Analysis",
        "total_el_stressed": "Total Expected Loss (Stressed)",
        "el_impact": "Crisis Impact (EL Increase)",
    }
}

# --- Sidebar Setup ---
with st.sidebar:
    st.header("🌐 Language / Bahasa")
    lang_opt = st.selectbox("Select Language", ["ID", "EN"])
    txt = LANG_DICT[lang_opt]
    st.markdown("---")


# --- 2. Pemuatan Model & Konfigurasi ---
@st.cache_resource
def load_all_artifacts():
    """Load all model files stored in the SAME folder as app.py"""

    artifacts = {}

    # 🔥 Fix: absolute directory of THIS file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Full paths to model files
    pd_path = os.path.join(BASE_DIR, "pd_model.pkl")
    lgd_path = os.path.join(BASE_DIR, "lgd_model.pkl")
    config_path = os.path.join(BASE_DIR, "feature_config.json")

    # Debug (remove later if needed)
    st.write("📁 BASE_DIR:", BASE_DIR)
    st.write("📄 Files in BASE_DIR:", os.listdir(BASE_DIR))

    try:
        # Check existence first
        if not os.path.exists(pd_path):
            raise FileNotFoundError(f"pd_model.pkl NOT FOUND at: {pd_path}")

        if not os.path.exists(lgd_path):
            raise FileNotFoundError(f"lgd_model.pkl NOT FOUND at: {lgd_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"feature_config.json NOT FOUND at: {config_path}")

        # Load models
        artifacts["PD"] = joblib.load(pd_path)
        artifacts["LGD"] = joblib.load(lgd_path)

        # Load config JSON
        with open(config_path, "r") as f:
            artifacts["CONFIG"] = json.load(f)

        return artifacts

    except Exception as e:
        st.error(f"❌ Error loading artifacts: {e}")
        return None


# Load artifacts
artifacts = load_all_artifacts()

# Sidebar Status
st.sidebar.subheader(f"🛠️ {txt['sidebar_model']}")

if not artifacts:
    st.sidebar.error(txt["model_missing"])
else:
    st.sidebar.success(txt["success_load"])

st.sidebar.info("Model PD dan LGD dimuat dari `pd_model.pkl` dan `lgd_model.pkl`.")
st.sidebar.markdown("---")


# --- 3. Preprocessing Input Data ---
def preprocess_input(df_raw, config):
    X = df_raw.copy()

    # NAICS 2-digit
    if "NAICS" in X.columns:
        X["NAICS_2"] = X["NAICS"].astype(str).str[:2].replace("", "Un")
        valid_naics = set(config.get("naics_categories", []))
        X["NAICS_2"] = X["NAICS_2"].apply(lambda x: x if x in valid_naics else "Un")
    else:
        X["NAICS_2"] = "Un"

    # Log Loan Amount
    if "DisbursementGross" in X.columns:
        X["DisbursementGross"] = pd.to_numeric(X["DisbursementGross"], errors="coerce").fillna(0)
        X["log_loan_amt"] = np.log1p(X["DisbursementGross"])
    else:
        X["log_loan_amt"] = 0

    # Binary / category flags
    X["new_business"] = X.get("NewExist", 1).apply(lambda x: 1 if x == 2 else 0)
    X["low_doc"] = X.get("LowDoc", "N").apply(lambda x: 1 if str(x).upper() == "Y" else 0)
    X["urban_flag"] = X.get("UrbanRural", 1).apply(lambda x: 1 if x > 0 else 0)

    if "ApprovalFY" not in X.columns:
        X["ApprovalFY"] = config.get("approval_fy_categories", [2000])[0]

    feature_cols = config.get("all_features", [])
    X_processed = X[feature_cols]

    return X_processed, X


# --- Demo Data ---
def generate_demo_data(config):
    naics_opts = config.get("naics_categories", ["33", "44", "51"])
    fy_opts = config.get("approval_fy_categories", [2010, 2015, 2020])

    return pd.DataFrame({
        "Name": ["PT. Cahaya Abadi", "CV. Digital Cepat", "UD. Makmur Jaya", "Koperasi Sejahtera"],
        "NAICS": [np.random.choice(naics_opts) + str(np.random.randint(10, 99)), "541511", "311812", "448130"],
        "ApprovalFY": np.random.choice(fy_opts, 4),
        "Term": np.random.randint(6, 60, 4),
        "NoEmp": np.random.randint(5, 50, 4),
        "NewExist": [1, 2, 1, 1],
        "UrbanRural": [1, 2, 1, 1],
        "LowDoc": ["N", "Y", "N", "N"],
        "DisbursementGross": np.random.randint(100000000, 500000000, 4)
    })


# --- Stress Test PD ---
def apply_vix_stress(pd_baseline, vix_index):
    multiplier = 1 + (np.maximum(0, vix_index - 20) / 100) * 1.5
    return np.minimum(pd_baseline * multiplier, 1.0)


# --- 5. UI Main Page ---
st.title(txt["title"])
st.markdown(f"**{txt['subtitle']}**")

st.header(f"1. {txt['upload_header']}")

uploaded_file = st.file_uploader(txt["upload_label"], type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        st.session_state["df_raw"] = df_raw
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.session_state.pop("df_raw", None)

else:
    if st.button(txt["demo_data"]):
        if artifacts:
            st.session_state["df_raw"] = generate_demo_data(artifacts["CONFIG"])
        else:
            st.warning("Tidak bisa membuat data demo tanpa file konfigurasi.")

# --- 6. Run Analysis ---
if "df_raw" in st.session_state and artifacts:
    df_raw = st.session_state["df_raw"]
    st.subheader("Preview Data Input")
    st.dataframe(df_raw.head())

    if st.button(txt["run_analysis"]):
        try:
            X_model_ready, df_feats = preprocess_input(df_raw, artifacts["CONFIG"])

            pd_pred = artifacts["PD"].predict_proba(X_model_ready)[:, 1]
            lgd_pred = artifacts["LGD"].predict(X_model_ready)
            lgd_pred = np.clip(lgd_pred, 0, 1)

            df_results = df_raw.copy()
            df_results[txt["col_pd"]] = pd_pred
            df_results[txt["col_lgd"]] = lgd_pred
            df_results["DisbursementGross"] = df_feats["DisbursementGross"]
            df_results[txt["col_el"]] = pd_pred * lgd_pred * df_feats["DisbursementGross"]

            st.session_state["results"] = df_results
            st.success("✅ Analisis selesai!")

        except Exception as e:
            st.error(f"❌ Kesalahan Prediksi Model: {e}")
            st.warning("Pastikan fitur input sesuai dengan training model.")

# --- 7. Display Results ---
if "results" in st.session_state:
    results = st.session_state["results"]

    st.header("2. Hasil Analisis Risiko")
    tab1, tab2 = st.tabs([txt["tab1"], txt["tab2"]])

    # --- TAB 1 ---
    with tab1:

        st.subheader(txt["portfolio_stress_test"])
        vix_val = st.slider(txt["stress_vix"], 10, 80, 20, 1)

        st.markdown(f"*{txt['stress_desc']}*")

        results["PD_Stressed"] = apply_vix_stress(results[txt["col_pd"]], vix_val)
        results["EL_Stressed"] = results["PD_Stressed"] * results[txt["col_lgd"]] * results["DisbursementGross"]

        total_el = results[txt["col_el"]].sum()
        stressed_el = results["EL_Stressed"].sum()
        avg_pd = results[txt["col_pd"]].mean()
        avg_lgd = results[txt["col_lgd"]].mean()
        impact = stressed_el - total_el

        st.subheader(txt["portfolio_summary"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(txt["total_el"], f"{txt['currency']} {total_el:,.0f}")
        c2.metric(txt["avg_pd"], f"{avg_pd:.2%}")
        c3.metric(txt["total_el_stressed"], f"{txt['currency']} {stressed_el:,.0f}")
        c4.metric(txt["el_impact"], f"{txt['currency']} {impact:,.0f}", delta=f"{impact:,.0f}")

        st.markdown("---")

        df_show = results[[
            "Name", "DisbursementGross",
            txt["col_pd"], "PD_Stressed",
            txt["col_lgd"],
            txt["col_el"], "EL_Stressed"
        ]]

        st.dataframe(df_show.style.format({
            txt["col_pd"]: "{:.2%}",
            "PD_Stressed": "{:.2%}",
            txt["col_lgd"]: "{:.2%}",
            txt["col_el"]: f"{txt['currency']} {{:,.0f}}",
            "EL_Stressed": f"{txt['currency']} {{:,.0f}}",
            "DisbursementGross": f"{txt['currency']} {{:,.0f}}"
        }))

        # Download
        st.download_button(
            txt["download_btn"],
            results.to_csv(index=False).encode("utf-8"),
            "credit_risk_results.csv"
        )

    # --- TAB 2 ---
    with tab2:
        st.subheader(txt["risk_profile"])

        debtors = results["Name"].unique().tolist()
        debtor = st.selectbox(txt["select_debtor"], debtors)

        d = results[results["Name"] == debtor].iloc[0]
        base_pd = d[txt["col_pd"]]
        base_lgd = d[txt["col_lgd"]]
        base_el = d[txt["col_el"]]
        exposure = d["DisbursementGross"]

        st.subheader(txt["stress_vix"])
        vix_d = st.slider("VIX Debitur", 10, 80, vix_val, 1)
        stressed_pd = apply_vix_stress(base_pd, vix_d)
        stressed_el = stressed_pd * base_lgd * exposure

        c1, c2, c3 = st.columns(3)
        c1.metric("PD (Stressed)", f"{stressed_pd:.2%}", delta=f"{(stressed_pd - base_pd):.2%}")
        c2.metric("LGD", f"{base_lgd:.2%}")
        c3.metric("EL (Stressed)", f"{txt['currency']} {stressed_el:,.0f}", delta=f"{(stressed_el - base_el):,.0f}")

        st.subheader(txt["base_vs_stress"])

        df_bar = pd.DataFrame({
            "Scenario": ["Baseline", f"Stressed (VIX {vix_d})"],
            "EL": [base_el, stressed_el]
        })

        fig = px.bar(df_bar, x="Scenario", y="EL", text="EL")
        fig.update_traces(texttemplate=f"{txt['currency']} %{{y:,.0f}}")
        st.plotly_chart(fig, use_container_width=True)

# End of app
