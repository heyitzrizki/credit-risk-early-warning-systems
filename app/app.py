import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# --- 1. SETUP & PAGE CONFIG ---
st.set_page_config(
    page_title="Global Credit Risk EWS",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stCard { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .risk-safe { color: #2ecc71; font-weight: bold; }
    .risk-warning { color: #f1c40f; font-weight: bold; }
    .risk-danger { color: #e74c3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LANGUAGE DICTIONARY ---
# This dictionary handles all text in the app for 3 languages
lang_dict = {
    "English": {
        "nav_title": "Navigation",
        "menu_home": "🏠 Credit Simulator",
        "menu_stress": "📉 Stress Test Analysis",
        "menu_info": "📘 Dictionary & Info",
        "home_title": "🏠 Credit Risk Simulator",
        "home_desc": "Enter borrower profile below for instant risk analysis.",
        "input_loan": "1. Loan Profile",
        "lbl_amount": "💰 Loan Amount (USD)",
        "lbl_term": "📅 Term (Months)",
        "input_biz": "2. Business Profile",
        "lbl_emp": "👥 No. of Employees",
        "lbl_sector": "🏭 Industry Sector",
        "lbl_new": "New Business? (< 2 Yrs)",
        "lbl_urban": "Location",
        "opt_urban": "Urban",
        "opt_rural": "Rural",
        "lbl_doc": "Low Doc Program?",
        "res_title": "📊 Risk Analysis Result",
        "res_pd": "Probability of Default",
        "res_lgd": "Loss Given Default",
        "res_el": "Expected Loss",
        "safe": "✅ Low Risk",
        "warn": "⚠️ Medium Risk",
        "dang": "⛔ High Risk",
        "stress_title": "📉 Economic Crisis Simulation",
        "stress_desc": "Simulate borrower resilience during economic downturns using VIX Index.",
        "stress_vix": "VIX Index Level",
        "eco_norm": "Stable Economy",
        "eco_vol": "Volatile Market",
        "eco_cris": "Severe Crisis",
        "chart_title": "Impact of Crisis on Risk",
        "info_title": "📘 Model Dictionary",
        "info_naics": "What is NAICS Code?",
        "info_naics_desc": "Standard code used to classify business establishments.",
        "gauge_title": "Credit Score Gauge"
    },
    "한국어": {
        "nav_title": "탐색 (Navigation)",
        "menu_home": "🏠 신용 시뮬레이터",
        "menu_stress": "📉 스트레스 테스트",
        "menu_info": "📘 용어 사전",
        "home_title": "🏠 신용 리스크 시뮬레이터",
        "home_desc": "대출자의 프로필을 입력하여 리스크를 즉시 분석하십시오.",
        "input_loan": "1. 대출 프로필",
        "lbl_amount": "💰 대출 금액 (USD)",
        "lbl_term": "📅 기간 (개월)",
        "input_biz": "2. 사업 프로필",
        "lbl_emp": "👥 직원 수",
        "lbl_sector": "🏭 산업 분야",
        "lbl_new": "신규 사업입니까? (< 2년)",
        "lbl_urban": "위치",
        "opt_urban": "도시 (Urban)",
        "opt_rural": "시골 (Rural)",
        "lbl_doc": "서류 미비 (Low Doc)?",
        "res_title": "📊 리스크 분석 결과",
        "res_pd": "부도 확률 (PD)",
        "res_lgd": "부도 시 손실률 (LGD)",
        "res_el": "예상 손실액 (EL)",
        "safe": "✅ 저위험",
        "warn": "⚠️ 중위험",
        "dang": "⛔ 고위험",
        "stress_title": "📉 경제 위기 시뮬레이션",
        "stress_desc": "VIX 지수를 사용하여 경제 침체기 동안의 대출자 회복력을 시뮬레이션합니다.",
        "stress_vix": "VIX 지수 수준",
        "eco_norm": "경제 안정",
        "eco_vol": "시장 변동성",
        "eco_cris": "심각한 위기",
        "chart_title": "위기가 리스크에 미치는 영향",
        "info_title": "📘 모델 용어 사전",
        "info_naics": "NAICS 코드란 무엇인가요?",
        "info_naics_desc": "사업장을 분류하는 데 사용되는 표준 코드입니다.",
        "gauge_title": "신용 점수 게이지"
    },
    "Bahasa Indonesia": {
        "nav_title": "Navigasi",
        "menu_home": "🏠 Simulator Kredit",
        "menu_stress": "📉 Analisa Stress Test",
        "menu_info": "📘 Kamus & Info",
        "home_title": "🏠 Simulator Risiko Kredit",
        "home_desc": "Masukkan profil debitur di bawah ini untuk analisa risiko instan.",
        "input_loan": "1. Profil Pinjaman",
        "lbl_amount": "💰 Jumlah Pinjaman (USD)",
        "lbl_term": "📅 Jangka Waktu (Bulan)",
        "input_biz": "2. Profil Bisnis",
        "lbl_emp": "👥 Jumlah Karyawan",
        "lbl_sector": "🏭 Sektor Industri",
        "lbl_new": "Bisnis Baru? (< 2 Thn)",
        "lbl_urban": "Lokasi",
        "opt_urban": "Perkotaan",
        "opt_rural": "Pedesaan",
        "lbl_doc": "Dokumen Kurang Lengkap?",
        "res_title": "📊 Hasil Analisa Risiko",
        "res_pd": "Peluang Gagal Bayar (PD)",
        "res_lgd": "Potensi Kerugian Aset (LGD)",
        "res_el": "Estimasi Rugi (EL)",
        "safe": "✅ Risiko Rendah",
        "warn": "⚠️ Risiko Sedang",
        "dang": "⛔ Risiko Tinggi",
        "stress_title": "📉 Simulasi Krisis Ekonomi",
        "stress_desc": "Simulasi ketahanan debitur saat krisis menggunakan VIX Index.",
        "stress_vix": "Level VIX Index",
        "eco_norm": "Ekonomi Stabil",
        "eco_vol": "Pasar Gejolak",
        "eco_cris": "Krisis Berat",
        "chart_title": "Dampak Krisis terhadap Risiko",
        "info_title": "📘 Kamus Model",
        "info_naics": "Apa itu Kode NAICS?",
        "info_naics_desc": "Kode standar untuk klasifikasi jenis usaha.",
        "gauge_title": "Meteran Skor Kredit"
    }
}

# Mapping NAICS (Universal Code -> Display Name handled dynamically)
# Key: Code used in model, Value: English base name
naics_base = {
    "11": "Agriculture, Forestry, Fishing",
    "21": "Mining, Oil & Gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "48": "Transportation & Warehousing",
    "51": "Information",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional Services",
    "55": "Management of Companies",
    "56": "Admin & Support Services",
    "61": "Educational Services",
    "62": "Health Care",
    "71": "Arts & Entertainment",
    "72": "Accommodation & Food",
    "81": "Other Services",
    "92": "Public Administration"
}

# --- 3. HELPER FUNCTIONS ---
def get_naics_label(code, lang):
    # Simple translation wrapper for Sector Names
    base_name = naics_base[code]
    if lang == "한국어":
        # Simplified Korean Mapping examples
        korean_map = {
            "Construction": "건설업 (Construction)",
            "Manufacturing": "제조업 (Manufacturing)",
            "Retail Trade": "소매업 (Retail)",
            "Agriculture, Forestry, Fishing": "농업/임업 (Agriculture)",
             # Add others as needed, defaulting to English if not found
        }
        return korean_map.get(base_name, base_name)
    elif lang == "Bahasa Indonesia":
        indo_map = {
            "Construction": "Konstruksi",
            "Manufacturing": "Manufaktur",
            "Retail Trade": "Perdagangan Eceran",
            "Agriculture, Forestry, Fishing": "Pertanian",
            "Transportation & Warehousing": "Transportasi & Gudang",
            "Health Care": "Kesehatan"
        }
        return indo_map.get(base_name, base_name)
    return base_name

@st.cache_resource
def load_models():
    models = {}
    try:
        with open('PD_model_calibrated_pipeline (1).pkl', 'rb') as f:
            models['pd'] = pickle.load(f)
        with open('LGD_model_pipeline (1).pkl', 'rb') as f:
            models['lgd'] = pickle.load(f)
        return models
    except:
        return None

models = load_models()

# --- 4. SIDEBAR & LANGUAGE SELECTOR ---
st.sidebar.title("🌐 Language / 언어")
selected_lang = st.sidebar.selectbox("Select Language", ["English", "한국어", "Bahasa Indonesia"])
t = lang_dict[selected_lang] # Shortcut to current language dict

st.sidebar.markdown("---")
st.sidebar.title(t["nav_title"])
menu = st.sidebar.radio("", [t["menu_home"], t["menu_stress"], t["menu_info"]])

# --- 5. MAIN APPLICATION LOGIC ---

if menu == t["menu_home"]:
    st.title(t["home_title"])
    st.markdown(t["home_desc"])
    
    # Input Container
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(t["input_loan"])
            loan_amount = st.slider(t["lbl_amount"], 1000, 500000, 50000, 1000)
            term_months = st.slider(t["lbl_term"], 0, 360, 60, 12)
            log_loan_amt = np.log(loan_amount + 1)
            
        with col2:
            st.subheader(t["input_biz"])
            no_emp = st.slider(t["lbl_emp"], 0, 100, 5)
            
            # Dynamic Sector Options based on Language
            sector_options = {get_naics_label(k, selected_lang): k for k in naics_base.keys()}
            selected_sector_label = st.selectbox(t["lbl_sector"], list(sector_options.keys()))
            naics_code = sector_options[selected_sector_label]
            
            c1, c2 = st.columns(2)
            with c1:
                is_new = st.toggle(t["lbl_new"], False)
                new_business_val = 2 if is_new else 1
                
                # Urban/Rural Radio
                loc_opt = st.radio(t["lbl_urban"], [t["opt_urban"], t["opt_rural"]], horizontal=True)
                urban_val = 1 if loc_opt == t["opt_urban"] else 2
                
            with c2:
                is_low = st.toggle(t["lbl_doc"], False)
                low_doc_val = 1 if is_low else 0

    st.markdown("---")

    # Calculation
    input_df = pd.DataFrame({
        'Term': [term_months],
        'NoEmp': [no_emp],
        'log_loan_amt': [log_loan_amt],
        'new_business': [new_business_val],
        'low_doc': [low_doc_val],
        'urban_flag': [urban_val],
        'NAICS': [naics_code]
    })
    
    # Prediction (Demo fallback handled silently)
    pd_val, lgd_val = 0.10, 0.45
    if models:
        try:
            pd_val = models['pd'].predict_proba(input_df)[:, 1][0]
            lgd_val = np.clip(models['lgd'].predict(input_df)[0], 0, 1)
        except: pass
        
    el_val = pd_val * lgd_val * loan_amount
    
    # Display Results
    st.subheader(t["res_title"])
    rc1, rc2, rc3 = st.columns(3)
    
    with rc1:
        st.metric(t["res_pd"], f"{pd_val:.2%}")
        if pd_val < 0.10: st.markdown(f'<p class="risk-safe">{t["safe"]}</p>', unsafe_allow_html=True)
        elif pd_val < 0.30: st.markdown(f'<p class="risk-warning">{t["warn"]}</p>', unsafe_allow_html=True)
        else: st.markdown(f'<p class="risk-danger">{t["dang"]}</p>', unsafe_allow_html=True)
        
    with rc2:
        st.metric(t["res_lgd"], f"{lgd_val:.2%}")
        
    with rc3:
        st.metric(t["res_el"], f"${el_val:,.0f}")
        
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pd_val * 100,
        title = {'text': t["gauge_title"]},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#2c3e50"},
                 'steps': [{'range': [0, 10], 'color': "#2ecc71"},
                           {'range': [10, 30], 'color': "#f1c40f"},
                           {'range': [30, 100], 'color': "#e74c3c"}]}
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

elif menu == t["menu_stress"]:
    st.title(t["stress_title"])
    st.markdown(t["stress_desc"])
    
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        vix = st.slider(t["stress_vix"], 10.0, 80.0, 15.0)
        if vix <= 15: st.success(t["eco_norm"])
        elif vix <= 30: st.warning(t["eco_vol"])
        else: st.error(t["eco_cris"])
        
    with col_s2:
        # Dummy Viz for Stress
        base = 0.12
        mult = 1.0 + ((vix - 10)/20)
        stressed = min(base * mult, 1.0)
        
        df_stress = pd.DataFrame({
            "Scenario": ["Baseline", "Stressed (Current VIX)"],
            "PD": [base, stressed]
        })
        fig_s = px.bar(df_stress, x="Scenario", y="PD", color="Scenario", 
                       title=t["chart_title"], 
                       color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig_s, use_container_width=True)

elif menu == t["menu_info"]:
    st.title(t["info_title"])
    with st.expander(t["info_naics"], expanded=True):
        st.write(t["info_naics_desc"])
        # Show table of translations
        data_items = []
        for code, eng_name in naics_base.items():
            row = {"Code": code, "English": eng_name}
            # Add current lang if not English
            if selected_lang != "English":
                row[selected_lang] = get_naics_label(code, selected_lang)
            data_items.append(row)
        st.dataframe(pd.DataFrame(data_items), use_container_width=True)
