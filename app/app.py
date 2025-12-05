import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import plotly.graph_objects as go

# ================================================================
# FIX PATH (ensure Streamlit finds models inside app folder)
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def p(x): return os.path.join(BASE_DIR, x)

# ================================================================
# LOAD MODELS (Pipeline models — includes preprocessing)
# ================================================================
@st.cache_resource
def load_models():
    try:
        pd_model = joblib.load(p("PD_pipeline.pkl"))
        lgd_model = joblib.load(p("LGD_pipeline.pkl"))
        shap_exp = joblib.load(p("pd_shap_explainer.pkl"))
        return pd_model, lgd_model, shap_exp
    except Exception as e:
        st.error(f"❌ Model Loading Error: {e}")
        return None, None, None

pd_model, lgd_model, shap_explainer = load_models()

# ================================================================
# HUMAN-FRIENDLY NAICS LABELS
# ================================================================
NAICS_MAP = {
    "11": "11 — Agriculture, Forestry, Fishing & Hunting",
    "21": "21 — Mining, Quarrying, Oil & Gas",
    "22": "22 — Utilities",
    "23": "23 — Construction",
    "31": "31 — Manufacturing (A)",
    "32": "32 — Manufacturing (B)",
    "33": "33 — Manufacturing (C)",
    "42": "42 — Wholesale Trade",
    "44": "44 — Retail Trade (A)",
    "45": "45 — Retail Trade (B)",
    "48": "48 — Transportation",
    "49": "49 — Warehousing",
    "51": "51 — Information Services",
    "52": "52 — Finance & Insurance",
    "53": "53 — Real Estate & Rental",
    "54": "54 — Professional Services",
    "55": "55 — Management Companies",
    "56": "56 — Administrative Support",
    "61": "61 — Education Services",
    "62": "62 — Health Care",
    "71": "71 — Arts & Entertainment",
    "72": "72 — Accommodation & Food",
    "81": "81 — Other Services",
    "92": "92 — Public Administration",
    "Un": "Unknown Industry"
}

NAICS_OPTIONS = list(NAICS_MAP.values())

def extract_naics_code(label):
    return label.split(" — ")[0]

# ================================================================
# STRESS TEST FUNCTION
# ================================================================
def apply_stress(pd_value, severity):
    """
    severity: 'Mild', 'Moderate', 'Severe'
    """
    factor = {"Mild": 1.1, "Moderate": 1.25, "Severe": 1.5}[severity]
    stressed = min(pd_value * factor, 1.0)
    return stressed

# ================================================================
# GAUGE CHART
# ================================================================
def draw_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 48}},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "orange"},
            "steps": [
                {"range": [0, 0.2], "color": "#138a36"},
                {"range": [0.2, 0.5], "color": "#f1c40f"},
                {"range": [0.5, 1], "color": "#e74c3c"},
            ],
        },
    ))
    fig.update_layout(height=350, margin=dict(l=30, r=30, t=30, b=30))
    return fig

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Credit Risk Dashboard",
    layout="wide",
)

st.title("🏦 Corporate Credit Risk Dashboard — PD, LGD, Expected Loss")
st.write("Human-friendly dashboard for non-technical financial decision makers.")

# ================================================================
# INPUT SECTION
# ================================================================
st.subheader("📌 Borrower Information")

col1, col2 = st.columns(2)

with col1:
    term = st.number_input("Loan Term (Months)", min_value=1, value=60)
    employees = st.number_input("Number of Employees", min_value=0, value=10)
    amount = st.number_input("Loan Amount (USD)", min_value=1.0, value=5000.0)
    naics_label = st.selectbox("Industry Sector (NAICS)", NAICS_OPTIONS)

with col2:
    new_business = st.radio("Is Borrower Newly Established?", ["No", "Yes"])
    low_doc = st.radio("Low Documentation Loan?", ["No", "Yes"])
    urban = st.radio("Urban Area?", ["No", "Yes"])
    approval_year = st.selectbox("Loan Approval Year", list(range(1960, 2025)))

# Convert to model inputs (pipeline will handle preprocessing)
input_data = {
    "Term": term,
    "NoEmp": employees,
    "log_loan_amt": np.log(amount),
    "new_business": 1 if new_business == "Yes" else 0,
    "low_doc": 1 if low_doc == "Yes" else 0,
    "urban_flag": 1 if urban == "Yes" else 0,
    "NAICS_2": extract_naics_code(naics_label),
    "ApprovalFY": int(approval_year),
}

# ================================================================
# RUN PREDICTION
# ================================================================
if st.button("Run Prediction"):
    if pd_model is None:
        st.error("Models failed to load.")
    else:
        df = pd.DataFrame([input_data])

        pd_pred = float(pd_model.predict_proba(df)[0][1])
        lgd_pred = float(lgd_model.predict(df)[0])
        el_pred = pd_pred * lgd_pred * amount  # monetary expected loss

        st.subheader("📊 Risk Level")

        gauge = draw_gauge(pd_pred)
        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("📈 Risk Assessment Results")

        colA, colB = st.columns(2)

        with colA:
            st.write(f"**Probability of Default (PD):** {pd_pred:.2%}")
            st.write(f"**Loss Given Default (LGD):** {lgd_pred:.2f}")
            st.write(f"**Expected Loss (EL):** **${el_pred:,.2f}**")

        # Stress Scenarios
        with colB:
            st.write("### 🔥 Stress Scenarios")

            for sev in ["Mild", "Moderate", "Severe"]:
                stressed_pd = apply_stress(pd_pred, sev)
                stressed_el = stressed_pd * lgd_pred * amount

                st.write(f"**{sev} Stress PD:** {stressed_pd:.2%}")
                st.write(f"Expected Loss under {sev}: **${stressed_el:,.2f}**")

        # SHAP Explainability
        st.subheader("🔍 SHAP Explainability (PD Model)")

        try:
            X_processed = pd_model[:-1].transform(df).toarray()
            shap_vals = shap_explainer.shap_values(X_processed)

            st.write("### Summary Plot")
            st.pyplot(shap.summary_plot(shap_vals, X_processed, show=False))

            st.write("### Top Features")
            mean_abs = np.abs(shap_vals).mean(axis=0)
            idx = np.argsort(mean_abs)[::-1][:10]
            st.write(pd.DataFrame({
                "Feature": idx,
                "Impact": mean_abs[idx]
            }))
        except:
            st.info("SHAP could not be generated for this input.")

