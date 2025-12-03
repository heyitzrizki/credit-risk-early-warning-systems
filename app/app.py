import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Early Warning System",
    page_icon="🏦",
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_models():
    """Load all deployment artifacts"""
    try:
        artifact_dir = "deployment_artifacts"
        
        pd_model = joblib.load(os.path.join(artifact_dir, "pd_model_pipeline.pkl"))
        lgd_model = joblib.load(os.path.join(artifact_dir, "lgd_model_pipeline.pkl"))
        feature_eng_func = joblib.load(os.path.join(artifact_dir, "feature_engineering_function.pkl"))
        pd_features = joblib.load(os.path.join(artifact_dir, "pd_features.pkl"))
        lgd_features = joblib.load(os.path.join(artifact_dir, "lgd_features.pkl"))
        
        st.success("✓ Models loaded successfully!")
        return pd_model, lgd_model, feature_eng_func, pd_features, lgd_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models
pd_model, lgd_model, create_features, pd_features, lgd_features = load_models()

# Title and header
st.title("🏦 Credit Risk Early Warning System")
st.markdown("### Small Business Loan Default Prediction")
st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Loan Information")
    
    term = st.number_input(
        "Loan Term (months)",
        min_value=0,
        max_value=600,
        value=120,
        help="Duration of the loan in months"
    )
    
    loan_amt = st.number_input(
        "Loan Amount ($)",
        min_value=0.0,
        max_value=10000000.0,
        value=50000.0,
        step=1000.0,
        help="Total loan disbursement amount"
    )
    
    naics = st.text_input(
        "NAICS Code",
        value="541110",
        help="North American Industry Classification System code"
    )
    
    approval_fy = st.selectbox(
        "Approval Fiscal Year",
        options=['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'],
        index=10,
        help="Year when the loan was approved"
    )

with col2:
    st.subheader("🏢 Business Information")
    
    noemp = st.number_input(
        "Number of Employees",
        min_value=0,
        max_value=10000,
        value=5,
        help="Total number of employees in the business"
    )
    
    new_exist = st.selectbox(
        "Business Type",
        options=[1, 2],
        format_func=lambda x: "Existing Business" if x == 1 else "New Business",
        help="Whether the business is new or existing"
    )
    
    urban_rural = st.selectbox(
        "Location Type",
        options=[0, 1, 2],
        format_func=lambda x: ["Unknown", "Urban", "Rural"][x],
        index=1,
        help="Urban or rural classification"
    )
    
    low_doc = st.selectbox(
        "Low Documentation Program",
        options=["N", "Y"],
        format_func=lambda x: "No" if x == "N" else "Yes",
        help="Whether loan uses low documentation process"
    )

st.markdown("---")

# Prediction button
if st.button("🎯 Calculate Risk Scores", type="primary", use_container_width=True):
    
    with st.spinner("Analyzing loan application..."):
        
        try:
            # Create input dataframe
            input_data = pd.DataFrame([{
                "Term": term,
                "NoEmp": noemp,
                "DisbursementGross": loan_amt,
                "NewExist": new_exist,
                "UrbanRural": urban_rural,
                "LowDoc": low_doc,
                "ApprovalFY": str(approval_fy),
                "NAICS": naics
            }])
            
            # Apply feature engineering
            features = create_features(input_data)
            
            # Ensure features match expected columns
            features_pd = features[pd_features]
            features_lgd = features[lgd_features]
            
            # Make predictions
            pd_score = pd_model.predict_proba(features_pd)[:, 1][0]
            lgd_score = lgd_model.predict(features_lgd)[0]
            
            # Clip LGD to valid range
            lgd_score = np.clip(lgd_score, 0, 1)
            
            # Calculate expected loss
            expected_loss = loan_amt * pd_score * lgd_score
            
            # Display results in cards
            st.markdown("## 📊 Risk Assessment Results")
            
            # Create three columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="Probability of Default (PD)",
                    value=f"{pd_score:.2%}",
                    delta=None,
                    help="Likelihood that the borrower will default"
                )
                
                # Risk level indicator
                if pd_score < 0.1:
                    st.success("🟢 Low Risk")
                elif pd_score < 0.25:
                    st.warning("🟡 Medium Risk")
                else:
                    st.error("🔴 High Risk")
            
            with metric_col2:
                st.metric(
                    label="Loss Given Default (LGD)",
                    value=f"{lgd_score:.2%}",
                    delta=None,
                    help="Expected loss percentage if default occurs"
                )
                
                # LGD severity
                if lgd_score < 0.3:
                    st.success("🟢 Low Severity")
                elif lgd_score < 0.6:
                    st.warning("🟡 Medium Severity")
                else:
                    st.error("🔴 High Severity")
            
            with metric_col3:
                st.metric(
                    label="Expected Loss (EL)",
                    value=f"${expected_loss:,.2f}",
                    delta=None,
                    help="Expected monetary loss (PD × LGD × Loan Amount)"
                )
                
                # EL percentage of loan
                el_percentage = (expected_loss / loan_amt) * 100
                st.info(f"📉 {el_percentage:.2f}% of loan amount")
            
            # Additional information
            st.markdown("---")
            st.markdown("### 📝 Risk Interpretation")
            
            interpretation_col1, interpretation_col2 = st.columns(2)
            
            with interpretation_col1:
                st.markdown("**Risk Factors:**")
                if new_exist == 2:
                    st.markdown("- ⚠️ New business (higher risk)")
                else:
                    st.markdown("- ✓ Existing business (lower risk)")
                
                if low_doc == "Y":
                    st.markdown("- ⚠️ Low documentation (less verification)")
                else:
                    st.markdown("- ✓ Standard documentation")
                
                if term > 120:
                    st.markdown("- ⚠️ Long loan term (increased risk)")
                else:
                    st.markdown("- ✓ Standard loan term")
            
            with interpretation_col2:
                st.markdown("**Recommendation:**")
                
                if pd_score < 0.1:
                    st.success("✅ **APPROVE** - Low default risk")
                    st.markdown("This loan application shows strong creditworthiness.")
                elif pd_score < 0.25:
                    st.warning("⚠️ **REVIEW** - Moderate risk")
                    st.markdown("Consider additional collateral or guarantees.")
                else:
                    st.error("❌ **DECLINE or REQUEST MITIGATION** - High risk")
                    st.markdown("Significant risk mitigation required before approval.")
            
            # Show input summary
            with st.expander("📄 View Input Summary"):
                st.json({
                    "Loan Information": {
                        "Loan Amount": f"${loan_amt:,.2f}",
                        "Term": f"{term} months",
                        "NAICS Code": naics,
                        "Approval Year": approval_fy
                    },
                    "Business Information": {
                        "Number of Employees": noemp,
                        "Business Type": "Existing" if new_exist == 1 else "New",
                        "Location": ["Unknown", "Urban", "Rural"][urban_rural],
                        "Low Documentation": "Yes" if low_doc == "Y" else "No"
                    }
                })
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            st.error("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Credit Risk Early Warning System | Powered by XGBoost ML Models</p>
        <p style='font-size: 12px; color: gray;'>
            This system uses machine learning to predict loan default probability and loss severity.
            Results should be used as decision support only.
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=EWS", width=150)
    st.markdown("## About")
    st.info("""
        This Early Warning System predicts:
        
        - **PD**: Probability of default
        - **LGD**: Loss given default
        - **EL**: Expected loss
        
        Built on SBA loan historical data.
    """)
    
    st.markdown("## Model Performance")
    st.metric("PD Model AUC", "0.968")
    st.metric("LGD Model R²", "0.589")
    
    st.markdown("## Help")
    with st.expander("📖 User Guide"):
        st.markdown("""
            **How to use:**
            1. Enter loan details
            2. Enter business information
            3. Click 'Calculate Risk Scores'
            4. Review the results
            
            **Risk Levels:**
            - 🟢 Low: PD < 10%
            - 🟡 Medium: PD 10-25%
            - 🔴 High: PD > 25%
        """)
