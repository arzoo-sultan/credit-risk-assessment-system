"""
Streamlit dashboard for the Credit Risk Assessment System.

Run with:
    streamlit run dashboard/app.py

Planned sections:
1. Applicant risk lookup — input form, calls the FastAPI /predict endpoint
   (or loads the model directly), shows probability + risk score + SHAP
   explanation plot.
2. Portfolio overview — distribution of risk scores across the dataset,
   default rate by segment (income bracket, age group, etc.).
3. Model performance — ROC/AUC curve, confusion matrix, feature importance.
4. Bias/fairness panel — default rate & approval rate across sensitive
   groups, to support the "bias detection" requirement.
"""
import streamlit as st

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("Credit Risk Assessment Dashboard")
st.caption("Placeholder — sections will be built out alongside model training.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Applicant Lookup", "Portfolio Overview", "Model Performance", "Bias & Fairness"]
)

with tab1:
    st.info("TODO: applicant input form + live prediction + SHAP explanation")

with tab2:
    st.info("TODO: risk score distribution, default rate by segment")

with tab3:
    st.info("TODO: ROC/AUC, confusion matrix, feature importance")

with tab4:
    st.info("TODO: fairness metrics across demographic groups")
