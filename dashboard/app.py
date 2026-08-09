import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("Credit Risk Assessment Dashboard")

API_URL = "http://127.0.0.1:8000"

st.header("Applicant Risk Lookup")
st.write("Paste applicant feature JSON below, or upload a payload file.")

uploaded_file = st.file_uploader("Upload applicant JSON", type="json")

if uploaded_file is not None:
    payload = json.load(uploaded_file)

    if st.button("Get Risk Assessment"):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()
            result = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Default Probability", f"{result['default_probability']*100:.1f}%")
            col2.metric("Risk Band", result['risk_band'])
            col3.metric("Credit Score", result['credit_score'])

            if result['flagged_high_risk']:
                st.error(f"⚠️ Flagged as high risk (threshold: {result['decision_threshold']:.2%})")
            else:
                st.success(f"✅ Not flagged as high risk (threshold: {result['decision_threshold']:.2%})")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API — make sure it's running at " + API_URL)
        except Exception as e:
            st.error(f"Error: {e}")



st.divider()
st.header("Portfolio Overview")

try:
    portfolio_df = pd.read_csv("reports/risk_scoring_results.csv")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applicants", f"{len(portfolio_df):,}")
    col2.metric("Overall Default Rate", f"{portfolio_df['actual_default'].mean()*100:.1f}%")
    col3.metric("Avg Credit Score", f"{portfolio_df['credit_score'].mean():.0f}")

    import plotly.express as px

    band_counts = portfolio_df['risk_band'].value_counts().reindex(['Low', 'Medium', 'High'])
    fig1 = px.bar(
        x=band_counts.index, y=band_counts.values,
        labels={'x': 'Risk Band', 'y': 'Number of Applicants'},
        title="Applicants by Risk Band"
    )
    st.plotly_chart(fig1, use_container_width=True)

    band_default_rate = portfolio_df.groupby('risk_band')['actual_default'].mean().reindex(['Low', 'Medium', 'High']) * 100
    fig2 = px.bar(
        x=band_default_rate.index, y=band_default_rate.values,
        labels={'x': 'Risk Band', 'y': 'Actual Default Rate (%)'},
        title="Actual Default Rate by Predicted Risk Band"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(
        portfolio_df, x='credit_score', nbins=30,
        title="Credit Score Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

except FileNotFoundError:
    st.warning("reports/risk_scoring_results.csv not found — download it from Drive into your local reports/ folder.")


st.divider()
st.header("Model Performance")

try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix

    fpr, tpr, _ = roc_curve(portfolio_df['actual_default'], portfolio_df['default_probability'])
    roc_auc = auc(fpr, tpr)

    fig_roc = px.line(
        x=fpr, y=tpr,
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        title=f"ROC Curve (AUC = {roc_auc:.4f})"
    )
    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='gray'))
    st.plotly_chart(fig_roc, use_container_width=True)

    decision_threshold = 0.7199
    y_pred_at_threshold = (portfolio_df['default_probability'] >= decision_threshold).astype(int)
    cm = confusion_matrix(portfolio_df['actual_default'], y_pred_at_threshold)

    fig_cm = px.imshow(
        cm, text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No Default', 'Default'], y=['No Default', 'Default'],
        title=f"Confusion Matrix (threshold = {decision_threshold:.2%})"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

except FileNotFoundError:
    st.warning("risk_scoring_results.csv not found.")



st.divider()
st.header("Bias & Fairness")

try:
    fairness_df = pd.read_csv("reports/fairness_analysis.csv")

    st.write("Comparing model predictions and real outcomes across gender, "
             "to check whether the model treats groups equitably.")

    gender_summary = fairness_df.groupby('gender').agg(
        avg_predicted_probability=('default_probability', 'mean'),
        actual_default_rate=('actual_default', 'mean'),
        count=('actual_default', 'size')
    )
    st.dataframe(gender_summary.style.format({
        'avg_predicted_probability': '{:.2%}',
        'actual_default_rate': '{:.2%}'
    }))

    fig_fair = px.bar(
        gender_summary.reset_index(),
        x='gender', y=['avg_predicted_probability', 'actual_default_rate'],
        barmode='group',
        labels={'value': 'Rate', 'gender': 'Gender', 'variable': 'Metric'},
        title="Predicted vs Actual Default Rate by Gender"
    )
    st.plotly_chart(fig_fair, use_container_width=True)

    st.info(
        "**Note:** Gender was excluded as a model input after SHAP analysis revealed "
        "it contributed independent risk signal beyond what's explained by financial "
        "features — a fairness concern under regulations like ECOA. Removing it cost "
        "only 0.003 AUC (0.7635 → 0.7604), confirming it wasn't necessary for predictive "
        "performance. The remaining gap in predicted probability above reflects real "
        "differences in the underlying financial data (e.g. income, credit history), "
        "not direct use of gender by the model."
        "Note: 'XNA' represents only 2 applicants (a rare unknown-gender placeholder in the "
            "source data) and is not statistically meaningful. Predicted probabilities appear "
            "elevated relative to actual rates across all groups due to class-weighting used to "
            "address the dataset's overall imbalance (see Model Performance tab) — the relevant "
            "comparison here is the relative gap between groups, not the absolute values."
    )
    
    

except FileNotFoundError:
    st.warning("reports/fairness_analysis.csv not found.")
