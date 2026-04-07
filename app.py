import streamlit as st
import pandas as pd

from config.settings import DATE_COLUMN
from data.db_connector import fetch_incident_data
from data.preprocessor import preprocess
from models.predictor import predict_future_risks
from models.ts_forecaster import run_ts_forecast


# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Incident Predictive Analysis",
    page_icon="⚠️",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# Custom UI Styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* Page background */
.main {
    background-color: #ffffff;
}

/* Sidebar input colors */
section[data-testid="stSidebar"] {
    background-color: #f5faff;
}

/* Input boxes */
div[data-baseweb="select"] > div {
    background-color: #eef7ff;
}

input {
    background-color: #eefaf0 !important;
}

/* Reduce metric label size */
div[data-testid="stMetricLabel"] {
    font-size: 14px !important;
    font-weight: 600;
}

/* Reduce metric value size */
div[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 600;
}

/* Section titles */
h1 {
    font-size: 34px !important;
}

h2 {
    font-size: 26px !important;
}

h3 {
    font-size: 22px !important;
}

</style>
""", unsafe_allow_html=True)

st.title(" 📊 Incident Predictive Analysis System")


# ─────────────────────────────────────────────────────────────
# Load Data (cached for performance)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_raw = fetch_incident_data()
    df_processed, encoders = preprocess(df_raw, fit=False)
    last_training_date = df_processed[DATE_COLUMN].max()
    return df_processed, last_training_date


df_processed, last_training_date = load_data()


# ─────────────────────────────────────────────────────────────
# Sidebar Inputs
# ─────────────────────────────────────────────────────────────
st.sidebar.header("User Input")

available_depts = sorted(df_processed["department_name"].unique())

department = st.sidebar.selectbox(
    "Select Department",
    available_depts
)

forecast_days = st.sidebar.number_input(
    "Enter Forecast Days",
    min_value=1,
    max_value=365,
    value=30
)

predict_button = st.sidebar.button("Run Prediction")


# ─────────────────────────────────────────────────────────────
# Run Prediction
# ─────────────────────────────────────────────────────────────
if predict_button:

    # ==========================================================
    # 1️⃣ INCIDENT RISK PREDICTION
    # ==========================================================
    st.header("🔎 Incident Risk Prediction")

    try:
        result = predict_future_risks(
            department,
            forecast_days,
            last_training_date
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Incident Type", result["incident_type"])
            st.metric("Predicted Severity", result["severity_type"])
            st.metric("Risk Level", result["risk_level"])

        with col2:
            st.metric("Probability", result["probability"])
            st.metric("Month", result["month"])
            st.metric("Season", result["season"])

        st.subheader("⚠️ Warnings")
        for w in result["warning"]:
            st.warning(w)
        
        st.subheader("✅ Recommendations")
            
        for r in result["recommendation"]:
                st.info(r)

        #st.subheader("Recommendation")
        #st.info(result["recommendation"])

        #st.subheader("Warning")
        #st.warning(result["warning"])

        #st.subheader("AI Safety Advisory")
        #st.info(result["ai_advice"])

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    # ==========================================================
    # 2️⃣ TIME SERIES FREQUENCY FORECAST
    # ==========================================================
    st.header("📈 Time Series Frequency Forecast")

    try:
        ts_result = run_ts_forecast(
            df=df_processed,
            department=department,
            forecast_days=forecast_days,
            last_training_date=last_training_date,
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            f"Next {forecast_days} Days Incidents",
            ts_result["total_incidents"]
        )

        col2.metric(
            "Trend",
            ts_result["trend"]
        )

        col3.metric(
            "High Risk Week",
            ts_result["high_risk_week"]
        )

        st.subheader("Expected Incident Distribution")

        if ts_result["distribution"]:
            dist_df = pd.DataFrame(
                list(ts_result["distribution"].items()),
                columns=["Incident Type", "Expected Count"]
            )
            st.bar_chart(dist_df.set_index("Incident Type"))
        else:
            st.info("Insufficient data for distribution analysis.")

    except Exception as e:
        st.warning(f"Time-Series Forecast skipped: {e}")


        