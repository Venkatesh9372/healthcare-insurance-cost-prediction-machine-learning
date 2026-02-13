import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Health Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------------
# GLOBAL CSS (NO SCROLL + COMPACT UI)
# -----------------------------------
st.markdown("""
<style>
/* Remove extra padding */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 1rem;
    max-height: 100vh;
}

/* Disable scrolling */
html, body {
    overflow-y: auto;
}

/* Sidebar compact */
section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* Card styling */
.card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
}

.result {
    font-size: 30px;
    font-weight: bold;
    color: #2e8b57;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Load Model
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ML_HealthCare_model.pkl")

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model(MODEL_PATH)

# -----------------------------------
# Header
# -----------------------------------
st.title("🏥 Health Insurance Cost Prediction")
st.caption("Personalized insurance cost estimation using Machine Learning")

# -----------------------------------
# Sidebar Inputs
# -----------------------------------
st.sidebar.header("👤 User Details")

name = st.sidebar.text_input("Full Name")

age = st.sidebar.number_input("Age", 18, 100, 30, step=1)
height_cm = st.sidebar.number_input("Height (cm)", 100, 220, 170)
weight_kg = st.sidebar.number_input("Weight (kg)", 30, 200, 70)

gender = st.sidebar.selectbox("Gender", ["male", "female"])
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.text_input("Region / City / State")

# -----------------------------------
# BMI Calculation
# -----------------------------------
height_m = height_cm / 100
bmi = round(weight_kg / (height_m ** 2), 2)

# -----------------------------------
# Main Layout
# -----------------------------------
col1, col2 = st.columns([2, 1])

# -------- USER SUMMARY --------
with col1:
    st.subheader("📋 User Summary")
    st.write(f"**Name:** {name if name else '—'}")
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Height:** {height_cm} cm")
    st.write(f"**Weight:** {weight_kg} kg")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**Children:** {children}")
    st.write(f"**Smoker:** {smoker}")
    st.write(f"**Region:** {region if region else '—'}")

# -------- PREDICTION --------
with col2:
    st.subheader("💡 Prediction")

    if st.button("🔍 Predict Insurance Cost", use_container_width=True):

        if name.strip() == "" or region.strip() == "":
            st.warning("Please enter your name and region.")
        else:
            input_data = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "BMI": [bmi],
                "Childrens": [children],
                "Smoker": [smoker],
                "Region": [region.lower()]
            })

            prediction = model.predict(input_data)[0]

            st.markdown(
                f"""
                <div class="card">
                    <p>Hey <b>{name}</b> 👋</p>
                    <p>Your estimated annual insurance cost is</p>
                    <div class="result">₹ {prediction:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------------------
# Footer
# -----------------------------------
st.markdown(
    "<hr style='margin-top:10px;margin-bottom:5px;'>"
    "📊 <b>Machine Learning Project</b> | Health Insurance Cost Prediction<br>"
    "Built using Streamlit & Scikit-learn",
    unsafe_allow_html=True
)

# Steps for run the sreamlit app
# 1. cd C:\Users\Venkstesh\VS Projects\Healthcare_Streamlit>
# 2. "C:/Users/Venkstesh/VS Projects/.venv/Scripts/Activate.ps1"
# 3. dir
# 4. streamlit run health.py