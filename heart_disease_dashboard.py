import streamlit as st
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# Define paths
model_path = "heart_disease_model.h5"
scaler_path = r"C:\Users\admin\Downloads\DataScience\venv\scaler.pkl"  # Use raw string to avoid unicode escape error

# Check if scaler exists before loading
if not os.path.exists(scaler_path):
    st.error(f"❌ Scaler file 'scaler.pkl' not found at {scaler_path}. Please check the path and try again.")
    st.stop()

# Load the trained model and scaler
model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Streamlit page configuration
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

# Navigation menu
menu = st.sidebar.radio("Navigation", ["Welcome", "Risk Analysis", "Insights"])

if menu == "Welcome":
    st.markdown("""
    <h1 style='font-size: 28px;'>💓📈 Heart Disease Risk Assessment Portal</h1>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### Project Overview
    This interactive dashboard helps assess the **risk of heart disease** based on patient medical attributes.
    It uses a **Neural Network model** trained on heart health datasets.

    🔍 Navigate to **Risk Analysis** to input patient data and evaluate risk.  
    📊 Explore **Insights** to understand project outcomes and takeaways.
    ---
    """)

elif menu == "Risk Analysis":
    #st.title("❤️ Heart Disease Risk Prediction Dashboard")
    st.markdown("""
    Please enter patient details below to evaluate the risk of heart disease using our trained model.
    """)

    # Button to autofill high-risk values
    if st.button("🢜 Load High-Risk Example"):
        st.session_state.age = 70
        st.session_state.sex = "Male"
        st.session_state.chest_pain = 4
        st.session_state.bp = 165
        st.session_state.cholesterol = 350
        st.session_state.fbs = "Yes"
        st.session_state.ekg = 2
        st.session_state.max_hr = 105
        st.session_state.ex_angina = "Yes"
        st.session_state.oldpeak = 3.5
        st.session_state.slope = 3
        st.session_state.vessels = 3
        st.session_state.thal = 7

    # Button to autofill low-risk values
    if st.button("🔍 Load Low-Risk Example"):
        st.session_state.age = 35
        st.session_state.sex = "Female"
        st.session_state.chest_pain = 2
        st.session_state.bp = 110
        st.session_state.cholesterol = 180
        st.session_state.fbs = "No"
        st.session_state.ekg = 0
        st.session_state.max_hr = 170
        st.session_state.ex_angina = "No"
        st.session_state.oldpeak = 0.0
        st.session_state.slope = 2
        st.session_state.vessels = 0
        st.session_state.thal = 3

    # Create input form
    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 20, 100, value=st.session_state.get("age", 50))
            sex = st.selectbox("Sex", ["Male", "Female"], index=0 if st.session_state.get("sex", "Male") == "Male" else 1)
            chest_pain = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4], index=[1, 2, 3, 4].index(st.session_state.get("chest_pain", 1)))
            bp = st.slider("Resting Blood Pressure (BP)", 90, 200, value=st.session_state.get("bp", 120))
            cholesterol = st.slider("Serum Cholesterol", 100, 600, value=st.session_state.get("cholesterol", 250))
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"], index=0 if st.session_state.get("fbs", "No") == "Yes" else 1)

        with col2:
            ekg = st.selectbox("Resting ECG Results", [0, 1, 2], index=[0, 1, 2].index(st.session_state.get("ekg", 0)))
            max_hr = st.slider("Maximum Heart Rate Achieved", 70, 220, value=st.session_state.get("max_hr", 150))
            ex_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"], index=0 if st.session_state.get("ex_angina", "No") == "Yes" else 1)
            oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, step=0.1, value=st.session_state.get("oldpeak", 1.0))
            slope = st.selectbox("Slope of the Peak ST Segment", [1, 2, 3], index=[1, 2, 3].index(st.session_state.get("slope", 1)))
            vessels = st.selectbox("# of Major Vessels (0-3) Colored by Fluoroscopy", [0, 1, 2, 3], index=[0, 1, 2, 3].index(st.session_state.get("vessels", 0)))
            thal = st.selectbox("Thalassemia", [3, 6, 7], index=[3, 6, 7].index(st.session_state.get("thal", 3)))

        submitted = st.form_submit_button("Predict")

    if submitted:
        sex_val = 1 if sex == "Male" else 0
        fbs_val = 1 if fbs == "Yes" else 0
        angina_val = 1 if ex_angina == "Yes" else 0

        input_array = np.array([[
            age, sex_val, chest_pain, bp, cholesterol, fbs_val, ekg, max_hr,
            angina_val, oldpeak, slope, vessels, thal
        ]])

        input_array_scaled = input_array.astype(float)
        input_array_scaled[:, [0, 3, 4, 7, 9, 11]] = scaler.transform(input_array_scaled[:, [0, 3, 4, 7, 9, 11]])

        prediction = model.predict(input_array_scaled)[0][0]
        probability = round(prediction * 100, 2)

        st.markdown("---")
        st.subheader("Prediction Result")
        st.write(f"🔍 Model raw prediction: {prediction:.4f}")
        if prediction > 0.5:
            st.error(f"⚠️ High risk of heart disease: **{probability}%**")
        else:
            st.success(f"✅ Low risk of heart disease: **{probability}%**")

        st.caption("Note: This tool is for educational purposes and does not replace professional medical advice.")

elif menu == "Insights":
    st.title("📊 Insights")
    st.markdown("""
    ### Summary
    - This project demonstrates how neural networks can predict heart disease risk based on clinical data.
    - Inputs like age, blood pressure, cholesterol, ECG, and more influence the prediction.
    - The dashboard provides real-time, user-friendly access to the trained model.

    ### Key Takeaway
    > Early detection and lifestyle monitoring are essential for preventing cardiovascular disease.

    ### Final Note
    ✅ This dashboard is built for academic and demonstration purposes only.
    Please consult a medical professional for actual diagnoses.
    """)
