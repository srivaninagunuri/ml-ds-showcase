# --- Page Config ---
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import pickle 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)


# Load model and scaler
df = pd.read_csv("telco_scaled_data.csv")  # Ensure this file exists in the same directory
scaler = joblib.load("scaler.pkl")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

    model = model[0]  # Extract the real model
    if isinstance(model, list):
        model = model[0]

# Load model columns from JSON
with open('model_columns.json', 'r') as f:
    model_columns = json.load(f)

X_test = joblib.load("X_test.pkl")
X_test = pd.DataFrame(X_test, columns=model_columns)  # ✅ FIXED LINE

y_test = joblib.load("y_test.pkl")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Save in session_state
st.session_state.model = model
st.session_state.feature_names = X_test.columns.tolist()
st.session_state.y_test = y_test
st.session_state.y_pred = y_pred
st.session_state.y_pred_prob = y_pred_prob



# --- Page Navigation ---
st.set_page_config(page_title="📊 Telco Churn Dashboard", layout="wide")

page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Visualization", "🔍 Prediction"])

# --- Home Page ---
if page == "🏠 Home":
    st.title("📞 Welcome to Telco Customer Churn Dashboard")
    st.markdown("""
    This interactive dashboard helps telecom companies:
    - Visualize churn trends
    - Predict customer churn risk
    - Take data-driven retention decisions

    Navigate to **Prediction** to test with new customer data,
    or explore the **Visualization** page for insights.
    """)

if page=="📈 Visualization":

    # --- Visualization Page ---

    st.title("📊 Customer Churn Insights")
    col1, col2, col3= st.columns(3)

    # --- 1️⃣ Churn Distribution ---
    with col1:
        #st.subheader("1️⃣ Churn Distribution")
        fig1, ax1 = plt.subplots()
        df['ChurnLabel'] = df['Churn'].map({0: 'No', 1: 'Yes'})
        sns.countplot(data=df, x='ChurnLabel', palette='coolwarm', ax=ax1)
        ax1.set_title("Overall Churn Distribution")
        st.pyplot(fig1)

    # --- 2️⃣ Churn by Contract ---
    with col2:
        #st.subheader("2️⃣ Churn by Contract Type")
        fig2, ax2 = plt.subplots()
        contract_cols = ['Contract_One year', 'Contract_Two year']
        df_contract = df.copy()
        df_contract['Contract'] = 'Month-to-month'
        df_contract.loc[df_contract['Contract_One year'] == 1, 'Contract'] = 'One year'
        df_contract.loc[df_contract['Contract_Two year'] == 1, 'Contract'] = 'Two year'
        sns.countplot(data=df_contract, x='Contract', hue='ChurnLabel', palette='Set2', ax=ax2)
        ax2.set_title("Churn by Contract Type")
        st.pyplot(fig2)

    with col3:
        #st.subheader("3️⃣ Monthly Charges by Churn")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='ChurnLabel', y='MonthlyCharges', palette='pastel', ax=ax3)
        ax3.set_title("Monthly Charges Distribution by Churn")
        st.pyplot(fig3)
   

    col4, col5,col6 = st.columns(3)
    # --- 4️⃣ Churn by Tenure Group ---
    with col4:
        #st.subheader("4️⃣ Churn by Tenure Group")
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, 72], labels=['0-12','13-24','25-48','49-60','61-72'])
        fig4, ax4 = plt.subplots()
        sns.countplot(data=df, x='TenureGroup', hue='ChurnLabel', palette='cool', ax=ax4)
        ax4.set_title("Churn by Tenure Group")
        st.pyplot(fig4)

    # --- 5️⃣ Churn by Senior Citizen ---

    with col5:
        #st.subheader("5️⃣ Churn by Senior Citizen Status")
        senior_churn = df.groupby(['SeniorCitizen', 'ChurnLabel']).size().unstack()
        senior_churn.index = senior_churn.index.map({0: 'Not Senior', 1: 'Senior'})
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        senior_churn.plot(kind='bar', stacked=True, ax=ax5, color=['skyblue', 'salmon'])
        ax5.set_title('Churn by Senior Citizen Status')
        ax5.set_ylabel('Customers')
        ax5.set_xlabel('Senior Citizen')
        ax5.set_xticklabels(['Not Senior', 'Senior'], rotation=0)
        st.pyplot(fig5)
    with col6:
        # --- 6️⃣ Churn by Payment Method ---
        #st.subheader("6️⃣ Churn by Payment Method")
        payment_cols = ['PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
        df_payment = df.copy()
        df_payment['PaymentMethod'] = 'Bank transfer (automatic)'
        df_payment.loc[df_payment['PaymentMethod_Credit card (automatic)'] == 1, 'PaymentMethod'] = 'Credit Card'
        df_payment.loc[df_payment['PaymentMethod_Electronic check'] == 1, 'PaymentMethod'] = 'Electronic Check'
        df_payment.loc[df_payment['PaymentMethod_Mailed check'] == 1, 'PaymentMethod'] = 'Mailed Check'
        fig6, ax6 = plt.subplots()
        sns.countplot(data=df_payment, x='PaymentMethod', hue='ChurnLabel', palette='magma', ax=ax6)
        ax6.set_title("Churn by Payment Method")
        ax6.set_xticklabels(['Bank transfer', 'Credit Card', 'Electronic Check', 'Mailed Check'], rotation=10)
        st.pyplot(fig6)

    # --- 🔁 Row: Confusion Matrix + ROC Curve ---
    #st.markdown("### 📉 Confusion Matrix & 📈 ROC Curve")
    col7, col8 = st.columns(2)

    with col7:
        #st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
        disp.plot(ax=ax_cm, cmap='Blues')
        ax_cm.set_aspect('equal')
        st.pyplot(fig_cm)

    with col8:
        #st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(st.session_state.y_test, st.session_state.y_pred_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    # --- 🔁 Row: Feature Importances + Performance ---
    #st.markdown("### 💡 Feature Importances & 📊 Model Performance")
    col9, col10 = st.columns(2)

    with col9:
        #st.subheader("Top 10 Feature Importances")
        model = st.session_state.model
        feature_names = st.session_state.feature_names
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = None

        if importances is not None:
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
            fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
            sns.barplot(data=feat_df, y='Feature', x='Importance', palette='viridis', ax=ax_imp)
            ax_imp.set_title("Top 10 Feature Importances")
            st.pyplot(fig_imp)
        else:
            st.warning("Model does not support feature importances.")

    with col10:
        #st.subheader("Model Performance")
        accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
        precision = precision_score(st.session_state.y_test, st.session_state.y_pred)
        recall = recall_score(st.session_state.y_test, st.session_state.y_pred)
        f1 = f1_score(st.session_state.y_test, st.session_state.y_pred)

        st.markdown(
        f"""
        <div style="
            background-color: #f0f2f6; 
            padding: 4px 8px; 
            border-radius: 6px; 
            box-shadow: 0 0 5px rgba(0,0,0,0.1); 
            font-size: 14px; 
            line-height: 1.4;
            width: 250px;   /* controls box width */
        ">
            <ul style="margin: 0; padding-left: 18px;">
                <li><b>Accuracy:</b> {accuracy:.2f}</li>
                <li><b>Precision (Churn):</b> {precision:.2f}</li>
                <li><b>Recall (Churn):</b> {recall:.2f}</li>
                <li><b>F1 Score (Churn):</b> {f1:.2f}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("**Note:** Visualizations are based on processed and encoded Telco churn data.")


# --- Prediction Page ---
elif page == "🔍 Prediction":
    st.title("📞 Telco Customer Churn Prediction Dashboard")
    st.markdown("Enter customer details below to predict churn probability.")

    # --- Sidebar Inputs ---
    st.sidebar.header("Customer Information")

    # ... (unchanged sidebar code for inputs)

    # --- Manual Input DataFrame ---
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 3000.0)

    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ])

    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    # --- Manual Input DataFrame ---
    input_data = {
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [1 if partner == "Yes" else 0],
        'Dependents': [1 if dependents == "Yes" else 0],
        'tenure': [tenure],
        'PhoneService': [1 if phone_service == "Yes" else 0],
        'PaperlessBilling': [1 if paperless_billing == "Yes" else 0],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender_Male': [1 if gender == "Male" else 0],
        'Contract_One year': [1 if contract == "One year" else 0],
        'Contract_Two year': [1 if contract == "Two year" else 0],
        'PaymentMethod_Credit card (automatic)': [1 if payment_method == "Credit card (automatic)" else 0],
        'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
        'PaymentMethod_Mailed check': [1 if payment_method == "Mailed check" else 0],
        'InternetService_Fiber optic': [1 if internet_service == "Fiber optic" else 0],
        'InternetService_No': [1 if internet_service == "No" else 0],
        'OnlineSecurity_No': [1 if online_security == "No" else 0],
        'OnlineSecurity_No internet service': [1 if online_security == "No internet service" else 0],
        'OnlineSecurity_Yes': [1 if online_security == "Yes" else 0],
        'OnlineBackup_No': [1 if online_backup == "No" else 0],
        'OnlineBackup_No internet service': [1 if online_backup == "No internet service" else 0],
        'OnlineBackup_Yes': [1 if online_backup == "Yes" else 0],
        'DeviceProtection_No': [1 if device_protection == "No" else 0],
        'DeviceProtection_No internet service': [1 if device_protection == "No internet service" else 0],
        'DeviceProtection_Yes': [1 if device_protection == "Yes" else 0],
        'TechSupport_No': [1 if tech_support == "No" else 0],
        'TechSupport_No internet service': [1 if tech_support == "No internet service" else 0],
        'TechSupport_Yes': [1 if tech_support == "Yes" else 0],
        'StreamingTV_No': [1 if streaming_tv == "No" else 0],
        'StreamingTV_No internet service': [1 if streaming_tv == "No internet service" else 0],
        'StreamingTV_Yes': [1 if streaming_tv == "Yes" else 0],
        'StreamingMovies_No': [1 if streaming_movies == "No" else 0],
        'StreamingMovies_No internet service': [1 if streaming_movies == "No internet service" else 0],
        'StreamingMovies_Yes': [1 if streaming_movies == "Yes" else 0]
    }

    # --- Convert input to DataFrame ---
    input_df = pd.DataFrame(input_data)

    # Ensure column alignment
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # --- Scale input ---
    input_scaled = scaler.transform(input_df)

    # --- Predict ---
    if st.button("🔍 Predict Churn"):
        prediction = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled)[0][1]

        st.subheader("🔮 Prediction Result")
        if prediction[0] == 1:
            st.error(f"⚠️ This customer is likely to churn. (Probability: {proba:.2f})")
                # 🎯 Actionable Suggestions
            st.markdown("### 🔧 Suggested Retention Actions:")
            st.markdown("""
            - 📞 **Contact the customer** with a personalized call or message.
            - 💬 **Offer a better plan or discount** to suit their needs.
            - 🤝 **Ask for feedback** to understand any dissatisfaction.
            - 📊 **Analyze their usage patterns** to tailor services better.
            - 🛠️ **Improve customer support responsiveness** if applicable.
            """)
        else:
            st.success(f"✅ This customer is likely to stay. (Probability: {1 - proba:.2f})")
            st.markdown("### 🌟 Customer Loyalty Suggestions:")
            st.markdown("""
            - 🏆 Offer loyalty rewards or referral bonuses.
            - 📈 Provide service usage insights to improve engagement.
            - 💡 Promote value-added services based on their profile.
            - 🤝 Ask for positive reviews or testimonials.
            - 📬 Send thank-you emails to reinforce satisfaction.
            """)


