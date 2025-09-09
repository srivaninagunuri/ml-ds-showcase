# ==========================
# STEP 8: Streamlit Dashboard
# ==========================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load saved model + vectorizer
model = joblib.load("E-commerce review_model.pkl")
vectorizer = joblib.load("E-commerce_tfidf_vectorizer.pkl")

# Load your latest dataset with predictions (from Step 7)
df_new = pd.read_csv("predicted_reviews.csv")

# ----------------------------
# Dashboard Layout
# ----------------------------
st.title("📊 E-commerce Review Sentiment Dashboard")

# Sidebar filters
platforms = df_new["platform"].unique().tolist()
selected_platform = st.sidebar.selectbox("Select Platform", ["All"] + platforms)

if selected_platform != "All":
    df_new = df_new[df_new["platform"] == selected_platform]

# ----------------------------
# 1️⃣ Overall Sentiment Distribution
# ----------------------------
# Define consistent sentiment colors
colors = {"negative": "red", "neutral": "orange", "positive": "green"}

# ----------------------------
# 1️⃣ Overall Sentiment Distribution
# ----------------------------
st.subheader("Overall Sentiment Distribution")

sent_counts = df_new["predicted_sentiment"].value_counts()

fig1, ax1 = plt.subplots(figsize=(4, 4))
ax1.pie(
    sent_counts,
    labels=sent_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=[colors[label] for label in sent_counts.index],  # consistent colors
    radius=0.7  # slightly smaller circle
)
ax1.axis("equal")

# ----------------------------
# 2️⃣ Sentiment by Platform
# ----------------------------
st.subheader("Sentiment by Platform")

platform_sent = (
    df_new.groupby("platform")["predicted_sentiment"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0) * 100
)

fig2, ax2 = plt.subplots(figsize=(6, 4))
platform_sent.plot(
    kind="bar",
    stacked=True,
    ax=ax2,
    color=[colors[col] for col in platform_sent.columns]  # consistent colors
)
ax2.set_ylabel("Percentage (%)")

# 👉 Move legend outside
ax2.legend(
    title="Sentiment",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

# ----------------------------
# Layout: side by side
# ----------------------------
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.pyplot(fig2)

# ----------------------------
# 3️⃣ Top Negative Products
# ----------------------------
st.subheader("⚠️ Top 5 Products with Negative Reviews")
top_neg = (
    df_new[df_new["predicted_sentiment"] == "negative"]["product_title"]
    .value_counts()
    .head(5)
)
st.table(top_neg)

# ----------------------------
# 4️⃣ Single Review Prediction
# ----------------------------
st.subheader("🔍 Try a New Review")

user_review = st.text_area("Enter review text here:")
if st.button("Predict Sentiment"):
    if user_review.strip():
        X = vectorizer.transform([user_review])
        prediction = model.predict(X)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text to analyze.")
