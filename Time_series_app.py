import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# ===============================
# Load Data & Model
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("processed_sales_data.csv", parse_dates=["Date"])

@st.cache_data
def load_forecast_results():
    return pd.read_csv("forecast_results.csv", parse_dates=["Date"])

@st.cache_data
def load_metrics():
    with open("metrics.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return joblib.load("xgb_forecast_model.pkl")

df = load_data()
forecast_df = load_forecast_results()
metrics = load_metrics()
model = load_model()

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["EDA Dashboard", "Forecasting Dashboard"])

# ===============================
# EDA Dashboard
# ===============================
if page == "EDA Dashboard":
    st.title("🛒 Retail Store Inventory & Demand Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        cat_demand = df.groupby("Category")["Demand"].sum().sort_values()
        fig, ax = plt.subplots()
        cat_demand.plot(kind="barh", ax=ax, color="navy")
        ax.set_title("Total Demand by Category")
        st.pyplot(fig)

    with col2:
        region_demand = df.groupby("Region")["Demand"].sum().sort_values()
        fig, ax = plt.subplots()
        region_demand.plot(kind="barh", ax=ax, color="purple")
        ax.set_title("Total Demand by Region")
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Demand"], color="orange")
        ax.set_title("Daily Demand Over Time")
        st.pyplot(fig)


    col4, col5, col6 = st.columns(3)
    with col4:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Inventory Level", y="Demand",
                        color="orange", alpha=0.5, ax=ax)
        ax.set_title("Inventory Level vs Demand")
        st.pyplot(fig)

    with col5:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Promotion", y="Demand", ax=ax)
        ax.set_title("Demand Distribution: Promotion vs No Promotion")
        st.pyplot(fig)

    with col6:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Weather Condition", y="Demand", ax=ax)
        ax.set_title("Demand by Weather Condition")
        st.pyplot(fig)

# ===============================
# Forecasting Dashboard
# ===============================
elif page == "Forecasting Dashboard":
    st.title("📈 XGBoost Time Series Forecasting Dashboard")

    # Split train/validation (80/20)
    train_size = int(len(forecast_df) * 0.8)
    train_data = forecast_df.iloc[:train_size]
    val_data = forecast_df.iloc[train_size:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # -------------------------------
    # Training: Actual vs Forecast
    # -------------------------------
    axes[0,0].plot(train_data["Date"], train_data["Actual"], label="Train Actual", color="blue")
    axes[0,0].plot(train_data["Date"], train_data["Forecast"], label="Train Predicted", linestyle="--", color="orange")
    axes[0,0].set_title("Training: Actual vs Forecast")
    axes[0,0].legend()

    # -------------------------------
    # Validation: Actual vs Forecast
    # -------------------------------
    axes[0,1].plot(val_data["Date"], val_data["Actual"], label="Validation Actual", color="blue")
    axes[0,1].plot(val_data["Date"], val_data["Forecast"], label="Validation Predicted", linestyle="--", color="orange")

    # Add 95% Prediction Interval shading
    if "Lower_PI" in val_data.columns and "Upper_PI" in val_data.columns:
        axes[0,1].fill_between(val_data["Date"], val_data["Lower_PI"], val_data["Upper_PI"],
                               color="gray", alpha=0.3, label="95% PI")

    axes[0,1].set_title("Validation: Actual vs Forecast")
    axes[0,1].legend()

    # -------------------------------
    # Training Residuals
    # -------------------------------
    residuals = train_data["Actual"].values - train_data["Forecast"].values
    axes[1,0].plot(train_data["Date"], residuals, color="green")
    axes[1,0].axhline(0, color="black", linestyle="--")
    axes[1,0].set_title("Training Residuals")

    # -------------------------------
    # Metrics Box
    # -------------------------------
    axes[1,1].axis("off")
    textstr = (
        f"📊 Training:\n"
        f"MAE = {metrics['MAE']:.2f}\n"
        f"RMSE = {metrics['RMSE']:.2f}\n"
        f"MAPE = {metrics['MAPE']:.2f}%\n\n"
        f"📊 Prediction Intervals:\n"
        f"Coverage = {metrics['PI_Coverage']:.2f}%\n"
        f"Avg Width = {metrics['Avg_Interval_Width']:.2f}"
    )
    axes[1,1].text(0.05, 0.5, textstr, fontsize=12,
                   verticalalignment="center", bbox=dict(facecolor="lightyellow", alpha=0.5))

    plt.tight_layout()
    st.pyplot(fig)
