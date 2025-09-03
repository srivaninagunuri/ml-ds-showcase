# 🚀 Srivani’s Data Science Portfolio | From Learning to Job-Ready  

Welcome to my Data Science Portfolio!  
I am **Srivani Nagunuri**, a certified **Data Science professional from IIT Madras (GUVI)** and an **aspiring Data Scientist**.  
This portfolio showcases my **top 5 end-to-end projects** that reflect real-world business problems,
hands-on implementation, and industry-ready solutions.  

💡 **Career Goal:** To leverage **Machine Learning, Deep Learning, NLP, and Generative AI**
to deliver impactful insights and business value.  

---

## 📋 Table of Contents  
- 🎯 [Why These Projects?](#-why-these-projects)  
- 🔥 [Project Showcase](#-project-showcase)  
  1. [Telecom Customer Churn Prediction](#1-telecom-customer-churn-prediction)  
  2. [GenAI Research Paper Insight Extractor](#2-genai-research-paper-insight-extractor)  
  3. [Retail Demand Forecasting (Time Series Pipeline)](#3-retail-demand-forecasting-time-series-pipeline)  
  4. [Twitter Sentiment Analysis Dashboard](#4-twitter-sentiment-analysis-dashboard)  
  5. [Heart Disease Prediction (Neural Network)](#5-heart-disease-prediction-neural-network)  

---

## 🎯 Why These Projects?  

These projects are carefully selected to demonstrate skills that hiring managers actively seek in 2025:  

✅ **Real-world Business Focus** – Each project solves a real business problem  
✅ **End-to-End Implementation** – Covers the pipeline: from data collection → cleaning → modeling → deployment  
✅ **Modern Tech Stack** – Uses industry tools (Python, ML, DL, NLP, Generative AI, Streamlit, SQL)  
✅ **Production-Ready Code** – Clean, modular, and scalable  
✅ **Diverse Skill Set** – Time Series, Classification, NLP, Dashboards, and GenAI  
✅ **Each project = 1 interview story** 📈  

---

## 🔥 Project Showcase  

### 1. Telecom Customer Churn Prediction 
Predicting customer churn to reduce revenue loss and improve retention strategies


<img width="991" height="174" alt="image" src="https://github.com/user-attachments/assets/cb03f0a6-9f9e-4f65-93cd-386fc4ca69bf" />
<img width="991" height="174" alt="image" src="https://github.com/user-attachments/assets/e42df50f-ad12-4d84-90e1-a7848ee5d7a1" />







**📉 Business Problem:**  
Telecom companies lose millions annually due to customer churn. Retaining existing customers is 5x cheaper than acquiring new ones. This project identifies at-risk customers before they leave.  

**🎯 Target Metrics:**  
- Recall > 75% for high-value customers  
- Precision > 80% to avoid customer annoyance  
- F1-Score > 77% for balanced performance  
- Estimated Business Impact: **$2.3M revenue saved annually**  

**📊 Dataset Source:**  
- Telco Customer Churn Dataset (7,043 customers, 21 features)  
- Features: Demographics, services, account info, charges  
- Target: Binary classification (Churn: Yes/No)  
- Class Distribution: 73.5% retained, 26.5% churned
- ## 📊 Dataset Source

**Retail Store Demand Forecast**  
- Size: 76,000 rows (2 years of daily data)  
- Forecast Target: Demand (units sold per day)  
- Features: Historical demand, promotion flags, discount rates, inventory levels  
- External Data: Weather, weather condition, holiday indicators  
- Engineered Features: Lag values, rolling averages, standard deviations  

📂 **Dataset:** [Download from Kaggle](https://www.kaggle.com/datasets/srivani nagunuri/Telecom Churn Dataset)


**🔧 Technical Highlights:**  
- Advanced feature engineering on customer behavior  
- Class imbalance handled using **SMOTE + cost-sensitive learning**  
- Ensemble methods: **Random Forest, XGBoost, Logistic Regression**  
- Bayesian hyperparameter optimization  
- Customer segmentation + threshold tuning for business KPIs  

**🛠 Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, Streamlit, Plotly  
📸 **Output Preview:**  

**Confusion Matrix**
<img width="993" height="419" alt="image" src="https://github.com/user-attachments/assets/a7e95c2f-5eb3-4c9c-b567-df7d0317db42" />
<img width="848" height="218" alt="image" src="https://github.com/user-attachments/assets/b81f79ba-1fd5-4015-8075-ccc9fdfd4d99" />



---

### 2. GenAI Research Paper Insight Extractor  
**📄 Business Problem:** Researchers spend hours reading papers to extract insights. This tool enables **semantic search + Q&A** over uploaded research papers.  

**🎯 Target Metrics:**  
- 90%+ accuracy in Q&A tasks  
- 70%+ reduction in reading time  
- Multi-format support (PDF, Text, Docs)  

**🔧 Technical Highlights:**  
- Retrieval-Augmented Generation (**RAG**) architecture  
- Vector embeddings with **OpenAI + Sentence Transformers**  
- Semantic chunking + similarity search with **ChromaDB**  
- Interactive **Streamlit interface**  

**🛠 Tech Stack:** Python, LangChain, OpenAI API, Streamlit, ChromaDB  

📸 Output Preview:

### AI Research Paper Insight Extractor

<img width="1083" height="400" alt="image" src="https://github.com/user-attachments/assets/e7922493-fec5-4ffa-b434-eeb3e3fc21a4" />
<img width="1079" height="497" alt="image" src="https://github.com/user-attachments/assets/0129d67b-7d57-42ff-a964-ad4428ea83c5" />


---

### 3. Retail Demand Forecasting (Time Series Pipeline) 
### Retail Store Inventory & Demand Dashboard
<img width="1053" height="268" alt="image" src="https://github.com/user-attachments/assets/003b54fd-9c40-483f-a004-cfea33dc30b6" />
<img width="1053" height="268" alt="image" src="https://github.com/user-attachments/assets/8e08d51e-9f2b-4396-8a3c-d7783f488be1" />


**📈 Business Problem:** Retailers face losses due to overstocking or stockouts. Accurate demand forecasting reduces inventory costs and improves sales.  

**🎯 Target Metrics:**  
- MAPE < 10% (short-term forecast 1–4 weeks)  
- RMSE improvement > 15% over baselines  
- 95% prediction intervals for uncertainty  

**🔧 Technical Highlights:**  
- Feature engineering: lags, rolling averages, seasonality decomposition  
- Models: **ARIMA, Prophet, LSTM, Transformer**  
- Automated hyperparameter tuning with **Optuna**  
- Deployment with **Streamlit Dashboard**  

**🛠 Tech Stack:** Python, TensorFlow, Prophet, Optuna, Streamlit, Plotly  

📸 *Output:* Forecasting Dashboard with 7-day & 28-day predictions  
<img width="804" height="465" alt="image" src="https://github.com/user-attachments/assets/5bf8aef1-3393-4f89-b044-e2913f9e04af" />


---

### 4. Twitter Sentiment Analysis Dashboard  
**🛒 Business Problem:** Companies need to track customer sentiment in real-time for **brand reputation management**. Manual tracking is slow and incomplete.  

**🎯 Target Metrics:**  
- Sentiment classification accuracy > 90%  
- Near real-time processing of tweets  
- Dashboard for visual insights  

**🔧 Technical Highlights:**  
- NLP preprocessing: tokenization, stopword removal, lemmatization  
- Sentiment classification with **Transformers (BERT)**  
- Data visualization with **Plotly & Streamlit**  
- Time series trend analysis of sentiment  

**🛠 Tech Stack:** Python, Transformers, Scikit-learn, Streamlit, Plotly  

📸 *Output:* Live Sentiment Dashboard  

---

### 5. Heart Disease Prediction (Neural Network)  

**🏥 Business Problem:** Early detection of heart disease reduces healthcare costs and saves lives.  



<img width="768" height="530" alt="image" src="https://github.com/user-attachments/assets/c5562e46-08b8-429a-a3ef-a2680c2c6c9f" />

**🎯 Target Metrics:**  
- Sensitivity > 90% (minimize false negatives)  
- AUC-ROC > 0.92 overall performance  

**🔧 Technical Highlights:**  
- Dataset preprocessing + feature scaling  
- Deep Neural Network (**Keras + TensorFlow**)  
- Model evaluation with ROC curves, precision-recall  
- Deployment with **Streamlit app**  

**🛠 Tech Stack:** Python, TensorFlow, Keras, Scikit-learn, Streamlit  

📸 *Output:* Heart Disease Risk Prediction Dashboard  

---

## 🤝 Connect  

📌 **GitHub:** [github.com/srivani](https://github.com/)  
📌 **LinkedIn:** [linkedin.com/in/srivani](https://linkedin.com/)  
📌 **Portfolio Dashboard Apps:** (add Streamlit/other links if hosted)  

---

⭐ If this portfolio inspires you, consider giving it a star!  
