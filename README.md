# ⚙️ Industrial Copper Modeling


### Video Link   https://www.linkedin.com/posts/vinod-r-467a48276_project-5-industrial-copper-modeling-ugcPost-7484861751298056192-3FZi/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAENuFI8BEqwFM8YjuiKM8CRU5uMliVj9tpo
---

## 🚀 Project Overview
This project focuses on analyzing and modeling **industrial copper sales and pricing data**.  
The dataset contains transactional and product details, but suffers from **skewness, noise, and outliers**.  
The goal is to build:
- A **Regression Model** to predict `Selling_Price`  
- A **Classification Model** to predict `Status` (WON/LOST)  
- An **interactive Streamlit app** where users can input values and get predictions in real time.

---

## 🛠️ Skills Takeaway
- Python scripting  
- Data Preprocessing & Cleaning  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Regression & Classification Modeling  
- Streamlit Web Application Development  

---

## 🌐 Domain
**Manufacturing / Industrial Analytics**

---

## 📌 Problem Statement
The copper industry faces challenges in:
- **Pricing predictions**: Skewed and noisy data makes manual predictions unreliable.  
- **Lead classification**: Identifying which leads are likely to convert (WON vs LOST).  

This project addresses these issues by:
- Cleaning and preprocessing the dataset.  
- Building ML models for regression and classification.  
- Deploying predictions via a **Streamlit GUI**.  

---

## ✅ Approach
1. **Data Understanding**  
   - Identify variable types (continuous, categorical).  
   - Handle rubbish values in `material_ref` (e.g., starting with `00000`).  
   - Drop irrelevant columns like `id`.  

2. **Data Preprocessing**  
   - Handle missing values (mean/median/mode).  
   - Treat outliers using IQR or Isolation Forest.  
   - Address skewness with log/Box-Cox transformations.  
   - Encode categorical variables (one-hot, label, or ordinal encoding).  

3. **Exploratory Data Analysis (EDA)**  
   - Visualize skewness and outliers using boxplots, violin plots, histograms.  
   - Correlation analysis with heatmaps.  

4. **Feature Engineering**  
   - Create new features if applicable.  
   - Drop highly correlated variables.  

5. **Model Building & Evaluation**  
   - Regression: Predict `Selling_Price` using tree-based models.  
   - Classification: Predict `Status` (WON/LOST) using Logistic Regression, ExtraTrees, XGBClassifier.  
   - Evaluate with metrics: Accuracy, Precision, Recall, F1, AUC (for classification); RMSE, R² (for regression).  
   - Hyperparameter tuning with GridSearchCV / Cross-validation.  

6. **Streamlit GUI**  
   - Input form for all features.  
   - Option to choose **Regression or Classification task**.  
   - Predict and display results dynamically.  
   - Models, encoders, and scalers stored with **pickle** for reuse.  

---

## 🎯 Results
- A **Regression model** that predicts `Selling_Price`.  
- A **Classification model** that predicts lead status (WON/LOST).  
- An **interactive Streamlit app** for real-time predictions.  
- Improved accuracy and reliability compared to manual methods.  
