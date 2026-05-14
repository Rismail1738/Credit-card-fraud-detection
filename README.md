# 🔍 Credit Card Fraud Detection

> A complete end-to-end machine learning pipeline to detect fraudulent credit card transactions in real time — built for FinGuard Analytics.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

---

## 📌 Project Overview

Credit card fraud costs the global economy billions of dollars every year. This project builds a machine learning system that automatically identifies fraudulent transactions from historical data with high accuracy and real-world business impact.

The core challenge: **only 0.17% of transactions are fraud** — making this an extreme class imbalance problem that requires careful handling.

---


## 📊 Dataset

| Property | Detail |
|----------|--------|
| Total Transactions | 284,807 |
| Features | 31 (Time, Amount, V1–V28, Class) |
| Fraud Cases | 492 (0.17%) |
| Legitimate Cases | 284,315 (99.83%) |
| Missing Values | None |

> ⚠️ Dataset not included due to file size limits.
> Download it here → [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Feature Notes:**
- `V1–V28` are anonymized by the bank using PCA for privacy
- `Amount` is the transaction value in dollars
- `Time` is seconds elapsed since the first transaction
- `Class` is the target: 0 = Legitimate, 1 = Fraud

---

## ⚙️ Pipeline Overview

```
Raw Data
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Scaling (StandardScaler)
   ↓
Train/Test Split (80/20)
   ↓
SMOTE (Fix Class Imbalance)
   ↓
Model Training × 3
   ↓
Evaluation (Precision, Recall, F1, AUC)
   ↓
PCA Analysis
   ↓
Feature Importance
   ↓
Business Impact Analysis
   ↓
Real-Time Transaction Predictor
```

---

## 🤖 Models Built

### 1. Logistic Regression (Baseline)
A simple linear model used as the baseline for comparison.

### 2. Random Forest 🏆
An ensemble of decision trees that votes together for a final prediction. Best model for deployment.

### 3. XGBoost (Tuned)
A gradient boosting model with hyperparameter tuning. Best model for risk scoring.

---

## 📈 Results

### Model Performance Comparison

| Model | Fraud Precision | Fraud Recall | Fraud F1 | AUC |
|-------|----------------|--------------|----------|-----|
| Logistic Regression | 6% | 93% | 0.11 | 0.979 |
| **Random Forest** ✅ | **93%** | **82%** | **0.87** | 0.942 |
| XGBoost Tuned | 20% | 90% | 0.32 | **0.988** |

### Cross Validation (5-Fold)

| Model | Average F1 | Consistency |
|-------|-----------|-------------|
| Logistic Regression | 0.948 | ±0.001 |
| **Random Forest** | **1.000** | **±0.000** |
| XGBoost Tuned | 0.990 | ±0.001 |

### PCA Impact on Random Forest

| Metric | Without PCA | With PCA |
|--------|-------------|----------|
| Precision | 93% | 63% |
| Recall | 82% | 85% |
| F1 Score | **0.87** | 0.72 |

> PCA reduces performance — full 30 features recommended for deployment.

---

## 💰 Business Impact Analysis

*Based on 56,962 test transactions with average fraud value of $122.21*

| | Random Forest | XGBoost |
|--|--------------|---------|
| Fraud Cases Caught | 80 | 88 |
| Fraud Cases Missed | 18 | 10 |
| Innocent Customers Wrongly Flagged | **6** ✅ | 358 ❌ |
| Money Saved | $9,776.80 | $10,754.48 |
| Money Lost | $2,199.78 | $1,222.10 |

> 💡 **Key Insight:** While XGBoost saves $977 more, it wrongly flags 358 innocent customers vs only 6 for Random Forest. For customer experience and trust, **Random Forest is the better business choice.**

---

## 🕵️ Feature Importance

| Rank | Feature | Importance | Signal Strength |
|------|---------|------------|----------------|
| 1 | **V14** | 41.3% | 🔴 Dominant |
| 2 | V17 | 13.7% | 🟠 Strong |
| 3 | V10 | 10.0% | 🟠 Strong |
| 4 | V12 | 7.8% | 🟡 Significant |
| 5 | V2 | 5.0% | 🟡 Moderate |
| 9 | Amount | 1.2% | 🟢 Weak |

> 💡 V14 alone drives 41.3% of all fraud detection decisions. The top 4 features combined account for 73% of detection power. Transaction Amount is surprisingly weak at just 1.2%.

---

## 🔑 Key Findings

- **Extreme imbalance** (0.17% fraud) requires SMOTE to train effectively
- **Random Forest** is the best deployment model with 93% precision and 0.87 F1
- **XGBoost** is the best risk scoring model with 0.988 AUC
- **PCA hurts performance** — precision drops from 93% to 63%
- **V14 is the #1 fraud signal** contributing 41.3% of detection power
- **Amount is a weak signal** despite fraud averaging $122 vs $88 for legit
- **Accuracy is misleading** — always use Precision, Recall and F1 for imbalanced data
- **AUC and F1 tell different stories** — choosing the right metric depends on business needs

---

## 🛠️ How To Run

### Option 1 — Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Download the dataset from Kaggle (link above)
3. Upload `creditcard.csv` to Colab
4. Run all cells from top to bottom

### Option 2 — Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook fraud_detection.ipynb
```

---

## 📦 Dependencies

```
pandas==2.1.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==1.7.6
imbalanced-learn==0.11.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 Full Report

A complete written report (`Fraud_Detection_Report.docx`) is included covering:
- Problem overview and business context
- Dataset analysis and class imbalance discussion
- Preprocessing steps and SMOTE explanation
- Model comparison and evaluation
- PCA impact analysis
- Feature importance findings
- Business impact calculations
- Conclusions and recommendations

---

## 💡 Future Improvements

- [ ] Investigate the meaning of V14 with domain experts
- [ ] Try Neural Networks for fraud detection
- [ ] Build a REST API for real-time predictions
- [ ] Add a dashboard for monitoring fraud patterns
- [ ] Explore cost-sensitive learning approaches
- [ ] Remove low importance features (V22, V24, V27)

---

## 👤 Author

Made with 💙 as part of a Machine Learning assignment at FinGuard Analytics
