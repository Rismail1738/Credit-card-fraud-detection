# 🔍 Credit Card Fraud Detection

A machine learning project built for FinGuard Analytics to detect 
fraudulent credit card transactions in real time.

---

## 📊 Project Overview

Credit card fraud is a massive global problem. This project builds 
a machine learning pipeline that can automatically identify 
fraudulent transactions from historical data with high accuracy.

The biggest challenge? Only **0.17% of transactions are fraud** — 
making this an extreme class imbalance problem.

---

## 📁 Dataset

- **Source:** Anonymized real credit card transactions
- **Size:** 284,807 transactions × 31 features
- **Fraud cases:** 492 (0.17%)
- **Features:** Time, Amount, and V1–V28 (PCA-anonymized by bank)
- **Missing values:** None

> ⚠️ The dataset is not included in this repo due to file size.
> Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## ⚙️ What This Project Does

| Step | Description |
|------|-------------|
| 📥 Data Loading | Load and explore the transaction dataset |
| 📊 EDA | Visualize fraud vs legitimate patterns |
| 🔧 Preprocessing | Scale Amount and Time features |
| ⚖️ SMOTE | Fix class imbalance with synthetic sampling |
| 🤖 Modelling | Train Logistic Regression and Random Forest |
| 🔬 PCA | Reduce dimensions and compare performance |
| 🏆 Evaluation | Compare models using Precision, Recall, F1 |
| 🕵️ Insights | Identify most important fraud signals |

---

## 📈 Results

### Model Comparison

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Logistic Regression | 6% | 93% | 0.11 |
| **Random Forest** | **93%** | **82%** | **0.87** |

### PCA Impact

| | Without PCA | With PCA |
|--|-------------|----------|
| Precision | 93% | 63% |
| Recall | 82% | 85% |
| F1 Score | **0.87** | 0.72 |

---

## 🕵️ Key Findings

- **V14 is the #1 fraud signal** — contributes 41.3% of detection power alone
- **Top 4 features** (V14, V17, V10, V12) account for 73% of all fraud detection
- **Transaction Amount** is surprisingly weak — only 1.2% importance
- **PCA reduces performance** — full 30 features gives better results
- **Accuracy is misleading** — always use Precision, Recall and F1 for imbalanced data

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

---

## 📦 Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

---

## 🚀 How To Run

1. Clone this repository
2. Download the dataset from Kaggle (link above)
3. Open the notebook in Google Colab
4. Run all cells from top to bottom

---

## 📄 Report

A full written report is included in this repository covering:
- Problem overview and dataset analysis
- Preprocessing steps and SMOTE explanation
- Model comparison and evaluation
- PCA impact analysis
- Feature importance findings
- Conclusions and recommendations

---

## 👤 Author

Made with 💙 as part of a Machine Learning assignment
