# Banking Fraud Prediction

This project focuses on detecting fraudulent transactions in banking data using both **unsupervised learning (clustering)** and **supervised learning (classification)** methods. It was built as part of the final submission for the BMLP Machine Learning Cohort by DBS Coding Camp and Dicoding.

## 🚀 Project Objectives

- Identify hidden patterns in transaction data using clustering (K-Means)
- Build classification models to detect fraud using Logistic Regression and Random Forest
- Evaluate and compare model performance based on key classification metrics
- Demonstrate the use of machine learning to enhance fraud detection in the financial sector

## 📊 Dataset

The dataset contains anonymized banking transaction data with both legitimate and fraudulent transactions. It includes features such as transaction type, amount, balance changes, and is labeled with a binary target for fraud detection.

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## 🔍 Project Workflow

### 1. Data Preprocessing
- Removed duplicate and irrelevant features
- Handled missing values and outliers
- Scaled data for clustering and modeling

### 2. Clustering (Unsupervised Learning)
- Applied K-Means clustering to explore transaction patterns
- Evaluated with silhouette score (~0.6)

### 3. Classification (Supervised Learning)
- Built two classification models:
  - Logistic Regression
  - Random Forest
- Evaluated with accuracy, precision, recall, and F1-score
- Random Forest achieved accuracy > 92%

## 📈 Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression|   ~90%   |    High   |  High  |  High    |
| Random Forest      | **>92%** |    High   |  High  |  High    |

## 📌 Key Takeaways

- Clustering helped uncover potential transaction segments
- Random Forest outperformed Logistic Regression in identifying fraud
- Machine learning can play a crucial role in proactive fraud detection for financial institutions
