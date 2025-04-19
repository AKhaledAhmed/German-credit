# German-credit
# German Credit Risk Classification

This project performs an end-to-end classification task on the **German Credit Risk Dataset** to predict whether a customer is a **good or bad credit risk** based on various demographic and financial features.

---

## 📌 Objective

To build and compare classification models that can accurately predict credit risk, helping financial institutions make informed lending decisions.

---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Size**: 1,000 records, 21 features
- **Target**: `Risk` (`good` or `bad`)
- **Features include**: Age, Sex, Housing, Credit Amount, Duration, Checking/Savings Account info, Purpose of the loan, etc.

---

## 🧹 Data Preprocessing

- Removed unnecessary columns
- Filled missing values with meaningful placeholders
- Recoded categorical features using ordinal and binary mapping
- Grouped `Purpose` values into:
  - `Essential`
  - `Conditional Essential`
  - `Non-Essential`
- Addressed class imbalance using `RandomOverSampler`

---

## 📈 Exploratory Data Analysis (EDA)

- Distribution of features by credit risk
- Boxplots for `Duration`, `Credit Amount`, and `Age`
- Feature correlations with the target variable

---

## 🤖 Models Used

1. **Logistic Regression** (baseline with `class_weight=balanced`)
2. **Random Forest Classifier**
3. **XGBoost Classifier**

Each model was evaluated using:
- Accuracy
- Recall
- F1-score
- Confusion Matrix
- Feature Importance

---

## 🔍 Model Evaluation

| Model                | Accuracy | Recall | F1-score |
|---------------------|----------|--------|----------|
| Random Forest        | ~        | ~      | ~        |
| XGBoost              | ~        | ~      | ~        |
| Logistic Regression  | ~        | ~      | ~        |

> _(Exact values depend on the dataset split and can be seen in the final visualization cell of the notebook.)_

All scores were visualized using Plotly for easy comparison.

---

## 🧰 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `plotly`
- `sklearn`
- `xgboost`
- `imblearn`

---
