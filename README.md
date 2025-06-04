# German Credit Risk Classification with Clustering-Enhanced Models

This project focuses on building a credit risk classification model using the German Credit dataset. It incorporates standard machine learning classifiers (Logistic Regression, Random Forest, XGBoost) and enhances performance by adding clustering-based features using HDBSCAN.

## 📌 Project Goals

- Predict credit risk (Good/Bad) using customer financial and demographic data.
- Evaluate classical machine learning models with and without cluster-based features.
- Use hyperparameter tuning and cross-validation for model selection.
- Compare model performance using accuracy, recall, F1-score, and ROC-AUC.
- Save the best-performing model for deployment.

---

## 📂 Dataset

- Source: UCI German Credit Data
- Records: 1000
- Features: 11 (categorical and numerical)
- Target: Risk (good = 1, bad = 0)

---

## 📊 Exploratory Data Analysis (EDA)

- Missing value imputation for ‘Saving accounts’ and ‘Checking account’.
- Simplified and grouped ‘Purpose’ feature into:
  - essential
  - conditional_essential
  - nonEssential
- Visualizations using Plotly and Seaborn:
  - Box plots for numeric vs. categorical features
  - Histograms for distributions

---

## 🛠️ Preprocessing

- Categorical encoding using OneHotEncoder
- Numerical feature scaling using StandardScaler / RobustScaler
- ColumnTransformer for feature engineering pipeline
- Train-test split with stratification on the target variable

---

## 🤖 Models Used

1. Logistic Regression (with class balancing)
2. Random Forest Classifier (with GridSearchCV)
3. XGBoost Classifier (with scale_pos_weight for imbalance)

All models were evaluated using:
- Training performance metrics
- Grid search with Stratified K-Fold cross-validation
- Final evaluation on accuracy, recall, F1, and ROC-AUC

---

## 🧪 Clustering Augmentation

- Applied HDBSCAN on preprocessed numerical data
- Transformed cluster labels into new features
- Added these features to model pipelines via FunctionTransformer

Models retrained and tuned:
- Logistic Regression + HDBSCAN cluster
- Random Forest + HDBSCAN cluster
- XGBoost + HDBSCAN cluster

---

## 📈 Results

Cross-validated performance (Mean ± Std):

| Model                  | Accuracy | Recall | F1 Score | ROC-AUC |
|------------------------|----------|--------|----------|---------|
| Logistic Regression    |   0.694     |   0.679   |  0.756    |    0.758   |
| Logistic + Clustering  |  0.696    |  0.724   |  0.768     |    0.758   |
| Random Forest          |  0.743     |    0.896    |  0.830     |   0.750    |
| Random Forest + Clust  |   0.740      |   0.934   | 0.834     |   0.745   |
| XGBoost                |   0.713     |   0.767    |  0.788      |   0.749    |
| XGBoost + Clustering   |  0.717      |  0.809   |  0.799     |    0.730    |

📌 Observation: Cluster-enhanced models improved recall and F1 score significantly.

---

## 💾 Model Saving

- Best-performing model (Random Forest with HDBSCAN features) saved using pickle:
  - File: best_model.pkl

---

## 📚 Libraries Used

- pandas, numpy
- scikit-learn
- xgboost
- hdbscan
- seaborn, matplotlib
- plotly

---

## 📈 Visualizations

- Interactive performance plots via Plotly
- PCA scatter plot of HDBSCAN clusters
- Confusion matrices and classification reports per model

---

## 📌 Future Improvements

- Hyperparameter tuning with Bayesian optimization
- Model interpretability using SHAP
- Deployment using Streamlit or Flask
"""
