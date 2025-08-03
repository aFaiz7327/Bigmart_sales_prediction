
# 🛒 BigMart Sales Prediction: Advanced Weighted Ensemble System

This repository contains a comprehensive machine learning pipeline to predict sales for BigMart retail products. It includes robust data preprocessing, advanced feature engineering, multi-model training with hyperparameter tuning, and a weighted ensemble for final predictions.

---

## 📊 Project Overview

The aim of this project is to predict the sales of various products across different BigMart outlets using historical sales data. To enhance prediction accuracy, a wide suite of regression models are evaluated, tuned, and the top performers are combined via weighted ensembling.

---

## 🚀 Key Features

- **Advanced Missing Value Imputation**: Context-aware filling based on `Item_Identifier`, `Item_Type`, and `Outlet_Type`.
- **Feature Engineering**: Rich set of time-based, categorical, interaction, and statistical features.
- **Model Zoo**: Over a dozen regression models including XGBoost, Random Forest, Ridge, SVR, MLP, etc.
- **Hyperparameter Tuning**: Performed using `RandomizedSearchCV` for top 3 models.
- **Ensemble Learning**: Final predictions are made via a weighted ensemble of the top 2 models.
- **Metrics Tracked**: RMSE, R², MAE for training and validation sets.

---

## 📁 Dataset

The pipeline uses the [BigMart Sales dataset](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-prediction/):

- `train_v9rqX0R.csv` – historical sales with target variable
- `test_AbJTz2l.csv` – test data without `Item_Outlet_Sales`

Place both files in the project directory to run the full pipeline.

---

## 🧱 Project Structure

```
📦 bigmart-ensemble
├── train_v9rqX0R.csv
├── test_AbJTz2l.csv
├── bigmart_pipeline.py
└── README.md
```

---

## 🛠️ Pipeline Stages

### 1. Load Data
```python
load_data(train, test)
```

### 2. Data Preprocessing
```python
preprocess_data(train_data, test_data)
```

### 3. Model Initialization
```python
initialize_models()
```

### 4. Model Training & Evaluation
```python
train_and_evaluate(models, X_train, y_train)
```

### 5. Weighted Ensemble
```python
predict_with_ensemble(ensemble_info, X_test)
```

### 6. Submission Generation
```python
create_submission(predictions, test_data)
```

### 7. Complete Pipeline
```python
main(train, test)
```

---

## 📈 Sample Output

- Submission file: `weighted_ensemble_submission_final.csv`
- RMSE, MAE, and R² are printed for all models and ensemble.

---

## ⚙️ Requirements

Install dependencies via pip:

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```

---

## 🧪 How to Run

```bash
python bigmart_pipeline.py
```

Ensure `train_v9rqX0R.csv` and `test_AbJTz2l.csv` are in the same directory.

---

## 📌 Notes

- Ensemble logic can be customized (e.g., stacking, voting).
- Code is modular: you can plug in other datasets or add models easily.
- Tuning is selective and efficient using `RandomizedSearchCV`.

---

## 📚 Reference

- Analytics Vidhya BigMart Sales Prediction Problem
- XGBoost Documentation
- Scikit-learn API Docs

---

## 📧 Contact

Feel free to raise issues or suggestions via GitHub. Contributions welcome!
