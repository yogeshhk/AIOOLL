# ML Module — Classical Machine Learning

## Overview

Academic-grade scikit-learn implementations covering the full ML pipeline:
feature engineering, model training, cross-validation, and explainability.

## Tasks

| Task | Algorithms | Dataset | Metric |
|------|-----------|---------|--------|
| Spam Detection | Naive Bayes, Logistic Regression, Linear SVM, Random Forest | SMS (50 samples) | AUC-ROC, F1 |
| House Price | Ridge Regression, Gradient Boosting | Synthetic (50 samples) | R², RMSE |

## Files

```
ml/
├── driver.py           # Main entry point — runs all tasks
├── ui/app.py           # Streamlit dashboard
├── data/
│   ├── sms_spam.csv    # SMS spam dataset
│   └── house_prices.csv # House price dataset
├── models/             # Saved model artifacts (auto-generated)
├── results/            # Plots and metrics (auto-generated)
└── tests/test_ml.py    # Pytest test suite
```

## Run

```bash
# Train all models
python driver.py

# Interactive UI
streamlit run ui/app.py

# Tests
pytest tests/ -v
```

## Key Academic Concepts

- **TF-IDF Vectorization:** Term Frequency–Inverse Document Frequency converts text to numerical features
- **Cross-Validation:** 5-fold stratified CV for unbiased performance estimates
- **Pipeline Architecture:** Prevents data leakage by encapsulating preprocessing + model
- **SHAP Values:** SHapley Additive exPlanations for model interpretability
