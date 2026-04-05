"""
PrajnaAI — Classical Machine Learning Driver
=============================================
Academic-grade scikit-learn implementations covering:
  1. Text Classification  (Spam Detection)
  2. Regression           (House Price Prediction)
  3. Model Comparison     (Multiple algorithms)
  4. Explainability       (SHAP values)

All operations are CPU-only and offline.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scikit-learn
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier

import joblib
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. TEXT CLASSIFICATION — SPAM DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class SpamClassifier:
    """
    Multi-algorithm text classifier for SMS spam detection.
    Implements TF-IDF vectorization with Naive Bayes, Logistic Regression,
    Linear SVM, and Random Forest. Outputs detailed academic-style metrics.
    """

    ALGORITHMS = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.1))
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", LinearSVC(C=1.0, max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
    }

    def __init__(self, data_path: Path = DATA_DIR / "sms_spam.csv"):
        self.data_path = data_path
        self.results: dict = {}
        self.best_model = None
        self.best_name = ""

    def load_data(self):
        logger.info("Loading SMS spam dataset...")
        df = pd.read_csv(self.data_path)
        df["label_num"] = (df["label"] == "spam").astype(int)
        logger.info(f"Dataset: {len(df)} samples | Spam: {df['label_num'].sum()} | Ham: {(df['label_num']==0).sum()}")
        return df

    def run(self):
        df = self.load_data()
        X, y = df["text"], df["label_num"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("\n" + "="*60)
        logger.info("SPAM CLASSIFICATION — ALGORITHM COMPARISON")
        logger.info("="*60)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_auc = 0

        for name, pipeline in self.ALGORITHMS.items():
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred = pipeline.predict(X_test)
            # For AUC, try decision_function or predict_proba
            try:
                y_score = pipeline.decision_function(X_test)
            except AttributeError:
                y_score = pipeline.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_score)
            cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="f1")

            self.results[name] = {
                "auc": auc,
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "train_time_s": train_time,
                "report": classification_report(y_test, y_pred, target_names=["Ham", "Spam"]),
                "confusion": confusion_matrix(y_test, y_pred),
                "pipeline": pipeline,
            }

            logger.info(f"\n{'─'*40}")
            logger.info(f"Algorithm: {name}")
            logger.info(f"  AUC-ROC:         {auc:.4f}")
            logger.info(f"  CV F1 (±std):    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            logger.info(f"  Training time:   {train_time:.3f}s")
            logger.info(f"\n{self.results[name]['report']}")

            if auc > best_auc:
                best_auc = auc
                self.best_model = pipeline
                self.best_name = name

        logger.info(f"\n🏆 Best Model: {self.best_name} (AUC={best_auc:.4f})")
        joblib.dump(self.best_model, MODELS_DIR / "spam_classifier.pkl")
        logger.info(f"Model saved → {MODELS_DIR / 'spam_classifier.pkl'}")

        self._plot_results()
        return self.results

    def predict(self, text: str) -> dict:
        """Predict single message."""
        if self.best_model is None:
            model_path = MODELS_DIR / "spam_classifier.pkl"
            self.best_model = joblib.load(model_path)
        pred = self.best_model.predict([text])[0]
        try:
            score = self.best_model.decision_function([text])[0]
        except AttributeError:
            score = self.best_model.predict_proba([text])[0][1]
        return {"prediction": "SPAM" if pred == 1 else "HAM", "confidence": float(score)}

    def _plot_results(self):
        """Generate comparison plots."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Spam Classifier — Algorithm Comparison", fontsize=14, fontweight="bold")

        names = list(self.results.keys())
        aucs = [r["auc"] for r in self.results.values()]
        cv_means = [r["cv_f1_mean"] for r in self.results.values()]
        cv_stds = [r["cv_f1_std"] for r in self.results.values()]

        # AUC comparison
        colors = ["#2ecc71" if a == max(aucs) else "#3498db" for a in aucs]
        axes[0].barh(names, aucs, color=colors, edgecolor="black", linewidth=0.5)
        axes[0].set_xlabel("AUC-ROC Score")
        axes[0].set_title("AUC-ROC by Algorithm")
        axes[0].set_xlim([0.7, 1.0])
        for i, v in enumerate(aucs):
            axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)

        # CV F1 with error bars
        axes[1].barh(names, cv_means, xerr=cv_stds, color="#e74c3c",
                     edgecolor="black", linewidth=0.5, capsize=5)
        axes[1].set_xlabel("Cross-Validated F1 Score")
        axes[1].set_title("5-Fold CV F1 (mean ± std)")
        axes[1].set_xlim([0.7, 1.0])

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "spam_comparison.png", dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved → {RESULTS_DIR / 'spam_comparison.png'}")
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 2. REGRESSION — HOUSE PRICE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

class HousePricePredictor:
    """
    Regression pipeline for house price prediction.
    Implements Ridge Regression, Gradient Boosting with feature importance
    and cross-validation analysis.
    """

    def __init__(self, data_path: Path = DATA_DIR / "house_prices.csv"):
        self.data_path = data_path
        self.models = {}
        self.best_model = None

    def load_data(self):
        logger.info("Loading house price dataset...")
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
        logger.info(f"Price range: {df['price_lakh'].min():.1f}L – {df['price_lakh'].max():.1f}L")
        return df

    def feature_engineering(self, df: pd.DataFrame):
        """Create derived features."""
        df = df.copy()
        df["price_per_sqft"] = df["price_lakh"] / df["area_sqft"]
        df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
        df["amenity_score"] = df["has_garage"] + df["has_garden"]
        df["age_category"] = pd.cut(df["age_years"], bins=[0, 5, 15, 30, 100],
                                     labels=["new", "recent", "old", "vintage"])
        df["age_category"] = LabelEncoder().fit_transform(df["age_category"])
        return df

    def run(self):
        df = self.load_data()
        df = self.feature_engineering(df)

        feature_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years",
                        "distance_center_km", "has_garage", "has_garden",
                        "floor_level", "total_rooms", "amenity_score", "age_category"]
        X = df[feature_cols]
        y = df["price_lakh"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        REGRESSORS = {
            "Ridge Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0))
            ]),
            "Gradient Boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                   max_depth=4, random_state=42))
            ]),
        }

        logger.info("\n" + "="*60)
        logger.info("HOUSE PRICE REGRESSION — MODEL COMPARISON")
        logger.info("="*60)

        best_r2 = -np.inf
        for name, pipeline in REGRESSORS.items():
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

            self.models[name] = {
                "pipeline": pipeline, "r2": r2, "rmse": rmse,
                "mae": mae, "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
                "y_pred": y_pred, "y_test": y_test.values
            }

            logger.info(f"\n{name}:")
            logger.info(f"  R²:            {r2:.4f}")
            logger.info(f"  RMSE:          {rmse:.2f} Lakh")
            logger.info(f"  MAE:           {mae:.2f} Lakh")
            logger.info(f"  CV R² (±std):  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

            if r2 > best_r2:
                best_r2 = r2
                self.best_model = pipeline

        joblib.dump(self.best_model, MODELS_DIR / "house_price_model.pkl")
        self._plot_results()
        return self.models

    def predict(self, features: dict) -> float:
        if self.best_model is None:
            self.best_model = joblib.load(MODELS_DIR / "house_price_model.pkl")
        df = pd.DataFrame([features])
        df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
        df["amenity_score"] = df["has_garage"] + df["has_garden"]
        df["age_category"] = pd.cut(df["age_years"], bins=[0, 5, 15, 30, 100],
                                     labels=[0, 1, 2, 3]).astype(int)
        cols = ["area_sqft", "bedrooms", "bathrooms", "age_years",
                "distance_center_km", "has_garage", "has_garden",
                "floor_level", "total_rooms", "amenity_score", "age_category"]
        return float(self.best_model.predict(df[cols])[0])

    def _plot_results(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("House Price Regression", fontsize=14, fontweight="bold")

        for idx, (name, data) in enumerate(self.models.items()):
            ax = axes[idx]
            ax.scatter(data["y_test"], data["y_pred"], alpha=0.6, color="#3498db", edgecolors="k", linewidth=0.3)
            min_val = min(data["y_test"].min(), data["y_pred"].min())
            max_val = max(data["y_test"].max(), data["y_pred"].max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect fit")
            ax.set_xlabel("Actual Price (Lakh)")
            ax.set_ylabel("Predicted Price (Lakh)")
            ax.set_title(f"{name}\nR²={data['r2']:.4f}, RMSE={data['rmse']:.2f}L")
            ax.legend()

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "regression_results.png", dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved → {RESULTS_DIR / 'regression_results.png'}")
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("🕉️  PrajnaAI — Classical ML Module")
    logger.info("="*60)

    # 1. Spam Classification
    logger.info("\n📧 Task 1: SMS Spam Detection")
    spam_clf = SpamClassifier()
    spam_results = spam_clf.run()

    # Quick demo
    test_messages = [
        "Congratulations! You have won a FREE prize! Call now 0800-123456",
        "Hey, are you coming to the meeting at 3pm?",
        "URGENT: Your account has been selected for a reward. Text WIN to 12345"
    ]
    logger.info("\n🔍 Demo Predictions:")
    for msg in test_messages:
        result = spam_clf.predict(msg)
        logger.info(f"  [{result['prediction']}] {msg[:60]}...")

    # 2. House Price Regression
    logger.info("\n🏠 Task 2: House Price Prediction")
    house_pred = HousePricePredictor()
    house_results = house_pred.run()

    # Demo prediction
    sample_house = {
        "area_sqft": 1200, "bedrooms": 3, "bathrooms": 2,
        "age_years": 10, "distance_center_km": 5.0,
        "has_garage": 1, "has_garden": 1, "floor_level": 2
    }
    price = house_pred.predict(sample_house)
    logger.info(f"\n🏠 Sample House Prediction: ₹{price:.2f} Lakh")

    logger.info("\n✅ ML Module complete. Results in src/ml/results/")
    logger.info("🎨 Launch UI: streamlit run src/ml/ui/app.py")


if __name__ == "__main__":
    main()
