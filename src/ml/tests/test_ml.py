"""
PrajnaAI — ML Module Tests
Tests for spam classifier and house price predictor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from driver import SpamClassifier, HousePricePredictor


class TestSpamClassifier:
    
    @pytest.fixture(scope="class")
    def trained_classifier(self):
        clf = SpamClassifier()
        clf.run()
        return clf

    def test_training_runs(self, trained_classifier):
        assert len(trained_classifier.results) == 4, "Should train 4 algorithms"

    def test_results_have_auc(self, trained_classifier):
        for name, result in trained_classifier.results.items():
            assert "auc" in result
            assert 0.5 <= result["auc"] <= 1.0, f"{name}: AUC must be between 0.5 and 1.0"

    def test_best_model_selected(self, trained_classifier):
        assert trained_classifier.best_model is not None
        assert trained_classifier.best_name != ""

    def test_spam_prediction(self, trained_classifier):
        spam_msg = "WIN a FREE prize call 08001234567 NOW unlimited cash reward"
        result = trained_classifier.predict(spam_msg)
        assert result["prediction"] in ["SPAM", "HAM"]
        assert isinstance(result["confidence"], float)

    def test_ham_prediction(self, trained_classifier):
        ham_msg = "Hey, are you free for lunch tomorrow?"
        result = trained_classifier.predict(ham_msg)
        assert result["prediction"] in ["SPAM", "HAM"]

    def test_auc_threshold(self, trained_classifier):
        """At least one algorithm should achieve AUC > 0.90 on this dataset."""
        max_auc = max(r["auc"] for r in trained_classifier.results.values())
        assert max_auc > 0.90, f"Best AUC {max_auc:.4f} is too low"

    def test_cv_f1_threshold(self, trained_classifier):
        """Cross-validated F1 should be reasonable."""
        max_f1 = max(r["cv_f1_mean"] for r in trained_classifier.results.values())
        assert max_f1 > 0.85, f"Best CV F1 {max_f1:.4f} is too low"


class TestHousePricePredictor:

    @pytest.fixture(scope="class")
    def trained_predictor(self):
        pred = HousePricePredictor()
        pred.run()
        return pred

    def test_training_runs(self, trained_predictor):
        assert len(trained_predictor.models) == 2

    def test_r2_positive(self, trained_predictor):
        for name, data in trained_predictor.models.items():
            assert data["r2"] > 0.7, f"{name}: R² too low ({data['r2']:.4f})"

    def test_prediction_positive(self, trained_predictor):
        features = {
            "area_sqft": 1200, "bedrooms": 3, "bathrooms": 2,
            "age_years": 10, "distance_center_km": 5.0,
            "has_garage": 1, "has_garden": 1, "floor_level": 2
        }
        price = trained_predictor.predict(features)
        assert price > 0, "Price should be positive"
        assert price < 1000, "Price in Lakh should be reasonable"

    def test_larger_house_costs_more(self, trained_predictor):
        small = {"area_sqft": 600, "bedrooms": 2, "bathrooms": 1,
                 "age_years": 15, "distance_center_km": 10.0,
                 "has_garage": 0, "has_garden": 0, "floor_level": 1}
        large = {"area_sqft": 2500, "bedrooms": 5, "bathrooms": 3,
                 "age_years": 3, "distance_center_km": 2.0,
                 "has_garage": 1, "has_garden": 1, "floor_level": 0}
        small_price = trained_predictor.predict(small)
        large_price = trained_predictor.predict(large)
        assert large_price > small_price, "Larger/better house should cost more"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
