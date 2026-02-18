"""
Tests for model loading and inference — site_sentinel.features + the trained pkl file.

These tests are designed to pass in CI where the model pkl file is not present.
Set the SITE_SENTINEL_SKIP_MODEL environment variable to skip model-file tests,
or just let pytest skip them automatically when the file is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Path to the trained model relative to the project root
_MODEL_PATH = Path(__file__).parent.parent / "models" / "rf_master_predictor_dual_lead_tuned.pkl"
_SKIP_MODEL = (
    os.getenv("SITE_SENTINEL_SKIP_MODEL") == "true"
    or not _MODEL_PATH.exists()
)
_SKIP_REASON = (
    "Model file not present (run pipeline/04_train_random_forest.py first, "
    "or set SITE_SENTINEL_SKIP_MODEL=true to skip these tests in CI)."
)

FEATURE_COLUMNS = [
    "rel_distance", "rel_speed", "speed_ms_vuln", "speed_ms_car",
    "accel_ms2_vuln", "accel_ms2_car", "ttc", "approach_speed",
    "rel_dist_avg_2s", "rel_speed_avg_2s", "future_rel_dist_avg_2s",
]


@pytest.fixture(scope="module")
def model_dict() -> dict:
    """Load the pkl model dict once for all tests in this module."""
    import joblib
    return joblib.load(_MODEL_PATH)


@pytest.mark.skipif(_SKIP_MODEL, reason=_SKIP_REASON)
class TestModelFile:
    def test_model_file_exists(self) -> None:
        assert _MODEL_PATH.exists(), f"Model file missing: {_MODEL_PATH}"

    def test_model_dict_has_preventive_key(self, model_dict: dict) -> None:
        assert "preventive" in model_dict, (
            f"Expected 'preventive' key in model dict. Got: {list(model_dict.keys())}"
        )

    def test_model_dict_has_standard_key(self, model_dict: dict) -> None:
        assert "standard" in model_dict, (
            f"Expected 'standard' key in model dict. Got: {list(model_dict.keys())}"
        )

    def test_preventive_model_has_predict_proba(self, model_dict: dict) -> None:
        model = model_dict["preventive"]
        assert hasattr(model, "predict_proba"), "Model must implement predict_proba"

    def test_feature_count_is_eleven(self, model_dict: dict) -> None:
        model = model_dict["preventive"]
        if hasattr(model, "feature_names_in_"):
            assert len(model.feature_names_in_) == 11, (
                f"Expected 11 features, got {len(model.feature_names_in_)}"
            )
        elif hasattr(model, "n_features_in_"):
            assert model.n_features_in_ == 11, (
                f"Expected 11 features, got {model.n_features_in_}"
            )

    def test_predict_output_shape(self, model_dict: dict) -> None:
        """Two sample rows → predict_proba shape should be (2, 2)."""
        model = model_dict["preventive"]
        X = pd.DataFrame(
            [
                {c: 5.0 if "dist" in c else 1.0 for c in FEATURE_COLUMNS},
                {c: 2.0 if "dist" in c else 0.5 for c in FEATURE_COLUMNS},
            ]
        )
        proba = model.predict_proba(X)
        assert proba.shape == (2, 2), f"Unexpected output shape: {proba.shape}"

    def test_predict_probabilities_in_range(self, model_dict: dict) -> None:
        """All predicted probabilities must be in [0, 1]."""
        model = model_dict["preventive"]
        X = pd.DataFrame([{c: np.random.uniform(0, 10) for c in FEATURE_COLUMNS}
                          for _ in range(10)])
        proba = model.predict_proba(X)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()
