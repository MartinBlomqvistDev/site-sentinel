"""
03a_train_xgboost.py — XGBoost baseline trained on the master dataset.

Runs 5-fold stratified cross-validation with SMOTE resampling and
RandomizedSearchCV hyperparameter tuning, then saves the final model.

Usage:
    python -m pipeline.03a_train_xgboost

Input:
    data/analysis_results/master_training_dataset_full.csv
Output:
    models/xgb_risk_predictor_tuned.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from site_sentinel.config import load_config  # noqa: E402
from site_sentinel.features.targets import create_dual_targets  # noqa: E402
from site_sentinel.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_cfg = load_config("model_training")
_rf_cfg = _cfg["random_forest"]
_cv_cfg = _cfg["cross_validation"]
_tgt_cfg = _cfg["targets"]
_xgb_cfg = _cfg.get("xgboost", {})

MASTER_CSV: str = _rf_cfg["master_csv"]
OUTPUT_MODEL: str = "models/xgb_risk_predictor_tuned.pkl"
FEATURES: list[str] = _cfg["features"]["columns"]

FRAME_RATE: float = load_config("pipeline")["processing"]["frame_rate"]
LEAD_TIME_S: float = _tgt_cfg["preventive_lead_time_s"]
TTC_THRESHOLD_S: float = _tgt_cfg["standard_ttc_threshold_s"]

CV_FOLDS: int = _cv_cfg["n_splits"]
SMOTE_SEED: int = _cfg["smote"]["random_state"]

PARAM_GRID: dict = _xgb_cfg.get("param_grid", {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
})


def main() -> None:
    logger.info("XGBoost baseline training on master dataset")

    master_csv = Path(MASTER_CSV)
    if not master_csv.exists():
        logger.error("Master dataset not found: %s — run 02_build_dataset first.", master_csv)
        return

    df = pd.read_csv(master_csv)
    df = create_dual_targets(df, lead_time_s=LEAD_TIME_S, frame_rate=FRAME_RATE, ttc_threshold_s=TTC_THRESHOLD_S)

    available_features = [f for f in FEATURES if f in df.columns]
    if len(available_features) < len(FEATURES):
        missing = set(FEATURES) - set(available_features)
        logger.warning("Missing feature columns (will skip): %s", missing)

    X = df[available_features].fillna(0)
    Y = df["Y_preventive"]

    if Y.sum() == 0:
        logger.error("No positive risk labels — check dataset and thresholds.")
        return

    logger.info(
        "Dataset: %d rows, %d positive labels (%.1f%%)",
        len(Y),
        int(Y.sum()),
        100 * Y.mean(),
    )

    kf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=_cv_cfg["shuffle"],
        random_state=_cv_cfg["random_state"],
    )
    precision_scores, recall_scores, f1_scores = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, Y), start=1):
        logger.info("Fold %d/%d", fold, CV_FOLDS)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        X_train_res, Y_train_res = SMOTE(random_state=SMOTE_SEED).fit_resample(X_train, Y_train)

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
        )
        search = RandomizedSearchCV(
            xgb,
            param_distributions=PARAM_GRID,
            n_iter=_rf_cfg["search"]["n_iter"],
            cv=_rf_cfg["search"]["cv"],
            scoring=_rf_cfg["search"]["scoring"],
            random_state=_rf_cfg["search"]["random_state"],
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train_res, Y_train_res)

        preds = search.best_estimator_.predict(X_test)
        p = precision_score(Y_test, preds, zero_division=0)
        r = recall_score(Y_test, preds, zero_division=0)
        f = f1_score(Y_test, preds, zero_division=0)
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f)
        logger.info("  Fold %d → Precision=%.4f  Recall=%.4f  F1=%.4f", fold, p, r, f)

    logger.info(
        "CV summary — Precision: %.4f ± %.4f  Recall: %.4f ± %.4f  F1: %.4f ± %.4f",
        np.mean(precision_scores), np.std(precision_scores),
        np.mean(recall_scores), np.std(recall_scores),
        np.mean(f1_scores), np.std(f1_scores),
    )

    logger.info("Training final model on full dataset...")
    X_res, Y_res = SMOTE(random_state=SMOTE_SEED).fit_resample(X, Y)
    final_search = RandomizedSearchCV(
        XGBClassifier(objective="binary:logistic", eval_metric="logloss"),
        param_distributions=PARAM_GRID,
        n_iter=_rf_cfg["search"]["n_iter"],
        cv=_rf_cfg["search"]["cv"],
        scoring=_rf_cfg["search"]["scoring"],
        random_state=_rf_cfg["search"]["random_state"],
        n_jobs=-1,
        verbose=0,
    )
    final_search.fit(X_res, Y_res)

    output_path = Path(OUTPUT_MODEL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_search.best_estimator_, output_path)
    logger.info("Model saved to %s  (best params: %s)", output_path, final_search.best_params_)


if __name__ == "__main__":
    main()
