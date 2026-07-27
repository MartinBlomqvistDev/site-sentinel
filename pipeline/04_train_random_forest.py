"""
04_train_random_forest.py — Dual-target Random Forest (the production model).

Trains two RF classifiers on the master dataset:
  - Y_preventive: will danger occur within the next 4 seconds?
  - Y_standard:   is TTC already dangerously low right now?

Both are saved together in a single dict (pkl file) so the renderer can load
them with one joblib.load call.

The Random Forest outperformed XGBoost, LSTM, and TCN on F1-score and was
chosen as the production model for two reasons: performance and interpretability.
Feature importances are meaningful in a safety-critical context.

Usage:
    python -m pipeline.04_train_random_forest

Input:
    data/analysis_results/master_training_dataset_full.csv
Output:
    models/rf_master_predictor_dual_lead_tuned.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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
_search_cfg = _rf_cfg["search"]

# `master_csv` lives under the shared `data:` block, not under `random_forest:`.
# Reading it from the wrong section made this module raise KeyError at import,
# so the script could not run at all against its own config.
MASTER_CSV: str = _cfg["data"]["master_csv"]
OUTPUT_MODEL: str = _rf_cfg["output_model"]
FEATURES: list[str] = _cfg["features"]["columns"]

FRAME_RATE: float = load_config("pipeline")["processing"]["frame_rate"]
LEAD_TIME_S: float = _tgt_cfg["preventive_lead_time_s"]
TTC_THRESHOLD_S: float = _tgt_cfg["standard_ttc_threshold_s"]
RISK_DISTANCE_M: float = _tgt_cfg["preventive_risk_distance_m"]

CV_FOLDS: int = _cv_cfg["n_splits"]
SMOTE_SEED: int = _cfg["smote"]["random_state"]
PARAM_GRID: dict = _rf_cfg["param_grid"]


#: Per-target cross-validation scores, written so the reported numbers can be
#: traced to the target that produced them. Without this the run logged both
#: targets and persisted only the models, so `configs/app.yaml` ended up carrying
#: one unlabelled precision/recall/F1 triple for a dual-target model. That
#: matters here: Y_standard is much the easier target, because TTC is itself an
#: input feature, so "is TTC below 2s right now" is close to reading a column.
METRICS_CSV = "data/analysis_results/rf_cv_metrics.csv"


def _train_rf(
    X: pd.DataFrame,
    y: pd.Series,
    kf: StratifiedKFold,
    label_name: str,
) -> tuple[RandomForestClassifier, dict, dict]:
    """
    Run cross-validated training for one target variable and return the final model.

    The final model is trained on the full dataset using the best hyperparameters
    found during CV — it's not the average of the fold models.

    Returns:
        The fitted model, the best hyperparameters, and the cross-validated
        scores. The scores are returned rather than only logged so `main` can
        persist them per target.
    """
    logger.info("Training %s model", label_name)
    precision_scores, recall_scores, f1_scores = [], [], []
    best_params: dict = {}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_res, y_train_res = SMOTE(random_state=SMOTE_SEED).fit_resample(X_train, y_train)

        rf = RandomForestClassifier(random_state=42, class_weight="balanced")
        search = RandomizedSearchCV(
            rf,
            param_distributions=PARAM_GRID,
            n_iter=_search_cfg["n_iter"],
            cv=_search_cfg["cv"],
            scoring=_search_cfg["scoring"],
            random_state=_search_cfg["random_state"],
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train_res, y_train_res)
        best_params = search.best_params_

        preds = search.best_estimator_.predict(X_test)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f)
        logger.info("  [%s] Fold %d/%d → P=%.4f  R=%.4f  F1=%.4f", label_name, fold, CV_FOLDS, p, r, f)

    logger.info(
        "[%s] CV — P: %.4f±%.4f  R: %.4f±%.4f  F1: %.4f±%.4f",
        label_name,
        np.mean(precision_scores), np.std(precision_scores),
        np.mean(recall_scores), np.std(recall_scores),
        np.mean(f1_scores), np.std(f1_scores),
    )

    metrics = {
        "target": label_name,
        "cv_folds": CV_FOLDS,
        "positives": int(y.sum()),
        "positive_rate": round(float(y.mean()), 6),
        "precision_mean": round(float(np.mean(precision_scores)), 4),
        "precision_std": round(float(np.std(precision_scores)), 4),
        "recall_mean": round(float(np.mean(recall_scores)), 4),
        "recall_std": round(float(np.std(recall_scores)), 4),
        "f1_mean": round(float(np.mean(f1_scores)), 4),
        "f1_std": round(float(np.std(f1_scores)), 4),
    }

    X_res, y_res = SMOTE(random_state=SMOTE_SEED).fit_resample(X, y)
    final_rf = RandomForestClassifier(random_state=42, class_weight="balanced", **best_params)
    final_rf.fit(X_res, y_res)
    logger.info("[%s] Final model trained. Best params: %s", label_name, best_params)

    return final_rf, best_params, metrics


def main() -> None:
    logger.info("Dual-target Random Forest training")

    master_csv = Path(MASTER_CSV)
    if not master_csv.exists():
        logger.error("Master dataset not found: %s — run 02_build_dataset first.", master_csv)
        return

    df = pd.read_csv(master_csv)
    df = create_dual_targets(
        df,
        lead_time_s=LEAD_TIME_S,
        frame_rate=FRAME_RATE,
        ttc_threshold_s=TTC_THRESHOLD_S,
        risk_distance_m=RISK_DISTANCE_M,
    )

    available_features = [f for f in FEATURES if f in df.columns]
    if len(available_features) < len(FEATURES):
        logger.warning("Missing features: %s", set(FEATURES) - set(available_features))

    X = df[available_features].fillna(0)
    y_pre = df["Y_preventive"]
    y_std = df["Y_standard"]

    if y_pre.sum() == 0 or y_std.sum() == 0:
        logger.error("No positive labels in one or both targets — check thresholds.")
        return

    logger.info(
        "Data loaded: %d rows | preventive positives=%d (%.1f%%) | standard positives=%d (%.1f%%)",
        len(df),
        int(y_pre.sum()), 100 * y_pre.mean(),
        int(y_std.sum()), 100 * y_std.mean(),
    )

    kf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=_cv_cfg["shuffle"],
        random_state=_cv_cfg["random_state"],
    )

    rf_pre, pre_params, pre_metrics = _train_rf(X, y_pre, kf, "Preventive (4s)")
    rf_std, std_params, std_metrics = _train_rf(X, y_std, kf, "Standard (TTC)")

    output_path = Path(OUTPUT_MODEL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preventive": rf_pre, "standard": rf_std}, output_path)

    logger.info("Dual-target RF models saved to %s", output_path)

    metrics_path = Path(METRICS_CSV)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([pre_metrics, std_metrics]).to_csv(metrics_path, index=False)
    logger.info("Per-target CV scores written to %s", metrics_path)
    logger.info(
        "Headline figures belong to the target named in that file. "
        "Update configs/app.yaml from it rather than by hand."
    )


if __name__ == "__main__":
    main()
