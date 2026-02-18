"""
03c_train_tcn.py — Temporal Convolutional Network baseline on the master dataset.

Dilated TCN with kernel size 3 and dilation rates [1, 2, 4, 8], giving a
receptive field of about 90 frames. Like the LSTM, it was included to check
whether explicit sequence modelling could beat the Random Forest on this data.
It couldn't — the tabular rolling features already capture what the model needs.

Usage:
    python -m pipeline.03c_train_tcn

Input:
    data/analysis_results/master_training_dataset_full.csv
Output:
    models/tcn_master_predictor.keras
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from site_sentinel.config import load_config  # noqa: E402
from site_sentinel.features.targets import create_dual_targets  # noqa: E402
from site_sentinel.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_cfg = load_config("model_training")
_tcn_cfg = _cfg.get("tcn", {})
_cv_cfg = _cfg["cross_validation"]
_tgt_cfg = _cfg["targets"]

MASTER_CSV: str = _cfg["random_forest"]["master_csv"]
OUTPUT_MODEL: str = "models/tcn_master_predictor.keras"
FEATURES: list[str] = _cfg["features"]["columns"]

FRAME_RATE: float = load_config("pipeline")["processing"]["frame_rate"]
LEAD_TIME_S: float = _tgt_cfg["preventive_lead_time_s"]
TTC_THRESHOLD_S: float = _tgt_cfg["standard_ttc_threshold_s"]

CV_FOLDS: int = _cv_cfg["n_splits"]
SEQUENCE_LENGTH: int = _tcn_cfg.get("sequence_length", 25)
EPOCHS: int = _tcn_cfg.get("epochs", 15)
BATCH_SIZE: int = _tcn_cfg.get("batch_size", 32)
NB_FILTERS: int = _tcn_cfg.get("nb_filters", 64)
KERNEL_SIZE: int = _tcn_cfg.get("kernel_size", 3)
DILATIONS: list[int] = _tcn_cfg.get("dilations", [1, 2, 4, 8])
DROPOUT_RATE: float = _tcn_cfg.get("dropout_rate", 0.2)


def _build_tcn(n_features: int) -> Sequential:
    model = Sequential([
        TCN(
            input_shape=(SEQUENCE_LENGTH, n_features),
            nb_filters=NB_FILTERS,
            kernel_size=KERNEL_SIZE,
            dilations=DILATIONS,
            use_skip_connections=True,
            dropout_rate=DROPOUT_RATE,
        ),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def main() -> None:
    logger.info("TCN baseline training on master dataset")

    master_csv = Path(MASTER_CSV)
    if not master_csv.exists():
        logger.error("Master dataset not found: %s — run 02_build_dataset first.", master_csv)
        return

    df = pd.read_csv(master_csv)
    df = create_dual_targets(df, lead_time_s=LEAD_TIME_S, frame_rate=FRAME_RATE, ttc_threshold_s=TTC_THRESHOLD_S)

    available_features = [f for f in FEATURES if f in df.columns]
    X_data = df[available_features].fillna(100)
    Y_data = df["Y_preventive"]

    if Y_data.sum() == 0:
        logger.error("No positive risk labels — check dataset and thresholds.")
        return

    logger.info(
        "Dataset: %d rows, %d positive labels (%.1f%%)",
        len(Y_data), int(Y_data.sum()), 100 * Y_data.mean(),
    )

    kf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=_cv_cfg["shuffle"],
        random_state=_cv_cfg["random_state"],
    )
    precision_scores, recall_scores, f1_scores = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_data, Y_data), start=1):
        logger.info("Fold %d/%d", fold, CV_FOLDS)

        X_train_fold = X_data.iloc[train_idx]
        X_test_fold = X_data.iloc[test_idx]
        Y_train_fold = Y_data.iloc[train_idx]
        Y_test_fold = Y_data.iloc[test_idx]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        train_gen = TimeseriesGenerator(
            X_train_scaled, Y_train_fold.to_numpy(),
            length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE,
        )
        test_gen = TimeseriesGenerator(
            X_test_scaled, Y_test_fold.to_numpy(),
            length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE,
        )
        Y_test_actual = Y_test_fold.iloc[SEQUENCE_LENGTH:]

        if len(Y_test_actual) == 0:
            logger.warning("Fold %d: test set too small for sequence length — skipping", fold)
            continue

        # Class weight computed from this fold's training labels only
        positive_count = Y_train_fold.sum()
        class_weight = {
            0: 1.0,
            1: float(len(Y_train_fold) - positive_count) / float(positive_count)
            if positive_count > 0
            else 1.0,
        }

        model = _build_tcn(len(available_features))
        model.fit(train_gen, epochs=EPOCHS, verbose=0, class_weight=class_weight, shuffle=False)

        preds_proba = model.predict(test_gen, verbose=0)
        preds = (preds_proba > 0.5).astype(int)

        p = precision_score(Y_test_actual, preds, zero_division=0)
        r = recall_score(Y_test_actual, preds, zero_division=0)
        f = f1_score(Y_test_actual, preds, zero_division=0)
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f)
        logger.info("  Fold %d → Precision=%.4f  Recall=%.4f  F1=%.4f", fold, p, r, f)

    if precision_scores:
        logger.info(
            "CV summary — Precision: %.4f ± %.4f  Recall: %.4f ± %.4f  F1: %.4f ± %.4f",
            np.mean(precision_scores), np.std(precision_scores),
            np.mean(recall_scores), np.std(recall_scores),
            np.mean(f1_scores), np.std(f1_scores),
        )

    logger.info("Training final TCN on full dataset...")
    scaler_final = MinMaxScaler()
    X_scaled_full = scaler_final.fit_transform(X_data)
    full_gen = TimeseriesGenerator(
        X_scaled_full, Y_data.to_numpy(),
        length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE,
    )
    pos_count = Y_data.sum()
    final_class_weight = {
        0: 1.0,
        1: float(len(Y_data) - pos_count) / float(pos_count) if pos_count > 0 else 1.0,
    }
    final_model = _build_tcn(len(available_features))
    final_model.fit(full_gen, epochs=EPOCHS, verbose=0, class_weight=final_class_weight, shuffle=False)

    output_path = Path(OUTPUT_MODEL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(output_path)
    logger.info("TCN model saved to %s", output_path)


if __name__ == "__main__":
    main()
