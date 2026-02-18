"""
02_build_dataset.py — Feature-engineer all CONCOR-D sessions into the master training CSV.

Walks every annotation CSV, parses trajectories, computes pairwise interaction
features for every (vehicle, worker) pair, and concatenates the results into a
single master dataset ready for model training.

Usage:
    python -m pipeline.02_build_dataset

Output:
    data/analysis_results/master_training_dataset_full.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from site_sentinel.config import load_config  # noqa: E402
from site_sentinel.data.parser import parse_trajectory_csv  # noqa: E402
from site_sentinel.features.engineering import (  # noqa: E402
    compute_interaction_features,
    compute_motion_features,
)
from site_sentinel.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_cfg = load_config("pipeline")
_proc = _cfg["processing"]
_mcfg = load_config("model_training")

FRAME_RATE: float = _proc["frame_rate"]
TIME_HORIZON_S: float = _proc["time_horizon_s"]
ROLLING_WINDOW_S: float = _proc["rolling_window_s"]

# Class name normalisation — the CONCOR-D dataset uses Title Case for object classes.
# We map all vehicle types to a common "vehicle" label and all vulnerable road users
# to "worker" so the shared feature engineering function can match pairs correctly.
_VEHICLE_CLASSES = {"Car", "Medium Vehicle", "Heavy Vehicle", "Bus", "Motorcycle"}
_VULNERABLE_CLASSES = {"Pedestrian", "Bicycle"}


def _normalise_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Map CONCOR-D class names to the two canonical labels: 'vehicle' and 'worker'."""
    df = df.copy()
    df["object_class"] = df["object_class"].apply(
        lambda c: "vehicle"
        if c in _VEHICLE_CLASSES
        else "worker"
        if c in _VULNERABLE_CLASSES
        else c
    )
    return df


def _process_session(csv_path: Path) -> pd.DataFrame | None:
    """
    Parse one session CSV and return a DataFrame of interaction features,
    or None if no valid pairs are found.
    """
    raw_df = parse_trajectory_csv(csv_path)
    if raw_df.empty:
        logger.warning("%s: empty after parsing — skipped", csv_path.name)
        return None

    raw_df = _normalise_classes(raw_df)
    motion_df = compute_motion_features(raw_df)

    interaction_df = compute_interaction_features(
        motion_df,
        frame_rate=FRAME_RATE,
        time_horizon_s=TIME_HORIZON_S,
        rolling_window_s=ROLLING_WINDOW_S,
        vehicle_class="vehicle",
        vulnerable_class="worker",
    )

    if interaction_df.empty:
        logger.debug("%s: no vehicle/worker pairs found", csv_path.name)
        return None

    interaction_df["source_file"] = csv_path.name
    return interaction_df


def main() -> None:
    raw_dir = Path(_cfg["data"]["raw_trajectory_dir"])
    output_csv = Path(_mcfg["random_forest"]["master_csv"])

    logger.info("Building master training dataset from %s", raw_dir)

    csv_files = sorted(raw_dir.rglob("*_ann.csv"))
    logger.info("Found %d annotation CSVs to process", len(csv_files))

    session_frames: list[pd.DataFrame] = []

    for i, csv_path in enumerate(csv_files, start=1):
        logger.info("[%d/%d] %s", i, len(csv_files), csv_path.name)
        try:
            result = _process_session(csv_path)
            if result is not None:
                session_frames.append(result)
        except Exception as e:
            logger.warning("Skipping %s: %s", csv_path.name, e)

    if not session_frames:
        logger.error("No interaction data extracted. Check dataset path and class names.")
        return

    master_df = pd.concat(session_frames, ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_csv, index=False)

    logger.info(
        "Master dataset complete: %d sessions, %d rows → %s",
        len(session_frames),
        len(master_df),
        output_csv,
    )


if __name__ == "__main__":
    main()
