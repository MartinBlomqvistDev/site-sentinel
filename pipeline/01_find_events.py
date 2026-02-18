"""
01_find_events.py — Rank the highest-risk near-miss events in the CONCOR-D dataset.

Scans all annotation CSVs, scores each session for interaction risk, and writes
the top N events to a CSV for review. Running this first gives you a ranked list
of the best candidate sessions for manual inspection and demo selection.

Usage:
    python -m pipeline.01_find_events

Output:
    data/analysis_results/top_events.csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable so site_sentinel.* works from any directory.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from site_sentinel.config import load_config  # noqa: E402
from site_sentinel.data.parser import parse_trajectory_csv  # noqa: E402
from site_sentinel.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_cfg = load_config("pipeline")
_ev = _cfg["events"]
_proc = _cfg["processing"]

# ---------------------------------------------------------------------------
# Constants pulled from config
# ---------------------------------------------------------------------------
FRAME_RATE: float = _proc["frame_rate"]
NUM_TOP_EVENTS: int = _ev["top_n"]
INTERACTION_DISTANCE_THRESHOLD: float = 3.0   # metres — pair must be this close
MAX_SPEED: float = 50.0                        # m/s — velocity clamp before scoring
MIN_TRACK_DURATION_S: float = _ev["min_session_time_s"]
MAX_REALISTIC_SPEED: float = _ev["approach_speed_cap"]
MIN_VALID_FRAME: int = 100                     # skip first N frames (tracker warm-up)
RISK_WEIGHT_DIRECTIONAL: float = _ev["risk_weight_directional"]
RISK_WEIGHT_PROXIMITY: float = _ev["risk_weight_proximity"]


def _classify_event(row: pd.Series) -> str:
    """Categorise the interaction type for human review."""
    try:
        delta_x = row["x_car"] - row["x_vuln"]
        delta_y = row["y_car"] - row["y_vuln"]
        dot = delta_x * row["velocity_x_car"] + delta_y * row["velocity_y_car"]
        car_speed = math.sqrt(row["velocity_x_car"] ** 2 + row["velocity_y_car"] ** 2)
        safe_dist = max(row["rel_distance"], 0.5)
        angle = math.degrees(math.acos(np.clip(dot / (safe_dist * car_speed + 1e-6), -1, 1)))
    except Exception:
        angle = 180.0

    if row["approach_speed"] > 1.0 and angle < 25:
        return "collision_course"
    if row["rel_distance"] < 2.0 and row["speed_car"] > 3.0:
        return "close_pass"
    if row["speed_car"] > 1.0 and row["rel_distance"] < 3.0:
        return "static_exposure"
    if 25 <= angle < 70:
        return "crossing_path"
    return "same_direction"


def _find_best_event(df: pd.DataFrame) -> pd.Series | None:
    """
    Return the highest-risk (vehicle, worker) interaction frame in this session,
    or None if no valid interaction is found.
    """
    if df.empty or "track_id" not in df.columns:
        return None

    df = df.sort_values(["track_id", "time"]).reset_index(drop=True)

    # Drop very short tracks — they're usually tracker artefacts
    durations = df.groupby("track_id")["time"].agg(lambda s: s.max() - s.min())
    valid_ids = durations[durations > MIN_TRACK_DURATION_S].index
    df = df[df["track_id"].isin(valid_ids)]

    vulnerable = df[df["object_class"].str.lower().isin(["pedestrian", "bicycle"])].copy()
    vehicles = df[df["object_class"].str.lower() == "car"].copy()

    if vulnerable.empty or vehicles.empty:
        return None

    # Assign integer frame numbers and drop initial tracker warm-up frames
    df["frame"] = (df["time"] * FRAME_RATE).round().astype(int)
    df = df[df["frame"] >= MIN_VALID_FRAME]
    if df.empty:
        return None

    # Compute velocities per-group for both subsets
    for subset in (vulnerable, vehicles):
        dt = subset.groupby("track_id")["time"].diff().replace(0, np.nan)
        subset["velocity_x"] = (
            (subset["x"] - subset.groupby("track_id")["x"].shift(1)) / dt
        ).fillna(0).clip(-MAX_SPEED, MAX_SPEED)
        subset["velocity_y"] = (
            (subset["y"] - subset.groupby("track_id")["y"].shift(1)) / dt
        ).fillna(0).clip(-MAX_SPEED, MAX_SPEED)
        subset["speed"] = np.sqrt(subset["velocity_x"] ** 2 + subset["velocity_y"] ** 2)
        subset["frame"] = (subset["time"] * FRAME_RATE).round().astype(int)

    merged = pd.merge(vulnerable, vehicles, on="frame", suffixes=("_vuln", "_car"))
    if merged.empty:
        return None

    merged = merged[(merged["frame"] > MIN_VALID_FRAME) & (merged["time_vuln"] >= 0)]

    delta_x = merged["x_car"] - merged["x_vuln"]
    delta_y = merged["y_car"] - merged["y_vuln"]
    delta_vx = merged["velocity_x_car"] - merged["velocity_x_vuln"]
    delta_vy = merged["velocity_y_car"] - merged["velocity_y_vuln"]

    merged["rel_distance"] = np.sqrt(delta_x ** 2 + delta_y ** 2)
    safe_dist = merged["rel_distance"].clip(lower=0.5)
    merged["approach_speed"] = (-(delta_x * delta_vx + delta_y * delta_vy) / safe_dist).clip(
        lower=0, upper=MAX_SPEED
    )
    merged["ttc"] = np.where(
        merged["approach_speed"] > 0,
        merged["rel_distance"] / merged["approach_speed"],
        np.nan,
    )

    candidates = merged[
        (merged["speed_car"] > 2.0)
        & (merged["speed_vuln"] >= 0.0)
        & (merged["rel_distance"] <= INTERACTION_DISTANCE_THRESHOLD)
        & (merged["rel_distance"] > 0.1)
    ].copy()

    if candidates.empty:
        return None

    capped_approach = candidates["approach_speed"].clip(0, MAX_REALISTIC_SPEED)
    capped_speed = candidates["speed_car"].clip(0, MAX_REALISTIC_SPEED)
    risk_directional = (1 / candidates["rel_distance"]) * capped_approach
    risk_proximity = (1 / (candidates["rel_distance"] ** 2)) * capped_speed
    candidates["risk_score"] = (
        RISK_WEIGHT_DIRECTIONAL * risk_directional
        + RISK_WEIGHT_PROXIMITY * risk_proximity
    )
    candidates["event_type"] = candidates.apply(_classify_event, axis=1)
    candidates["near_miss"] = candidates["rel_distance"] < 1.5

    return candidates.loc[candidates["risk_score"].idxmax()].copy()


def main() -> None:
    raw_dir = Path(_cfg["data"]["raw_trajectory_dir"])
    output_csv = Path(_cfg["data"]["analysis_results_dir"]) / "top_events.csv"

    logger.info("Scanning trajectory files in %s", raw_dir)

    csv_files = sorted(raw_dir.rglob("*_ann.csv"))
    logger.info("Found %d annotation CSVs", len(csv_files))

    all_events: list[dict] = []

    for i, csv_path in enumerate(csv_files, start=1):
        logger.info("[%d/%d] %s", i, len(csv_files), csv_path.name)
        try:
            raw_df = parse_trajectory_csv(csv_path)
            if raw_df.empty:
                continue
            best = _find_best_event(raw_df)
            if best is not None:
                event = best.to_dict()
                event["file"] = csv_path.name
                event["full_path"] = str(csv_path)
                all_events.append(event)
        except Exception as e:
            logger.warning("Skipping %s: %s", csv_path.name, e)

    if not all_events:
        logger.error("No near-miss events found across all sessions.")
        return

    results_df = pd.DataFrame(all_events)
    top_df = results_df.sort_values("risk_score", ascending=False).head(NUM_TOP_EVENTS)

    display_cols = [
        "risk_score", "rel_distance", "approach_speed", "ttc",
        "event_type", "near_miss", "file", "frame", "time_vuln",
        "track_id_vuln", "track_id_car", "full_path",
    ]
    display_cols = [c for c in display_cols if c in top_df.columns]
    top_df = top_df[display_cols].reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(output_csv, index=False)
    logger.info("Top %d events saved to %s", NUM_TOP_EVENTS, output_csv)

    # Print a brief summary of the top 5 for quick orientation
    for _, row in top_df.head(5).iterrows():
        near_miss_flag = " [NEAR-MISS]" if row.get("near_miss") else ""
        logger.info(
            "  %s  →  t=%.2fs  frame=%d  risk=%.3f%s",
            row.get("file", ""),
            row.get("time_vuln", 0),
            int(row.get("frame", 0)),
            row.get("risk_score", 0),
            near_miss_flag,
        )


if __name__ == "__main__":
    main()
