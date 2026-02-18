"""
Feature engineering for the Site Sentinel risk prediction pipeline.

All functions here are pure (no file I/O, no side effects, deterministic output
for a given input). This makes them straightforward to unit-test and to reason
about when debugging the pipeline.

The feature computation follows two stages:

  1. compute_motion_features(df)
     Per-object kinematics: velocity components, scalar speed, and acceleration.
     Input: the raw long-format DataFrame from the parser.
     Output: same DataFrame with four new columns added.

  2. compute_interaction_features(df, cfg)
     Pairwise danger metrics between every (vehicle, worker) pair in each session.
     Input: the motion-augmented DataFrame.
     Output: a new long-format DataFrame, one row per (vehicle_id, worker_id, timestep).

The 11 columns produced by these two functions are exactly the features the
trained Random Forest model expects.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fallback TTC value when vehicles are diverging (no collision on current trajectory).
# Using 100 rather than inf avoids NaN propagation through downstream computations.
_TTC_DIVERGING_FILL = 100.0


def compute_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-object kinematic columns to the trajectory DataFrame.

    The raw parser output has position (x, y) and a pre-computed speed column,
    but we re-derive velocity components and acceleration from first principles
    so all kinematic features are internally consistent.

    New columns added:
        velocity_x   (m/s) — eastward velocity component
        velocity_y   (m/s) — northward velocity component
        speed_ms     (m/s) — scalar speed (sqrt of velocity components)
        accel_ms2    (m/s²) — rate of change of scalar speed

    Args:
        df: Long-format trajectory DataFrame with columns:
            track_id, time (s), x (m UTM), y (m UTM).

    Returns:
        The same DataFrame with four additional columns.
        Rows are sorted by (track_id, time). The first timestep of each
        track has NaN for velocity and acceleration (no previous position to
        diff against); these are filled with 0.
    """
    df = df.sort_values(["track_id", "time"]).copy()

    grouped = df.groupby("track_id", sort=False)
    dt = grouped["time"].diff()
    dx = grouped["x"].diff()
    dy = grouped["y"].diff()

    # Avoid division by zero when consecutive timestamps are identical
    safe_dt = dt.replace(0, np.nan)

    df["velocity_x"] = (dx / safe_dt).fillna(0.0)
    df["velocity_y"] = (dy / safe_dt).fillna(0.0)
    df["speed_ms"] = np.sqrt(df["velocity_x"] ** 2 + df["velocity_y"] ** 2)

    d_speed = grouped["speed_ms"].diff()
    df["accel_ms2"] = (d_speed / safe_dt).fillna(0.0)

    return df


def compute_interaction_features(
    df: pd.DataFrame,
    frame_rate: float,
    time_horizon_s: float,
    rolling_window_s: float,
    vehicle_class: str = "car",
    vulnerable_class: str = "pedestrian",
) -> pd.DataFrame:
    """
    Compute pairwise danger metrics for every (vehicle, worker) pair in the scene.

    For each session in the DataFrame, we find all vehicles and all workers,
    then compute interaction features for every pair at every shared timestep.
    This produces the training rows used by the Random Forest.

    Feature columns produced:
        rel_distance       (m)   — Euclidean distance between the pair
        rel_speed          (m/s) — magnitude of the relative velocity vector
        speed_ms_vuln      (m/s) — worker scalar speed (renamed from speed_ms)
        speed_ms_car       (m/s) — vehicle scalar speed
        accel_ms2_vuln     (m/s²)
        accel_ms2_car      (m/s²)
        ttc                (s)   — time-to-collision (100 if diverging)
        approach_speed     (m/s) — rate of gap closure; positive = converging
        future_rel_dist    (m)   — projected distance after time_horizon_s
        rel_dist_avg_2s    (m)   — rolling mean of rel_distance
        rel_speed_avg_2s   (m/s) — rolling mean of rel_speed
        future_rel_dist_avg_2s (m) — rolling mean of future_rel_dist

    Args:
        df: Motion-augmented trajectory DataFrame (output of compute_motion_features).
            Must have columns: track_id, object_class, time, x, y,
            velocity_x, velocity_y, speed_ms, accel_ms2.
        frame_rate: Recording frame rate in Hz (used to convert rolling_window_s to frames).
        time_horizon_s: How far ahead to project future positions for the preventive feature.
        rolling_window_s: Length of the rolling average window in seconds.
        vehicle_class: The object_class label for vehicles in this dataset.
        vulnerable_class: The object_class label for workers/pedestrians.

    Returns:
        Long-format DataFrame with one row per (vehicle_id, worker_id, timestep),
        indexed by (track_id_vuln, track_id_car, time).
        Returns an empty DataFrame if no valid pairs are found.
    """
    rolling_frames = max(1, int(frame_rate * rolling_window_s))

    vehicles = df[df["object_class"] == vehicle_class]
    workers = df[df["object_class"] == vulnerable_class]

    if vehicles.empty or workers.empty:
        logger.warning(
            "No vehicle/worker pairs found (vehicles=%d, workers=%d). "
            "Check object_class values: found %s",
            len(vehicles),
            len(workers),
            df["object_class"].unique().tolist(),
        )
        return pd.DataFrame()

    pair_frames: list[pd.DataFrame] = []

    for veh_id in vehicles["track_id"].unique():
        veh = vehicles[vehicles["track_id"] == veh_id].set_index("time")

        for worker_id in workers["track_id"].unique():
            wrk = workers[workers["track_id"] == worker_id].set_index("time")

            # Align on shared timestamps (inner join)
            merged = pd.merge(
                veh[["x", "y", "velocity_x", "velocity_y", "speed_ms", "accel_ms2"]],
                wrk[["x", "y", "velocity_x", "velocity_y", "speed_ms", "accel_ms2"]],
                left_index=True,
                right_index=True,
                suffixes=("_car", "_vuln"),
            )

            if merged.empty:
                continue

            merged = merged.sort_index()
            pair_df = _compute_pair_metrics(merged, time_horizon_s, rolling_frames)
            pair_df["track_id_car"] = veh_id
            pair_df["track_id_vuln"] = worker_id
            pair_frames.append(pair_df)

    if not pair_frames:
        return pd.DataFrame()

    result = pd.concat(pair_frames, ignore_index=False)
    result.index.name = "time"
    return result.reset_index()


def _compute_pair_metrics(
    merged: pd.DataFrame,
    time_horizon_s: float,
    rolling_frames: int,
) -> pd.DataFrame:
    """
    Compute all pairwise metrics for a single (vehicle, worker) pair.

    Args:
        merged: DataFrame aligned on time with *_car and *_vuln column suffixes.
        time_horizon_s: Lookahead for the future distance feature.
        rolling_frames: Number of frames in the rolling average window.

    Returns:
        DataFrame with all interaction feature columns, indexed by time.
    """
    out = pd.DataFrame(index=merged.index)

    # --- Relative distance ---
    dx = merged["x_car"] - merged["x_vuln"]
    dy = merged["y_car"] - merged["y_vuln"]
    out["rel_distance"] = np.sqrt(dx**2 + dy**2)

    # --- Relative velocity and scalar relative speed ---
    dvx = merged["velocity_x_car"] - merged["velocity_x_vuln"]
    dvy = merged["velocity_y_car"] - merged["velocity_y_vuln"]
    out["rel_speed"] = np.sqrt(dvx**2 + dvy**2)

    # --- Per-object kinematics ---
    out["speed_ms_vuln"] = merged["speed_ms_vuln"]
    out["speed_ms_car"] = merged["speed_ms_car"]
    out["accel_ms2_vuln"] = merged["accel_ms2_vuln"]
    out["accel_ms2_car"] = merged["accel_ms2_car"]

    # --- Approach speed ---
    # Dot product of the separation vector and the relative velocity.
    # Positive means the gap is closing (the vehicle is approaching the worker).
    dot_dx_dv = dx * dvx + dy * dvy
    safe_dist = out["rel_distance"].replace(0, np.nan)
    out["approach_speed"] = -(dot_dx_dv / safe_dist).fillna(0.0)

    # --- Time-to-collision ---
    # Derived from the quadratic closest-approach formula assuming constant velocities.
    # Negative or infinite TTC (diverging tracks) is replaced with the sentinel value.
    dot_dv_dv = dvx**2 + dvy**2
    safe_dv = dot_dv_dv.replace(0, np.nan)
    raw_ttc = -(dot_dx_dv / safe_dv)
    out["ttc"] = raw_ttc.where(raw_ttc > 0, np.nan).fillna(_TTC_DIVERGING_FILL)

    # --- Future relative distance (preventive feature) ---
    # Linear extrapolation: where will each object be after time_horizon_s seconds?
    future_x_car = merged["x_car"] + merged["velocity_x_car"] * time_horizon_s
    future_y_car = merged["y_car"] + merged["velocity_y_car"] * time_horizon_s
    future_x_vuln = merged["x_vuln"] + merged["velocity_x_vuln"] * time_horizon_s
    future_y_vuln = merged["y_vuln"] + merged["velocity_y_vuln"] * time_horizon_s

    future_dx = future_x_car - future_x_vuln
    future_dy = future_y_car - future_y_vuln
    # Clip to a small positive value to avoid division by zero in the risk score
    out["future_rel_dist"] = np.sqrt(future_dx**2 + future_dy**2).clip(lower=0.1)

    # --- Rolling averages ---
    min_p = 1  # include partial windows at the start of each session
    out["rel_dist_avg_2s"] = (
        out["rel_distance"].rolling(rolling_frames, min_periods=min_p).mean()
    )
    out["rel_speed_avg_2s"] = (
        out["rel_speed"].rolling(rolling_frames, min_periods=min_p).mean()
    )
    out["future_rel_dist_avg_2s"] = (
        out["future_rel_dist"].rolling(rolling_frames, min_periods=min_p).mean()
    )

    return out
