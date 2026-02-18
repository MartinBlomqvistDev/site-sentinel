# scripts/3c_feature_eng_for_demo_restore.py
"""
Full feature restoration for demo (restores raw + derived features previously used).
Parses flattened trajectory CSV (one row per track with many trajectory samples).
Outputs:
 - data/analysis_results/features_<date>_all_objects.csv  (per-sample, all objects)
 - data/analysis_results/ultimate_features_<date>_vuln_vs_car.csv (interaction features)
"""

import os
import math
import time
from datetime import datetime
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
RAW_TRAJECTORY_CSV = r"C:\DS24\Site_Sentinel\data\raw\20190918_1500_Sid_StP_3W_d_1_3_ann_for_demo.csv"
OUTPUT_DIR = r"C:\DS24\Site_Sentinel\data\analysis_results"
FRAME_RATE = 29.97
TIME_HORIZON = 1.5  # seconds for preventive prediction
TARGET_ACTORS = {
    "vulnerable_id": 1,  # adjust to your demo
    "car_id": 3
}

# Rolling windows (in seconds)
ROLL_WINDOWS_S = [1.0, 2.0, 3.0]

# Parser params (depends on your CSV export)
FIXED_META_COLS = 12   # number of metadata columns before the flattened trajectory blob (adjust if necessary)
TRAJ_GROUP_SIZE = 9    # x,y,speed_kmh,tan_acc,lat_acc,time,angle_rad,img_x_px,img_y_px

# --- Helpers ---


def parse_flattened_trajectory_csv(path, fixed_cols=FIXED_META_COLS, group_size=TRAJ_GROUP_SIZE):
    """
    Read trajectory CSV which is one row per track, with flattened trajectory samples in a big blob.
    Returns a tidy DataFrame with one row per trajectory sample.
    """
    rows = []
    header = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # Read header line (keep it for reference)
        first = f.readline()
        if not first:
            raise ValueError("Empty file")
        header = [h.strip() for h in first.strip().split(";")]
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            # split into parts - keep empty fields removed
            parts = [p.strip() for p in line.split(";")]
            # if file included trailing semicolons, filter only empty at end:
            # we need at least fixed_cols + 1
            if len(parts) <= fixed_cols:
                continue
            meta = parts[:fixed_cols]
            traj = parts[fixed_cols:]
            # meta columns: we cannot guarantee exact names, attempt to coerce sensible pieces
            # index 0: Track ID, index 1: Type/class, index 2: Score [%] maybe...
            try:
                track_id = int(float(meta[0]))
            except Exception:
                # skip if track id is not parseable
                continue
            obj_type = meta[1] if len(meta) > 1 else "Unknown"
            # iterate trajectory groups
            for i in range(0, len(traj), group_size):
                chunk = traj[i:i + group_size]
                if len(chunk) < group_size:
                    continue
                # Try to coerce floats; if fails, skip this sample
                try:
                    x = float(chunk[0])
                    y = float(chunk[1])
                    speed_kmh = float(chunk[2])
                    tan_acc = float(chunk[3])
                    lat_acc = float(chunk[4])
                    t = float(chunk[5])
                    angle = float(chunk[6])
                    img_x = float(chunk[7])
                    img_y = float(chunk[8])
                except Exception:
                    continue
                rows.append({
                    "trackId": track_id,
                    "class": obj_type,
                    "raw_score_perc": meta[2] if len(meta) > 2 else None,
                    "x": x,
                    "y": y,
                    "speed_kmh": speed_kmh,
                    "tan_acc_ms2": tan_acc,
                    "lat_acc_ms2": lat_acc,
                    "time": t,
                    "angle_rad": angle,
                    "img_x_px": img_x,
                    "img_y_px": img_y,
                })
    df = pd.DataFrame(rows)
    # Ensure types
    if not df.empty:
        df = df.sort_values(["trackId", "time"]).reset_index(drop=True)
    return df


def compute_per_track_motion(df):
    """
    Given tidy df with columns trackId, time, x, y, speed_kmh (optional),
    compute velocity components, speeds in m/s, accelerations, heading, cumulative distance, etc.
    """
    if df.empty:
        return df

    df = df.copy()
    # convert speed to m/s if available
    df["speed_ms_from_kmh"] = df["speed_kmh"] / 3.6

    # group and compute diffs
    df["delta_t"] = df.groupby("trackId")["time"].diff().fillna(0.0)
    df["delta_x"] = df.groupby("trackId")["x"].diff().fillna(0.0)
    df["delta_y"] = df.groupby("trackId")["y"].diff().fillna(0.0)

    # velocity components (m/s) from position differences => robust fallback if speed_kmh missing
    df["velocity_x"] = (df["delta_x"] / df["delta_t"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["velocity_y"] = (df["delta_y"] / df["delta_t"]).replace([np.inf, -np.inf], 0).fillna(0)

    # speed magnitude in m/s: prefer speed_kmh if present else compute from delta
    df["speed_ms"] = df["speed_ms_from_kmh"].where(df["speed_kmh"].notnull() & (df["speed_kmh"] != 0),
                                                   np.sqrt(df["velocity_x"] ** 2 + df["velocity_y"] ** 2))
    df["speed_kmh_recalc"] = df["speed_ms"] * 3.6

    # delta speed and acceleration
    df["delta_speed"] = df.groupby("trackId")["speed_ms"].diff().fillna(0)
    df["accel_ms2_from_delta"] = (df["delta_speed"] / df["delta_t"]).replace([np.inf, -np.inf], 0).fillna(0)

    # Use provided tangential acceleration if meaningful, else fallback
    # keep both: tan_acc_ms2_provided (from CSV) and tan_acc_ms2_calc
    df["tan_acc_ms2_provided"] = df["tan_acc_ms2"]
    df["tan_acc_ms2_calc"] = df["accel_ms2_from_delta"]

    # lateral acceleration: prefer CSV field if nonzero
    df["lat_acc_ms2_provided"] = df["lat_acc_ms2"]

    # heading from velocity vector (rad)
    df["heading_rad"] = np.arctan2(df["velocity_y"], df["velocity_x"]).fillna(0.0)

    # cumulative/traveled distance per track
    df["segment_dist"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)
    df["cum_dist"] = df.groupby("trackId")["segment_dist"].cumsum().fillna(0)

    # track-level aggregations (first/last)
    # compute for convenience: track_width/length placeholders if you want later
    return df


def compute_interaction_features(all_df, vuln_id, car_id, time_horizon=TIME_HORIZON, frame_rate=FRAME_RATE):
    """
    Compute advanced pairwise features between vulnerable actor and car actor.
    Returns merged dataframe aligned by time (inner join on nearest times).
    """
    # Filter tracks
    v = all_df[all_df["trackId"] == vuln_id].copy()
    c = all_df[all_df["trackId"] == car_id].copy()
    if v.empty or c.empty:
        print("⚠️ One of the target tracks is empty. Returning empty interaction DF.")
        return pd.DataFrame()

    # We want matched times. Both df have 'time' values; perform outer merge and forward/backward fill
    merged = pd.merge_asof(
        v.sort_values("time"),
        c.sort_values("time"),
        on="time",
        suffixes=("_vuln", "_car"),
        direction="nearest",
        tolerance=1.0 / frame_rate  # allow matching within one frame
    )

    # Drop rows without a valid match
    merged = merged.dropna(subset=["x_car", "y_car", "x_vuln", "y_vuln"])

    # Relative vector from vuln to car: car - vuln
    merged["dx"] = merged["x_car"] - merged["x_vuln"]
    merged["dy"] = merged["y_car"] - merged["y_vuln"]

    # Euclidean distance
    merged["rel_distance"] = np.sqrt(merged["dx"] ** 2 + merged["dy"] ** 2)

    # Relative velocity vector (car minus vuln)
    merged["dvx"] = merged["velocity_x_car"] - merged["velocity_x_vuln"]
    merged["dvy"] = merged["velocity_y_car"] - merged["velocity_y_vuln"]

    # Relative speed (signed along line-of-sight) and magnitude
    # Approach speed = projection of relative velocity onto relative displacement (negative means approaching)
    dot = merged["dx"] * merged["dvx"] + merged["dy"] * merged["dvy"]
    merged["approach_speed_signed"] = - dot / merged["rel_distance"].replace(0, np.nan)
    merged["rel_speed_mag"] = np.sqrt(merged["dvx"] ** 2 + merged["dvy"] ** 2)

    # Another useful scalar: relative speed along line-of-sight magnitude
    merged["rel_speed_along_los"] = merged["approach_speed_signed"].fillna(0.0)

    # TTC: time to collision along relative motion (robust)
    dv2 = merged["dvx"] ** 2 + merged["dvy"] ** 2
    # Numerator is - dot(dx, dv) where negative dot means approaching
    ttc = - dot / dv2.replace(0, np.nan)
    # Only keep positive TTC (future collision), else set to a large number
    merged["ttc"] = np.where((ttc > 0) & (ttc < 1e6), ttc, np.nan)
    merged["ttc"] = merged["ttc"].fillna(100.0)

    # Preventive features: future relative distance if both keep current velocity for time_horizon
    merged["future_rel_distance"] = merged["rel_distance"] - (merged["rel_speed_along_los"] * time_horizon)
    merged["future_rel_distance"] = merged["future_rel_distance"].clip(lower=0.1)

    merged["preventive_risk"] = 1.0 / merged["future_rel_distance"]

    # Convert speeds to m/s fields for model inputs
    merged["speed_ms_vuln"] = merged["speed_ms_vuln"].fillna(merged["speed_ms_from_kmh_vuln"])
    merged["speed_ms_car"] = merged["speed_ms_car"].fillna(merged["speed_ms_from_kmh_car"])

    # Provide accelerations (choose provided or calculated)
    merged["accel_ms2_vuln"] = merged["tan_acc_ms2_provided_vuln"].fillna(merged["tan_acc_ms2_calc_vuln"])
    merged["accel_ms2_car"] = merged["tan_acc_ms2_provided_car"].fillna(merged["tan_acc_ms2_calc_car"])

    # Add some ratios & indicators
    merged["rel_distance_over_speed"] = merged["rel_distance"] / (merged["rel_speed_mag"].replace(0, np.nan))
    merged["is_approaching"] = merged["rel_speed_along_los"] > 0

    # Rolling averages for key signals: rel_distance, rel_speed_along_los, future_rel_distance
    # First we need a time index and ensure sorted by time
    merged = merged.sort_values("time").reset_index(drop=True)
    for window_s in ROLL_WINDOWS_S:
        w = max(1, int(round(window_s * FRAME_RATE)))
        prefix = f"r{int(window_s)}s"
        merged[f"{prefix}_rel_distance_mean"] = merged["rel_distance"].rolling(window=w, min_periods=1).mean()
        merged[f"{prefix}_rel_speed_mean"] = merged["rel_speed_along_los"].rolling(window=w, min_periods=1).mean()
        merged[f"{prefix}_future_rel_distance_mean"] = merged["future_rel_distance"].rolling(window=w, min_periods=1).mean()

    # Provide final selection of columns commonly used by model / renderer
    # Keep everything else for debugging
    return merged


def write_output_csvs(all_df, interaction_df, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_path = os.path.join(out_dir, f"features_{ts}_all_objects.csv")
    interact_path = os.path.join(out_dir, f"ultimate_features_{ts}_vuln_vs_car.csv")

    all_df.to_csv(all_path, index=False)
    interaction_df.to_csv(interact_path, index=False)
    print(f"✅ Wrote all-objects features to: {all_path}")
    print(f"✅ Wrote interaction ultimate features to: {interact_path}")


def main():
    print("=== RESTORE: Full Feature Engineering for Demo ===")
    print(f"Reading raw trajectories from: {RAW_TRAJECTORY_CSV}")
    all_raw = parse_flattened_trajectory_csv(RAW_TRAJECTORY_CSV)
    if all_raw.empty:
        print("No trajectory data parsed. Exiting.")
        return

    print(f"Parsed {len(all_raw)} samples across {all_raw['trackId'].nunique()} tracks.")

    # Compute per-track motion features
    all_feats = compute_per_track_motion(all_raw)

    # Rename some columns for clarity & compatibility with other parts of the pipeline
    # Keep both original and convenient names
    all_feats = all_feats.rename(columns={
        "time": "time",
        "speed_kmh": "speed_kmh",
        "speed_ms": "speed_ms",
        "heading_rad": "heading_rad"
    })

    # For compatibility: ensure speed_ms_from_kmh columns exist per suffix style used elsewhere
    # create suffixed columns for merged usage
    # We'll create duplicates when merging later (pandas merge_asof will append suffixes automatically)
    # So leave names as-is.

    # Compute interaction features for the chosen target pair
    vuln = TARGET_ACTORS["vulnerable_id"]
    car = TARGET_ACTORS["car_id"]

    print(f"Computing interaction features for vulnerable_id={vuln} and car_id={car} ...")
    interaction = compute_interaction_features(all_feats, vuln, car)

    # Optional: If you want full pairwise interactions across many pairs, implement here (not done to keep output size controlled)

    # Save outputs
    write_output_csvs(all_feats, interaction, out_dir=OUTPUT_DIR)
    print("✅ Full feature restoration complete.")


if __name__ == "__main__":
    main()
