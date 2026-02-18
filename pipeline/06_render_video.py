"""
06_render_video.py — Produce the annotated near-miss demo video.

Reads the trajectory CSV (with both UTM and pixel coordinates), runs the trained
Random Forest model frame by frame, and overlays risk state annotations onto the
source video. The alert state is stabilised with a 15-frame history window and
a hysteresis state machine (SAFE → APPROACHING → IMMINENT) to prevent flickering.

This script went through 23 iterations before the timing, visual clarity, and
alert threshold tuning felt right. The key insight was that the smoothed risk
signal needs separate enter/exit thresholds (hysteresis) to prevent the alert
state from chattering around transition points.

Usage:
    python -m pipeline.06_render_video

Input:
    data/full_trajectories_PIXELS.csv
    models/rf_master_predictor_dual_lead_tuned.pkl
    data/analysis_results/homography_matrix.npy
    data/analysis_results/transform_params.json
    data/raw/20190918_1500_Sid_StP_3W_d_1_3_cal.mp4
Output:
    data/analysis_results/final_demo_v16.mp4  (configured in pipeline.yaml)
"""

from __future__ import annotations

import collections
import json
import sys
from itertools import combinations
from math import hypot, sqrt
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from site_sentinel.config import load_config  # noqa: E402
from site_sentinel.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_cfg = load_config("pipeline")
_vid = _cfg["video"]
_rend = _vid["renderer"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_TRAJECTORY_CSV: str = _vid["trajectory_csv_path"]
MODEL_PATH: str = load_config("model_training")["random_forest"]["output_model"]
VIDEO_PATH: str = _vid["source_path"]
OUTPUT_VIDEO_PATH: str = _vid["output_path"]
HOMOGRAPHY_PATH: str = _vid["homography_path"]
PARAMS_PATH: str = _vid["transform_params_path"]

# ---------------------------------------------------------------------------
# Prediction & physics
# ---------------------------------------------------------------------------
TIME_HORIZON: float = _cfg["processing"]["time_horizon_s"]
FRAME_RATE: float = _cfg["processing"]["frame_rate"]
INTERACTION_DISTANCE_THRESHOLD: float = 35.0  # metres — max range for pair tracking
SAFETY_BUBBLE_RADIUS_PIXELS: float = _rend["safety_bubble_radius_px"]

# ---------------------------------------------------------------------------
# Target worker ID — the specific person tracked in the demo session
# ---------------------------------------------------------------------------
TARGET_PERSON_ID: int = 114

# ---------------------------------------------------------------------------
# Windowing — which section of the video to render
# ---------------------------------------------------------------------------
MAIN_EVENT_START_TIME: float = 4 * 60 + 7  # 4m07s — the near-miss peak
WINDOW_BEFORE: float = 10.0  # seconds before event
WINDOW_AFTER: float = 10.0   # seconds after event

# ---------------------------------------------------------------------------
# Object classes
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = {"Car", "Medium Vehicle", "Heavy Vehicle", "Bus", "Motorcycle"}
VULNERABLE_CLASSES = {"Pedestrian", "Bicycle"}

# ---------------------------------------------------------------------------
# Risk weights
# ---------------------------------------------------------------------------
AI_MODEL_WEIGHT: float = _rend["ai_model_weight"]
PROXIMITY_WEIGHT: float = _rend["proximity_weight"]
PREVENTIVE_WEIGHT: float = _rend["preventive_weight"]

# ---------------------------------------------------------------------------
# Temporal smoothing & hysteresis thresholds
# ---------------------------------------------------------------------------
RISK_HISTORY_WINDOW: int = _rend["risk_history_window"]
IMMINENT_PERSISTENCE_FRAMES: int = _rend["imminent_frame_count"]
APPROACHING_ENTER_THRESHOLD: float = _rend["approaching_enter_threshold"]
APPROACHING_EXIT_THRESHOLD: float = _rend["approaching_exit_threshold"]
IMMINENT_ENTER_THRESHOLD: float = _rend["imminent_enter_threshold"]
IMMINENT_EXIT_THRESHOLD: float = _rend["imminent_exit_threshold"]
IMMINENT_MIN_APPROACH_SPEED: float = 0.5  # m/s — must be actively closing in

# ---------------------------------------------------------------------------
# Drawing sizes & colours (BGR)
# ---------------------------------------------------------------------------
DEFAULT_BOX_SIZE = 15
DEFAULT_LINE_THICK = 1
VULNERABLE_BOX_SIZE = 20
WORKER_LINE_THICK = 2
WORKER_BOX_SIZE = 25
ALERT_BOX_SIZE = 25
ALERT_LINE_THICK = 2
LINE_THRESHOLD = 0.5  # pixels — minimum movement before drawing prediction line

COLOR_SAFE = (0, 255, 0)
COLOR_APPROACHING = (0, 165, 255)
COLOR_IMMINENT = (0, 0, 255)
COLOR_DEFAULT = (255, 180, 0)

# ---------------------------------------------------------------------------
# Pixel-annotated CSV parsing constants
# The trajectory CSV for rendering has 12 metadata cols then 9 values per timestep.
# ---------------------------------------------------------------------------
_NUM_FIXED_COLS = 12
_NUM_TRAJ_COLS = 9
_IDX_UTM_X = 0
_IDX_UTM_Y = 1
_IDX_TIME = 5
_IDX_PIXEL_X = 7
_IDX_PIXEL_Y = 8
_CSV_SKIP_LINES = 80


def _parse_pixel_csv(filepath: Path) -> pd.DataFrame:
    """Parse the DFS Viewer pixel-annotated trajectory CSV."""
    all_points: list[dict] = []
    skipped = 0

    with filepath.open(encoding="utf-8", errors="replace") as f:
        for _ in range(_CSV_SKIP_LINES):
            f.readline()
        f.readline()  # header

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            try:
                track_id = int(parts[0])
                obj_type = parts[1].strip()
            except (ValueError, IndexError):
                skipped += 1
                continue

            traj_data = parts[_NUM_FIXED_COLS:]
            for i in range(0, len(traj_data), _NUM_TRAJ_COLS):
                chunk = traj_data[i : i + _NUM_TRAJ_COLS]
                if len(chunk) < _NUM_TRAJ_COLS:
                    continue
                try:
                    all_points.append({
                        "trackId": track_id,
                        "class": obj_type,
                        "time": float(chunk[_IDX_TIME]),
                        "x": float(chunk[_IDX_UTM_X]),
                        "y": float(chunk[_IDX_UTM_Y]),
                        "pixel_x": float(chunk[_IDX_PIXEL_X]),
                        "pixel_y": float(chunk[_IDX_PIXEL_Y]),
                    })
                except (ValueError, IndexError):
                    continue

    if skipped:
        logger.warning("Skipped %d unparseable rows", skipped)

    if not all_points:
        logger.error("No data points parsed from %s", filepath)
        return pd.DataFrame()

    df = pd.DataFrame(all_points)
    for col in ["trackId", "time", "x", "y", "pixel_x", "pixel_y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().astype({"trackId": int})


def _compute_motion(df: pd.DataFrame) -> pd.DataFrame:
    """Add velocity, speed, and acceleration columns to the trajectory DataFrame."""
    if df.empty:
        return df
    df = df.sort_values(["trackId", "time"]).reset_index(drop=True)
    safe_dt = df.groupby("trackId")["time"].diff().replace(0, np.nan)
    df["velocity_x"] = (
        (df.groupby("trackId")["x"].diff() / safe_dt)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    df["velocity_y"] = (
        (df.groupby("trackId")["y"].diff() / safe_dt)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    df["speed_ms"] = np.sqrt(df["velocity_x"] ** 2 + df["velocity_y"] ** 2)
    df["accel_ms2"] = (
        (df.groupby("trackId")["speed_ms"].diff() / safe_dt)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    cols = ["time", "trackId", "class", "x", "y", "velocity_x", "velocity_y", "speed_ms", "accel_ms2"]
    if "pixel_x" in df.columns:
        cols += ["pixel_x", "pixel_y"]
    return df[cols]


def _apply_transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply centering + Y-inversion + rotation to align UTM with the image coordinate frame."""
    df = df.copy()
    x_c = df["x"] - params["x_mean"]
    y_c = df["y"] - params["y_mean"]
    y_c_inv = params["y_centered_max"] - y_c
    cos_t = np.cos(-params["theta"])
    sin_t = np.sin(-params["theta"])
    df["x_transformed"] = x_c * cos_t - y_c_inv * sin_t
    df["y_transformed"] = x_c * sin_t + y_c_inv * cos_t
    return df


def _compute_pair_features(frame_objects: list[dict], fps: float) -> list[dict]:
    """Compute interaction features for all (vehicle, worker) pairs at one frame."""
    result = []
    if len(frame_objects) < 2:
        return result

    req = ["x", "y", "velocity_x", "velocity_y", "speed_ms", "accel_ms2"]

    for obj1, obj2 in combinations(frame_objects, 2):
        if not all(
            k in obj1 and k in obj2
            and isinstance(obj1.get(k), (int, float))
            and np.isfinite(obj1.get(k, np.nan))
            and isinstance(obj2.get(k), (int, float))
            and np.isfinite(obj2.get(k, np.nan))
            for k in req
        ):
            continue

        vehicle, vulnerable = None, None
        if obj1["class"] in VEHICLE_CLASSES and obj2["class"] in VULNERABLE_CLASSES:
            vehicle, vulnerable = obj1, obj2
        elif obj2["class"] in VEHICLE_CLASSES and obj1["class"] in VULNERABLE_CLASSES:
            vehicle, vulnerable = obj2, obj1
        else:
            continue

        dist = sqrt(
            (vehicle["x"] - vulnerable["x"]) ** 2
            + (vehicle["y"] - vulnerable["y"]) ** 2
        )
        if dist >= INTERACTION_DISTANCE_THRESHOLD:
            continue

        pf: dict = {
            "vehicle_id": vehicle["trackId"],
            "vulnerable_id": vulnerable["trackId"],
            "time": vehicle["time"],
        }
        for k in ["x", "y", "velocity_x", "velocity_y", "speed_ms", "accel_ms2"]:
            pf[f"{k}_car"] = vehicle.get(k, 0.0)
            pf[f"{k}_vuln"] = vulnerable.get(k, 0.0)

        pf["rel_distance"] = dist
        pf["rel_speed"] = sqrt(
            (pf["velocity_x_car"] - pf["velocity_x_vuln"]) ** 2
            + (pf["velocity_y_car"] - pf["velocity_y_vuln"]) ** 2
        )

        dx = pf["x_car"] - pf["x_vuln"]
        dy = pf["y_car"] - pf["y_vuln"]
        dvx = pf["velocity_x_car"] - pf["velocity_x_vuln"]
        dvy = pf["velocity_y_car"] - pf["velocity_y_vuln"]

        dot_dx_dv = dx * dvx + dy * dvy
        pf["approach_speed"] = -dot_dx_dv / dist if dist > 0.1 else 0.0

        dot_dv_dv = dvx ** 2 + dvy ** 2
        raw_ttc = -dot_dx_dv / dot_dv_dv if dot_dv_dv > 1e-6 else np.inf
        pf["ttc"] = raw_ttc if 0 < raw_ttc < 1000 else 100.0

        future_dist = dist - pf["approach_speed"] * TIME_HORIZON if pf["approach_speed"] > 0 else dist
        pf["future_rel_distance"] = max(0.1, future_dist)
        pf["preventive_risk"] = 1.0 / pf["future_rel_distance"]

        # Single-frame approximation for rolling averages
        pf["rel_dist_avg_2s"] = dist
        pf["rel_speed_avg_2s"] = pf["rel_speed"]
        pf["future_rel_dist_avg_2s"] = pf["future_rel_distance"]

        result.append(pf)

    return result


def _rounded_rectangle(img, pt1, pt2, color, thickness: int = 2, radius: int = 10) -> None:
    """Draw a rectangle with rounded corners using cv2 primitives."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    if x1 >= x2 or y1 >= y2:
        return
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def main() -> None:
    logger.info("Rendering annotated demo video (v16)")
    logger.info(
        "Risk weights: AI=%.2f  Proximity=%.2f  Preventive=%.2f | "
        "Smoothing window: %d frames | "
        "Thresholds: approaching=%.2f  imminent=%.2f",
        AI_MODEL_WEIGHT, PROXIMITY_WEIGHT, PREVENTIVE_WEIGHT,
        RISK_HISTORY_WINDOW,
        APPROACHING_ENTER_THRESHOLD, IMMINENT_ENTER_THRESHOLD,
    )

    # --- Load trajectory data ---
    csv_path = Path(RAW_TRAJECTORY_CSV)
    all_objects_df = _parse_pixel_csv(csv_path)
    if all_objects_df.empty:
        logger.error("No data from %s — aborting.", csv_path)
        return
    logger.info(
        "Parsed %d points from %d tracks", len(all_objects_df), all_objects_df["trackId"].nunique()
    )

    motion_df = _compute_motion(all_objects_df)

    # --- Open video ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = FRAME_RATE
        logger.warning("Video FPS unavailable — using configured value %.2f", fps)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %dx%d @ %.2f FPS, %d total frames", frame_width, frame_height, fps, total_frames)

    # --- Load model ---
    try:
        model_dict = joblib.load(MODEL_PATH)
        model = model_dict.get(
            "preventive",
            model_dict.get("prevention", next(iter(model_dict.values()))),
        )
        expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        logger.info("Model loaded. Expected features: %s", expected_features)
    except Exception as e:
        logger.error("Cannot load model from %s: %s", MODEL_PATH, e)
        cap.release()
        return

    # --- Load homography & transform params ---
    try:
        H = np.load(HOMOGRAPHY_PATH)
        logger.info("Homography matrix loaded.")
    except Exception as e:
        logger.error("Cannot load homography: %s", e)
        cap.release()
        return

    try:
        with open(PARAMS_PATH, encoding="utf-8") as f:
            params = json.load(f)
        logger.info("Transform parameters loaded.")
    except Exception as e:
        logger.error("Cannot load transform params: %s", e)
        cap.release()
        return

    # --- Apply UTM transform ---
    all_objects_tf = _apply_transform(motion_df, params)
    all_objects_tf["frame"] = (all_objects_tf["time"] * fps).round().astype(int)
    all_objects_tf = all_objects_tf.dropna(subset=["frame"])
    objects_by_frame = {
        int(f): g.to_dict("records")
        for f, g in all_objects_tf.groupby("frame")
    }
    logger.info("Transform applied. Ready to render.")

    # --- Frame window ---
    start_frame = max(0, int((MAIN_EVENT_START_TIME - WINDOW_BEFORE) * fps))
    end_frame = min(total_frames, int((MAIN_EVENT_START_TIME + WINDOW_AFTER) * fps))
    n_frames = max(0, end_frame - start_frame + 1)
    logger.info(
        "Rendering frames %d–%d (%d frames / %.1fs)",
        start_frame, end_frame, n_frames, n_frames / fps,
    )

    # --- Video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        logger.error("Cannot open VideoWriter for %s", OUTPUT_VIDEO_PATH)
        cap.release()
        return

    base_features = [
        "rel_distance", "rel_speed", "speed_ms_car", "speed_ms_vuln",
        "accel_ms2_car", "accel_ms2_vuln", "ttc", "approach_speed",
        "rel_dist_avg_2s", "rel_speed_avg_2s", "future_rel_dist_avg_2s",
    ]
    features_to_use = expected_features if expected_features is not None else base_features

    # --- State tracking ---
    # Use deque for O(1) append and automatic eviction of old risk values
    risk_history: collections.deque = collections.deque(maxlen=RISK_HISTORY_WINDOW)
    current_state = "SAFE"

    frame_id = -1
    processed = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        actual_frame = start_frame + frame_id

        if actual_frame > end_frame:
            break

        current_objects = objects_by_frame.get(actual_frame, [])
        future_pixel_positions: dict[int, tuple[float, float]] = {}

        # --- Pass 1: project future positions for all objects ---
        for obj in current_objects:
            try:
                if "x_transformed" not in obj or "y_transformed" not in obj:
                    continue
                x_tf = float(obj["x_transformed"])
                y_tf = float(obj["y_transformed"])
                if np.isnan(x_tf) or np.isnan(y_tf):
                    continue

                pt = np.array([[[x_tf, y_tf]]], dtype=np.float32)
                px_py = cv2.perspectiveTransform(pt, H)
                px, py = float(px_py[0][0][0]), float(px_py[0][0][1])

                vx = float(obj.get("velocity_x", 0.0))
                vy = float(obj.get("velocity_y", 0.0))
                fx_utm = float(obj["x"]) + vx * TIME_HORIZON
                fy_utm = float(obj["y"]) + vy * TIME_HORIZON

                fx_c = fx_utm - params["x_mean"]
                fy_c = fy_utm - params["y_mean"]
                fy_c_inv = params["y_centered_max"] - fy_c
                cos_t = np.cos(-params["theta"])
                sin_t = np.sin(-params["theta"])
                fx_tf = fx_c * cos_t - fy_c_inv * sin_t
                fy_tf = fx_c * sin_t + fy_c_inv * cos_t

                fpx, fpy = px, py
                if not (np.isnan(fx_tf) or np.isnan(fy_tf)):
                    fpt = np.array([[[fx_tf, fy_tf]]], dtype=np.float32)
                    fpx_py = cv2.perspectiveTransform(fpt, H)
                    if fpx_py is not None:
                        fpx = float(np.clip(fpx_py[0][0][0], 0, frame_width - 1))
                        fpy = float(np.clip(fpx_py[0][0][1], 0, frame_height - 1))

                future_pixel_positions[int(obj["trackId"])] = (fpx, fpy)

            except Exception as e:
                logger.debug("Future position error obj %s frame %d: %s", obj.get("trackId"), actual_frame, e)

        # --- Pass 2: compute pair risks ---
        pair_features = _compute_pair_features(current_objects, fps)
        vehicle_risk_to_worker: dict[int, float] = {}
        max_risk = 0.0
        intruding_vehicles: set[int] = set()
        worker_future_pos = future_pixel_positions.get(TARGET_PERSON_ID)

        for pf in pair_features:
            risk_prob = proximity_risk = preventive_risk_val = 0.0

            if features_to_use:
                try:
                    feat_dict = {f: pf.get(f, 0) for f in features_to_use}
                    feat_df = pd.DataFrame([feat_dict])[features_to_use]
                    if expected_features:
                        feat_df = feat_df.reindex(columns=expected_features, fill_value=0)
                    if not feat_df.isnull().values.any():
                        risk_prob = float(model.predict_proba(feat_df)[0][1])

                    rel_dist = pf.get("rel_distance", 0)
                    if rel_dist and np.isfinite(rel_dist) and rel_dist > 0.1:
                        proximity_risk = (1 / (rel_dist + 0.1)) * np.exp(-rel_dist / 10.0)

                    pr = pf.get("preventive_risk", 0)
                    if pr and np.isfinite(pr):
                        preventive_risk_val = float(pr)

                except Exception as e:
                    logger.debug("Prediction error frame %d: %s", actual_frame, e)

            boosted = min(1.0, max(0.0,
                risk_prob * AI_MODEL_WEIGHT
                + proximity_risk * PROXIMITY_WEIGHT
                + preventive_risk_val * PREVENTIVE_WEIGHT,
            ))

            veh_id = pf["vehicle_id"]
            vuln_id = pf["vulnerable_id"]

            if vuln_id == TARGET_PERSON_ID:
                approach = pf.get("approach_speed", 0)
                effective_risk = (
                    boosted if approach > IMMINENT_MIN_APPROACH_SPEED
                    else min(boosted, APPROACHING_ENTER_THRESHOLD)
                )
                vehicle_risk_to_worker[veh_id] = max(
                    vehicle_risk_to_worker.get(veh_id, 0.0), effective_risk
                )
                max_risk = max(max_risk, vehicle_risk_to_worker[veh_id])

                veh_future = future_pixel_positions.get(veh_id)
                if worker_future_pos and veh_future:
                    if hypot(
                        worker_future_pos[0] - veh_future[0],
                        worker_future_pos[1] - veh_future[1],
                    ) < SAFETY_BUBBLE_RADIUS_PIXELS:
                        intruding_vehicles.add(veh_id)

        # --- Temporal smoothing ---
        risk_history.append(max_risk)
        smoothed_risk = float(np.mean(risk_history)) if risk_history else 0.0
        high_risk_count = sum(1 for r in risk_history if r > 0.8)

        # --- Hysteresis state machine ---
        if current_state == "IMMINENT":
            if smoothed_risk < IMMINENT_EXIT_THRESHOLD:
                current_state = "APPROACHING"
        elif current_state == "APPROACHING":
            if high_risk_count >= IMMINENT_PERSISTENCE_FRAMES and smoothed_risk > IMMINENT_ENTER_THRESHOLD:
                current_state = "IMMINENT"
            elif smoothed_risk < APPROACHING_EXIT_THRESHOLD:
                current_state = "SAFE"
        else:
            if smoothed_risk > APPROACHING_ENTER_THRESHOLD:
                current_state = "APPROACHING"

        # --- Overlay: risk text ---
        if current_state == "IMMINENT":
            indicator_color = COLOR_IMMINENT
            risk_text = "WORKER: IMMINENT"
        elif current_state == "APPROACHING":
            indicator_color = COLOR_APPROACHING
            risk_text = "WORKER: APPROACHING"
        else:
            indicator_color = COLOR_SAFE
            risk_text = f"WORKER RISK: {smoothed_risk:.0%}"

        tx, ty = frame_width - 300, 40
        (tw, th), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
        cv2.putText(frame, risk_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, indicator_color, 2)

        # --- Draw objects ---
        for row in current_objects:
            try:
                obj_id = int(row["trackId"])
                obj_class = row.get("class", "Unknown")
                is_main_worker = obj_id == TARGET_PERSON_ID
                is_intruder = obj_id in intruding_vehicles

                box_color = COLOR_DEFAULT
                box_size = DEFAULT_BOX_SIZE
                line_thick = DEFAULT_LINE_THICK
                show_warning = False

                if is_main_worker:
                    if current_state == "IMMINENT":
                        box_color = COLOR_IMMINENT
                        show_warning = True
                    elif current_state == "APPROACHING":
                        box_color = COLOR_APPROACHING
                        show_warning = True
                    else:
                        box_color = COLOR_SAFE
                    box_size = ALERT_BOX_SIZE
                    line_thick = ALERT_LINE_THICK
                elif is_intruder:
                    risk_level = vehicle_risk_to_worker.get(obj_id, 0.0)
                    box_color = COLOR_IMMINENT if risk_level > 0.8 else COLOR_APPROACHING
                    box_size = ALERT_BOX_SIZE
                    line_thick = ALERT_LINE_THICK
                elif obj_class in VULNERABLE_CLASSES:
                    box_color = COLOR_SAFE
                    box_size = VULNERABLE_BOX_SIZE

                if "x_transformed" not in row or "y_transformed" not in row:
                    continue
                x_tf = float(row["x_transformed"])
                y_tf = float(row["y_transformed"])
                if np.isnan(x_tf) or np.isnan(y_tf):
                    continue

                pt = np.array([[[x_tf, y_tf]]], dtype=np.float32)
                px_py = cv2.perspectiveTransform(pt, H)
                if px_py is None:
                    continue
                px, py = float(px_py[0][0][0]), float(px_py[0][0][1])

                if not (0 <= px < frame_width and 0 <= py < frame_height):
                    continue

                fpx, fpy = future_pixel_positions.get(obj_id, (px, py))
                _rounded_rectangle(
                    frame,
                    (px - box_size, py - box_size),
                    (px + box_size, py + box_size),
                    box_color, line_thick,
                )

                if is_main_worker or is_intruder:
                    cv2.putText(
                        frame, f"ID:{obj_id}",
                        (int(px), int(py - box_size - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2,
                    )

                if show_warning and is_main_worker:
                    cv2.putText(
                        frame, "WARNING!",
                        (int(px - box_size * 2), int(py + box_size + 15)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_IMMINENT, 2,
                    )

                if (is_main_worker or is_intruder) and (
                    abs(px - fpx) > LINE_THRESHOLD or abs(py - fpy) > LINE_THRESHOLD
                ):
                    cv2.line(frame, (int(px), int(py)), (int(fpx), int(fpy)), box_color, line_thick)

            except Exception as e:
                logger.debug("Draw error obj %s frame %d: %s", row.get("trackId"), actual_frame, e)

        out.write(frame)
        processed += 1

        if processed % 100 == 0 or processed == 1:
            remaining = max(0, n_frames - processed)
            logger.info("Rendered %d/%d frames (%.0fs remaining)", processed, n_frames, remaining / fps)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info("Done. %d frames written to %s", processed, OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
