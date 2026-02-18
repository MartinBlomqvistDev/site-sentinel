"""
05_calibrate_camera.py — Compute the homography matrix for UTM → pixel projection.

Reads the pixel-annotated trajectory CSV (a special DFS Viewer export that includes
both UTM coordinates and corresponding image pixel coordinates for each timestep),
then uses RANSAC-based findHomography to fit a perspective transform.

The result is saved as:
  homography_matrix.npy  — 3×3 perspective transform matrix
  transform_params.json  — centering and rotation parameters applied before the
                           homography (x_mean, y_mean, y_centered_max, theta)

These two files are loaded by pipeline/06_render_video.py at rendering time.

Usage:
    python -m pipeline.05_calibrate_camera

Input:
    data/full_trajectories_PIXELS.csv  (configured in pipeline.yaml)
Output:
    data/analysis_results/homography_matrix.npy
    data/analysis_results/transform_params.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from site_sentinel.config import load_config  # noqa: E402
from site_sentinel.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_cfg = load_config("pipeline")
_vid = _cfg["video"]

# ---------------------------------------------------------------------------
# Paths — from config
# ---------------------------------------------------------------------------
INPUT_CSV_PATH: str = _vid["trajectory_csv_path"]
OUTPUT_DIR: str = _cfg["data"]["analysis_results_dir"]
OUTPUT_H_PATH: str = str(Path(OUTPUT_DIR) / "homography_matrix.npy")
OUTPUT_PARAMS_PATH: str = str(Path(OUTPUT_DIR) / "transform_params.json")

# ---------------------------------------------------------------------------
# CSV format constants
# The pixel-annotated CSV has 9 values per timestep instead of the standard 7:
#   x [m], y [m], speed, tangential_acc, lateral_acc, time, heading, pixel_x, pixel_y
# ---------------------------------------------------------------------------
NUM_FIXED_COLS: int = 12   # metadata columns before the trajectory blob
NUM_TRAJ_COLS_PER_STEP: int = 9  # values per timestep in this special export
IDX_UTM_X: int = 0    # x [m]   — UTM easting
IDX_UTM_Y: int = 1    # y [m]   — UTM northing
IDX_TIME: int = 5     # time [s]
IDX_PIXEL_X: int = 7  # image x [px]
IDX_PIXEL_Y: int = 8  # image y [px]

# Number of header/metadata lines at the top of the pixel-annotated CSV
# before the actual trajectory data begins.
CSV_SKIP_LINES: int = 80


def _parse_pixel_csv(csv_path: Path) -> pd.DataFrame:
    """
    Parse the pixel-annotated DFS Viewer CSV.

    This is a different format from the standard 7-col annotation CSVs:
    each timestep blob has 9 fields, the last two being the pixel coordinates
    that the DFS Viewer computed from its own calibration.

    Returns a DataFrame with columns:
        track_id, object_type, time, utm_x, utm_y, pixel_x, pixel_y
    """
    all_points: list[dict] = []

    with csv_path.open(encoding="utf-8", errors="replace") as f:
        # Skip DFS Viewer file metadata
        for _ in range(CSV_SKIP_LINES):
            f.readline()
        f.readline()  # header line

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            try:
                track_id = int(parts[0])
                obj_type = parts[1].strip()
            except (ValueError, IndexError):
                logger.warning("Skipping unparseable row: %s...", line[:50])
                continue

            traj_data = parts[NUM_FIXED_COLS:]
            for i in range(0, len(traj_data), NUM_TRAJ_COLS_PER_STEP):
                chunk = traj_data[i : i + NUM_TRAJ_COLS_PER_STEP]
                if len(chunk) < NUM_TRAJ_COLS_PER_STEP:
                    continue
                try:
                    all_points.append({
                        "track_id": track_id,
                        "object_type": obj_type,
                        "time": float(chunk[IDX_TIME]),
                        "utm_x": float(chunk[IDX_UTM_X]),
                        "utm_y": float(chunk[IDX_UTM_Y]),
                        "pixel_x": float(chunk[IDX_PIXEL_X]),
                        "pixel_y": float(chunk[IDX_PIXEL_Y]),
                    })
                except (ValueError, IndexError):
                    continue

    if not all_points:
        raise ValueError(f"No valid trajectory data found in {csv_path}")

    df = pd.DataFrame(all_points)
    logger.info(
        "Parsed %d data points from %d tracks", len(df), df["track_id"].nunique()
    )
    return df


def _compute_rotation_angle(df: pd.DataFrame) -> tuple[float, dict]:
    """
    Estimate the rotation angle that aligns the road direction with the image axes.

    Centres the UTM coordinates, inverts the Y axis (UTM grows up, images grow down),
    then fits a line through the cloud of points to find the dominant direction.

    Returns (theta, transform_params) where theta is in radians.
    """
    x_mean = df["utm_x"].mean()
    y_mean = df["utm_y"].mean()

    x_centered = df["utm_x"] - x_mean
    y_centered = df["utm_y"] - y_mean
    y_centered_max = y_centered.max()
    y_centered_inv = y_centered_max - y_centered

    # Fit a line to find the dominant axis angle
    X_reg = x_centered.values.reshape(-1, 1)
    y_reg = y_centered_inv.values
    valid_mask = ~np.isnan(X_reg).flatten() & ~np.isnan(y_reg)

    if valid_mask.sum() < 2:
        raise ValueError("Not enough data points for rotation estimation.")

    model = LinearRegression().fit(X_reg[valid_mask], y_reg[valid_mask])
    slope = model.coef_[0]
    theta = np.arctan(slope)

    logger.info("Computed rotation angle: %.2f degrees", -np.rad2deg(theta))

    params = {
        "x_mean": float(x_mean),
        "y_mean": float(y_mean),
        "y_centered_max": float(y_centered_max),
        "theta": float(theta),
    }
    return theta, params


def _apply_transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply the centering + Y-inversion + rotation to produce the source points for homography."""
    df = df.copy()
    x_mean = params["x_mean"]
    y_mean = params["y_mean"]
    y_centered_max = params["y_centered_max"]
    theta = params["theta"]

    x_c = df["utm_x"] - x_mean
    y_c = df["utm_y"] - y_mean
    y_c_inv = y_centered_max - y_c

    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    df["x_transformed"] = x_c * cos_t - y_c_inv * sin_t
    df["y_transformed"] = x_c * sin_t + y_c_inv * cos_t
    return df


def main() -> None:
    logger.info("Camera calibration — automatic RANSAC homography")

    csv_path = Path(INPUT_CSV_PATH)
    if not csv_path.exists():
        logger.error("Input CSV not found: %s", csv_path)
        sys.exit(1)

    # --- Parse ---
    df = _parse_pixel_csv(csv_path)

    # --- Rotation estimation ---
    theta, params = _compute_rotation_angle(df)

    # --- Transform source points ---
    df_tf = _apply_transform(df, params)

    # --- Fit homography ---
    src_pts = df_tf[["x_transformed", "y_transformed"]].to_numpy().astype(np.float32)
    dst_pts = df_tf[["pixel_x", "pixel_y"]].to_numpy().astype(np.float32)

    valid_mask = ~np.isnan(src_pts).any(axis=1) & ~np.isnan(dst_pts).any(axis=1)
    src_pts = src_pts[valid_mask]
    dst_pts = dst_pts[valid_mask]

    if len(src_pts) < 4:
        logger.error("Too few valid point pairs (%d) for homography.", len(src_pts))
        sys.exit(1)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        logger.error("Homography computation failed.")
        sys.exit(1)

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    logger.info(
        "Homography computed using %d/%d inlier points", n_inliers, len(src_pts)
    )

    # --- Save ---
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_H_PATH, H)
    logger.info("Homography matrix saved to %s", OUTPUT_H_PATH)

    with open(OUTPUT_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
    logger.info("Transform parameters saved to %s", OUTPUT_PARAMS_PATH)

    logger.info("Calibration complete. Run 06_render_video.py next.")


if __name__ == "__main__":
    main()
