"""
Coordinate transforms for projecting UTM world coordinates onto video pixel space.

The camera calibration script (pipeline/05_calibrate_camera.py) produces two
artefacts that this module consumes at inference time:

    homography_matrix.npy   — 3×3 perspective transform matrix (UTM → pixel),
                              computed by OpenCV RANSAC findHomography.
    transform_params.json   — centering and rotation parameters applied to the
                              raw UTM coordinates before the homography is applied.

The transform pipeline is:
    1. Subtract the UTM centroid (x_mean, y_mean)
    2. Invert the Y axis (UTM northing increases upward; image rows increase downward)
    3. Apply a rotation by angle theta to align the road with the image horizontal
    4. Apply the homography matrix to get (pixel_col, pixel_row)

Usage:

    from site_sentinel.vision.transform import load_transform, utm_to_pixel

    homography, params = load_transform(
        "data/analysis_results/homography_matrix.npy",
        "data/analysis_results/transform_params.json",
    )
    col, row = utm_to_pixel(411370.0, 5655230.0, homography, params)
    # Returns (-1, -1) if the point falls outside the frame
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_transform(
    homography_path: Path | str,
    params_path: Path | str,
) -> tuple[np.ndarray, dict]:
    """
    Load the saved homography matrix and coordinate transform parameters.

    Args:
        homography_path: Path to the .npy file containing the 3×3 homography matrix.
        params_path: Path to the JSON file containing x_mean, y_mean, and theta.

    Returns:
        (homography, params) where homography is a (3, 3) float64 ndarray
        and params is the parsed JSON dict.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    homography_path = Path(homography_path)
    params_path = Path(params_path)

    for p in (homography_path, params_path):
        if not p.exists():
            raise FileNotFoundError(f"Transform file not found: {p}")

    homography = np.load(homography_path)
    with params_path.open(encoding="utf-8") as f:
        params = json.load(f)

    logger.debug("Loaded homography (%s) and params (%s)", homography_path.name, params_path.name)
    return homography, params


def utm_to_pixel(
    x_utm: float,
    y_utm: float,
    homography: np.ndarray,
    params: dict,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[int, int]:
    """
    Project a single UTM coordinate pair to image pixel (column, row).

    The steps here mirror what pipeline/05_calibrate_camera.py did when
    computing the homography, so the same transform is applied consistently.

    Args:
        x_utm: Easting in metres (UTM).
        y_utm: Northing in metres (UTM).
        homography: 3×3 perspective transform matrix from load_transform().
        params: Transform parameters dict from load_transform().
        frame_width: Frame width in pixels. If provided with frame_height,
                     points outside the frame are returned as (-1, -1).
        frame_height: Frame height in pixels.

    Returns:
        (col, row) as integers, or (-1, -1) if the point is out of frame
        or the homography denominator is near zero.
    """
    x_mean: float = params["x_mean"]
    y_mean: float = params["y_mean"]
    theta: float = params["theta"]

    # Step 1: centre
    xc = x_utm - x_mean
    yc = -(y_utm - y_mean)  # invert Y: UTM northing grows up, images grow down

    # Step 2: rotate to align road direction with image axes
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = xc * cos_t - yc * sin_t
    yr = xc * sin_t + yc * cos_t

    # Step 3: apply homography in homogeneous coordinates
    h = homography
    denom = h[2, 0] * xr + h[2, 1] * yr + h[2, 2]
    if abs(denom) < 1e-10:
        return -1, -1

    col = int(round((h[0, 0] * xr + h[0, 1] * yr + h[0, 2]) / denom))
    row = int(round((h[1, 0] * xr + h[1, 1] * yr + h[1, 2]) / denom))

    # Step 4: bounds check
    if frame_width is not None and frame_height is not None:
        if col < 0 or col >= frame_width or row < 0 or row >= frame_height:
            return -1, -1

    return col, row
