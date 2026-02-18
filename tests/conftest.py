"""
Shared pytest fixtures for the Site Sentinel test suite.

All fixtures are synthetic — no real dataset files required. Values are chosen
so that expected feature outputs are trivial to compute by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Tiny trajectory DataFrame (parser output shape)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_trajectory_df() -> pd.DataFrame:
    """
    Three objects, five timesteps each.

    Object layout:
      - track_id "1", object_class "Car"   — moving at constant 10 m/s eastward
      - track_id "2", object_class "Pedestrian" — stationary at (50, 100)
      - track_id "3", object_class "Bicycle" — moving at 5 m/s northward

    Timestep spacing is exactly 1.0 second so all kinematic diffs are clean integers.
    """
    records = []
    for t in range(5):
        records.append({
            "track_id": "1", "object_class": "Car",
            "x": float(10 * t), "y": 0.0, "time": float(t),
            "speed": 10.0, "tangential_acc": 0.0, "lateral_acc": 0.0, "heading": 0.0,
        })
        records.append({
            "track_id": "2", "object_class": "Pedestrian",
            "x": 50.0, "y": 100.0, "time": float(t),
            "speed": 0.0, "tangential_acc": 0.0, "lateral_acc": 0.0, "heading": 0.0,
        })
        records.append({
            "track_id": "3", "object_class": "Bicycle",
            "x": 0.0, "y": float(5 * t), "time": float(t),
            "speed": 5.0, "tangential_acc": 0.0, "lateral_acc": 0.0, "heading": 90.0,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Minimal CSV content for parser tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_csv_content() -> str:
    """
    Minimal valid DFS Viewer annotation CSV with two objects and two timesteps each.
    The 12 metadata columns are placeholders; only columns 0 (track_id), 1 (class),
    and the trajectory blob (column 12+) matter to the parser.
    """
    meta = ";".join(["x"] * 10)  # 10 filler metadata columns after track_id + class
    # Each timestep: x, y, speed, tan_acc, lat_acc, time, heading
    traj_obj1 = "1.0;2.0;3.0;0.0;0.0;0.0;45.0;7.0;0.0;0.0;0.0;1.0;2.0;3.0;0.0;0.0;1.0;45.0"
    traj_obj2 = "10.0;20.0;0.0;0.0;0.0;0.0;0.0;10.0;20.0;0.0;0.0;1.0;0.0;10.0;20.0;0.0;0.0;1.0;0.0"
    lines = [
        f"1;Car;{meta};{traj_obj1}",
        f"2;Pedestrian;{meta};{traj_obj2}",
    ]
    return "\n".join(lines)


@pytest.fixture()
def valid_csv_file(tmp_path: Path, valid_csv_content: str) -> Path:
    """Write valid_csv_content to a temp file and return the Path."""
    p = tmp_path / "test_session_ann.csv"
    p.write_text(valid_csv_content, encoding="utf-8")
    return p


@pytest.fixture()
def malformed_csv_file(tmp_path: Path) -> Path:
    """CSV with one row whose trajectory blob has a token count not divisible by 7."""
    meta = ";".join(["x"] * 10)
    content = f"1;Car;{meta};1.0;2.0;3.0"  # only 3 tokens — malformed
    p = tmp_path / "malformed_ann.csv"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Minimal interaction DataFrame (feature engineering output shape)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_interaction_df() -> pd.DataFrame:
    """
    One vehicle-worker pair, 10 timesteps at 1-second intervals.

    The vehicle moves at constant 5 m/s toward the stationary worker.
    Initial distance = 50 m, so TTC = 50/5 = 10 s.

    Expected values (for tests):
      - rel_distance at t=0: 50.0
      - approach_speed at t=0: 5.0
      - ttc at t=0: 10.0
    """
    records = []
    for t in range(10):
        dist = 50.0 - 5.0 * t  # vehicle closing at 5 m/s
        records.append({
            "track_id_vuln": "2",
            "track_id_car": "1",
            "time": float(t),
            "rel_distance": max(0.1, dist),
            "rel_speed": 5.0,
            "speed_ms_vuln": 0.0,
            "speed_ms_car": 5.0,
            "accel_ms2_vuln": 0.0,
            "accel_ms2_car": 0.0,
            "approach_speed": 5.0,
            "ttc": max(0.1, dist) / 5.0,
            "future_rel_dist": max(0.1, dist - 5.0 * 4.0),
            "rel_dist_avg_2s": max(0.1, dist),
            "rel_speed_avg_2s": 5.0,
            "future_rel_dist_avg_2s": max(0.1, dist - 5.0 * 4.0),
        })
    return pd.DataFrame(records)
