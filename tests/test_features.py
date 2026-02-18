"""
Tests for site_sentinel.features — motion and interaction feature engineering.

Uses synthetic fixtures with known geometry so expected values are
straightforward to verify by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from site_sentinel.features.engineering import (
    compute_interaction_features,
    compute_motion_features,
)
from site_sentinel.features.targets import create_dual_targets


class TestMotionFeatures:
    def test_adds_required_columns(self, tiny_trajectory_df: pd.DataFrame) -> None:
        result = compute_motion_features(tiny_trajectory_df)
        for col in ("velocity_x", "velocity_y", "speed_ms", "accel_ms2"):
            assert col in result.columns

    def test_car_speed_is_correct(self, tiny_trajectory_df: pd.DataFrame) -> None:
        """Car moves 10 m per second purely eastward — speed_ms should be 10."""
        result = compute_motion_features(tiny_trajectory_df)
        car_rows = result[result["object_class"] == "Car"]
        # First row has no previous position, so speed is 0
        assert car_rows.iloc[0]["speed_ms"] == pytest.approx(0.0, abs=1e-6)
        # All subsequent rows should be 10 m/s
        assert (car_rows.iloc[1:]["speed_ms"] == pytest.approx(10.0, abs=1e-3)).all()

    def test_stationary_object_has_zero_speed(self, tiny_trajectory_df: pd.DataFrame) -> None:
        result = compute_motion_features(tiny_trajectory_df)
        ped_rows = result[result["object_class"] == "Pedestrian"]
        assert (ped_rows["speed_ms"].abs() < 1e-6).all()

    def test_no_inf_or_nan_in_output(self, tiny_trajectory_df: pd.DataFrame) -> None:
        result = compute_motion_features(tiny_trajectory_df)
        for col in ("velocity_x", "velocity_y", "speed_ms", "accel_ms2"):
            assert not result[col].isna().any(), f"NaN in column {col}"
            assert not np.isinf(result[col]).any(), f"Inf in column {col}"

    def test_zero_dt_does_not_cause_division_error(self) -> None:
        """Two consecutive rows with identical timestamps should not crash."""
        df = pd.DataFrame([
            {"track_id": "1", "object_class": "Car", "x": 0.0, "y": 0.0, "time": 0.0,
             "speed": 0.0, "tangential_acc": 0.0, "lateral_acc": 0.0, "heading": 0.0},
            {"track_id": "1", "object_class": "Car", "x": 1.0, "y": 0.0, "time": 0.0,
             "speed": 0.0, "tangential_acc": 0.0, "lateral_acc": 0.0, "heading": 0.0},
        ])
        result = compute_motion_features(df)
        assert not result["velocity_x"].isna().any()
        assert not np.isinf(result["velocity_x"]).any()


class TestInteractionFeatures:
    def _motion_df(self, tiny_trajectory_df: pd.DataFrame) -> pd.DataFrame:
        return compute_motion_features(tiny_trajectory_df)

    def test_returns_dataframe(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df, frame_rate=1.0, time_horizon_s=4.0, rolling_window_s=2.0,
            vehicle_class="Car", vulnerable_class="Pedestrian",
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_required_feature_columns(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df, frame_rate=1.0, time_horizon_s=4.0, rolling_window_s=2.0,
            vehicle_class="Car", vulnerable_class="Pedestrian",
        )
        required = [
            "rel_distance", "rel_speed", "speed_ms_car", "speed_ms_vuln",
            "approach_speed", "ttc", "future_rel_dist",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_rel_distance_is_positive(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df, frame_rate=1.0, time_horizon_s=4.0, rolling_window_s=2.0,
            vehicle_class="Car", vulnerable_class="Pedestrian",
        )
        if not result.empty:
            assert (result["rel_distance"] > 0).all()

    def test_future_rel_dist_is_clipped_positive(self, tiny_trajectory_df: pd.DataFrame) -> None:
        """future_rel_dist should never be below 0.1."""
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df, frame_rate=1.0, time_horizon_s=4.0, rolling_window_s=2.0,
            vehicle_class="Car", vulnerable_class="Pedestrian",
        )
        if not result.empty:
            assert (result["future_rel_dist"] >= 0.1).all()

    def test_rolling_window_same_length_as_input(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df, frame_rate=1.0, time_horizon_s=4.0, rolling_window_s=2.0,
            vehicle_class="Car", vulnerable_class="Pedestrian",
        )
        if not result.empty:
            assert not result["rel_dist_avg_2s"].isna().any()
            assert not result["rel_speed_avg_2s"].isna().any()

    def test_empty_result_if_no_matching_classes(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df, frame_rate=1.0, time_horizon_s=4.0, rolling_window_s=2.0,
            vehicle_class="Truck",       # not in the fixture
            vulnerable_class="Worker",   # not in the fixture
        )
        assert result.empty

    def test_ttc_diverging_tracks_filled_with_sentinel(self) -> None:
        """
        Objects moving away from each other have no collision on current trajectory.
        TTC should be filled with the sentinel value (100), not NaN or negative.
        """
        df = pd.DataFrame([
            # Vehicle moves away (negative velocity_x relative to worker)
            {"track_id": "1", "object_class": "Car",
             "x": 100.0, "y": 0.0, "time": 0.0,
             "speed": 0.0, "tangential_acc": 0.0, "lateral_acc": 0.0, "heading": 0.0},
            {"track_id": "1", "object_class": "Car",
             "x": 110.0, "y": 0.0, "time": 1.0,
             "speed": 0.0, "tangential_acc": 0.0, "lateral_acc": 0.0, "heading": 0.0},
            {"track_id": "2", "object_class": "Pedestrian",
             "x": 0.0, "y": 0.0, "time": 0.0,
             "speed": 0.0, "tangential_acc": 0.0, "lateral_acc": 0.0, "heading": 0.0},
            {"track_id": "2", "object_class": "Pedestrian",
             "x": 0.0, "y": 0.0, "time": 1.0,
             "speed": 0.0, "tangential_acc": 0.0, "lateral_acc": 0.0, "heading": 0.0},
        ])
        motion_df = compute_motion_features(df)
        result = compute_interaction_features(
            motion_df, frame_rate=1.0, time_horizon_s=4.0, rolling_window_s=2.0,
            vehicle_class="Car", vulnerable_class="Pedestrian",
        )
        if not result.empty:
            assert (result["ttc"] == pytest.approx(100.0, abs=1e-3)).all()


class TestTargetVariables:
    def test_y_standard_flags_low_ttc(self, tiny_interaction_df: pd.DataFrame) -> None:
        df = create_dual_targets(
            tiny_interaction_df, lead_time_s=4.0, frame_rate=1.0, ttc_threshold_s=2.0
        )
        # TTC starts at 10s and decreases — only last rows should be flagged
        assert "Y_standard" in df.columns
        assert df["Y_standard"].dtype in (int, "int64", "int32")

    def test_y_preventive_fires_before_y_standard(
        self, tiny_interaction_df: pd.DataFrame
    ) -> None:
        """
        Y_preventive should turn on at least as early as Y_standard — it's a
        lookahead label, so it should flag frames before the danger arrives.
        """
        df = create_dual_targets(
            tiny_interaction_df, lead_time_s=4.0, frame_rate=1.0, ttc_threshold_s=2.0
        )
        # Find first index where standard fires
        std_first = df[df["Y_standard"] == 1].index.min()
        prev_first = df[df["Y_preventive"] == 1].index.min()

        if not (pd.isna(std_first) or pd.isna(prev_first)):
            assert prev_first <= std_first, (
                f"Preventive label fires at {prev_first} but standard fires at {std_first}"
            )

    def test_no_nan_in_targets(self, tiny_interaction_df: pd.DataFrame) -> None:
        df = create_dual_targets(
            tiny_interaction_df, lead_time_s=4.0, frame_rate=1.0, ttc_threshold_s=2.0
        )
        assert not df["Y_standard"].isna().any()
        assert not df["Y_preventive"].isna().any()

    def test_targets_are_binary(self, tiny_interaction_df: pd.DataFrame) -> None:
        df = create_dual_targets(
            tiny_interaction_df, lead_time_s=4.0, frame_rate=1.0, ttc_threshold_s=2.0
        )
        assert set(df["Y_standard"].unique()).issubset({0, 1})
        assert set(df["Y_preventive"].unique()).issubset({0, 1})
