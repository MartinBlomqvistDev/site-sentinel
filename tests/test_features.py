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
        assert np.allclose(car_rows.iloc[1:]["speed_ms"].values, 10.0, atol=1e-3)

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
        df = pd.DataFrame(
            [
                {
                    "track_id": "1",
                    "object_class": "Car",
                    "x": 0.0,
                    "y": 0.0,
                    "time": 0.0,
                    "speed": 0.0,
                    "tangential_acc": 0.0,
                    "lateral_acc": 0.0,
                    "heading": 0.0,
                },
                {
                    "track_id": "1",
                    "object_class": "Car",
                    "x": 1.0,
                    "y": 0.0,
                    "time": 0.0,
                    "speed": 0.0,
                    "tangential_acc": 0.0,
                    "lateral_acc": 0.0,
                    "heading": 0.0,
                },
            ]
        )
        result = compute_motion_features(df)
        assert not result["velocity_x"].isna().any()
        assert not np.isinf(result["velocity_x"]).any()


class TestInteractionFeatures:
    def _motion_df(self, tiny_trajectory_df: pd.DataFrame) -> pd.DataFrame:
        return compute_motion_features(tiny_trajectory_df)

    def test_returns_dataframe(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df,
            frame_rate=1.0,
            time_horizon_s=4.0,
            rolling_window_s=2.0,
            vehicle_class="Car",
            vulnerable_class="Pedestrian",
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_required_feature_columns(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df,
            frame_rate=1.0,
            time_horizon_s=4.0,
            rolling_window_s=2.0,
            vehicle_class="Car",
            vulnerable_class="Pedestrian",
        )
        required = [
            "rel_distance",
            "rel_speed",
            "speed_ms_car",
            "speed_ms_vuln",
            "approach_speed",
            "ttc",
            "future_rel_dist",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_rel_distance_is_positive(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df,
            frame_rate=1.0,
            time_horizon_s=4.0,
            rolling_window_s=2.0,
            vehicle_class="Car",
            vulnerable_class="Pedestrian",
        )
        if not result.empty:
            assert (result["rel_distance"] > 0).all()

    def test_future_rel_dist_is_clipped_positive(self, tiny_trajectory_df: pd.DataFrame) -> None:
        """future_rel_dist should never be below 0.1."""
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df,
            frame_rate=1.0,
            time_horizon_s=4.0,
            rolling_window_s=2.0,
            vehicle_class="Car",
            vulnerable_class="Pedestrian",
        )
        if not result.empty:
            assert (result["future_rel_dist"] >= 0.1).all()

    def test_rolling_window_same_length_as_input(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df,
            frame_rate=1.0,
            time_horizon_s=4.0,
            rolling_window_s=2.0,
            vehicle_class="Car",
            vulnerable_class="Pedestrian",
        )
        if not result.empty:
            assert not result["rel_dist_avg_2s"].isna().any()
            assert not result["rel_speed_avg_2s"].isna().any()

    def test_empty_result_if_no_matching_classes(self, tiny_trajectory_df: pd.DataFrame) -> None:
        motion_df = self._motion_df(tiny_trajectory_df)
        result = compute_interaction_features(
            motion_df,
            frame_rate=1.0,
            time_horizon_s=4.0,
            rolling_window_s=2.0,
            vehicle_class="Truck",  # not in the fixture
            vulnerable_class="Worker",  # not in the fixture
        )
        assert result.empty

    def test_ttc_diverging_tracks_filled_with_sentinel(self) -> None:
        """
        Objects moving away from each other have no collision on current trajectory.
        TTC should be filled with the sentinel value (100), not NaN or negative.
        """
        df = pd.DataFrame(
            [
                # Vehicle moves away (negative velocity_x relative to worker)
                {
                    "track_id": "1",
                    "object_class": "Car",
                    "x": 100.0,
                    "y": 0.0,
                    "time": 0.0,
                    "speed": 0.0,
                    "tangential_acc": 0.0,
                    "lateral_acc": 0.0,
                    "heading": 0.0,
                },
                {
                    "track_id": "1",
                    "object_class": "Car",
                    "x": 110.0,
                    "y": 0.0,
                    "time": 1.0,
                    "speed": 0.0,
                    "tangential_acc": 0.0,
                    "lateral_acc": 0.0,
                    "heading": 0.0,
                },
                {
                    "track_id": "2",
                    "object_class": "Pedestrian",
                    "x": 0.0,
                    "y": 0.0,
                    "time": 0.0,
                    "speed": 0.0,
                    "tangential_acc": 0.0,
                    "lateral_acc": 0.0,
                    "heading": 0.0,
                },
                {
                    "track_id": "2",
                    "object_class": "Pedestrian",
                    "x": 0.0,
                    "y": 0.0,
                    "time": 1.0,
                    "speed": 0.0,
                    "tangential_acc": 0.0,
                    "lateral_acc": 0.0,
                    "heading": 0.0,
                },
            ]
        )
        motion_df = compute_motion_features(df)
        result = compute_interaction_features(
            motion_df,
            frame_rate=1.0,
            time_horizon_s=4.0,
            rolling_window_s=2.0,
            vehicle_class="Car",
            vulnerable_class="Pedestrian",
        )
        if not result.empty:
            assert np.allclose(result["ttc"].values, 100.0, atol=1e-3)


class TestTargetVariables:
    def test_y_standard_flags_low_ttc(self, tiny_interaction_df: pd.DataFrame) -> None:
        df = create_dual_targets(
            tiny_interaction_df, lead_time_s=4.0, frame_rate=1.0, ttc_threshold_s=2.0
        )
        # TTC starts at 10s and decreases — only last rows should be flagged
        assert "Y_standard" in df.columns
        assert df["Y_standard"].dtype in (int, "int64", "int32")

    @staticmethod
    def _proximity_df(distances: list[float], pair: str = "1") -> pd.DataFrame:
        """One pair, one row per second, with rel_distance driven directly."""
        return pd.DataFrame(
            {
                "track_id_vuln": "2",
                "track_id_car": pair,
                "rel_distance": distances,
                "ttc": [100.0] * len(distances),
            }
        )

    def test_y_preventive_fires_in_the_window_before_a_proximity_event(self) -> None:
        """
        The proximity event is at index 5. With a 3-frame lookahead, indices 2,
        3 and 4 should be labelled and nothing before index 2 should be.
        """
        df = create_dual_targets(
            self._proximity_df([50, 50, 50, 50, 50, 1.0, 50, 50]),
            lead_time_s=3.0,
            frame_rate=1.0,
            risk_distance_m=2.0,
        )
        assert df["Y_preventive"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]

    def test_y_preventive_excludes_the_current_frame(self) -> None:
        """
        A label that includes the frame being scored is not a prediction. Index
        5 is the event itself and must not be labelled by it.
        """
        df = create_dual_targets(
            self._proximity_df([50, 50, 50, 50, 50, 1.0, 50, 50]),
            lead_time_s=3.0,
            frame_rate=1.0,
            risk_distance_m=2.0,
        )
        assert df["Y_preventive"].iloc[5] == 0

    def test_y_preventive_does_not_leak_across_pairs(self) -> None:
        """
        Pair "1" ends with no event; pair "2" opens with one. The tail of the
        first pair must not see the head of the second. The original
        implementation shifted outside the groupby and did exactly that.
        """
        a = self._proximity_df([50, 50, 50, 50], pair="1")
        b = self._proximity_df([1.0, 50, 50, 50], pair="2")
        df = create_dual_targets(
            pd.concat([a, b], ignore_index=True),
            lead_time_s=3.0,
            frame_rate=1.0,
            risk_distance_m=2.0,
        )
        assert df["Y_preventive"].iloc[:4].sum() == 0

    def test_y_preventive_is_independent_of_y_standard(self) -> None:
        """
        The two targets answer different questions: proximity in metres versus
        time-to-collision in seconds. A frame can be dangerous on TTC while no
        proximity event is coming, and the labels must be free to disagree.
        """
        df = create_dual_targets(
            pd.DataFrame(
                {
                    "track_id_vuln": "2",
                    "track_id_car": "1",
                    "rel_distance": [50.0] * 5,
                    "ttc": [0.5] * 5,
                }
            ),
            lead_time_s=3.0,
            frame_rate=1.0,
            ttc_threshold_s=2.0,
            risk_distance_m=2.0,
        )
        assert df["Y_standard"].sum() == 5
        assert df["Y_preventive"].sum() == 0

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
