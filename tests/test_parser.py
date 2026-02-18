"""
Tests for site_sentinel.data.parser — the canonical DFS Viewer CSV parser.

All tests use synthetic data from conftest.py fixtures.
No real dataset files are required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from site_sentinel.data.parser import parse_trajectory_csv


class TestParseValidFile:
    def test_returns_dataframe(self, valid_csv_file: Path) -> None:
        df = parse_trajectory_csv(valid_csv_file)
        assert not df.empty

    def test_has_required_columns(self, valid_csv_file: Path) -> None:
        df = parse_trajectory_csv(valid_csv_file)
        for col in ("track_id", "object_class", "x", "y", "time", "speed"):
            assert col in df.columns, f"Missing column: {col}"

    def test_object_classes_preserved(self, valid_csv_file: Path) -> None:
        df = parse_trajectory_csv(valid_csv_file)
        classes = set(df["object_class"].unique())
        assert "Car" in classes
        assert "Pedestrian" in classes

    def test_sorted_by_track_then_time(self, valid_csv_file: Path) -> None:
        df = parse_trajectory_csv(valid_csv_file)
        for track_id, group in df.groupby("track_id"):
            assert group["time"].is_monotonic_increasing, (
                f"Time not monotonic for track {track_id}"
            )


class TestColumnDtypes:
    def test_numeric_columns_are_float(self, valid_csv_file: Path) -> None:
        df = parse_trajectory_csv(valid_csv_file)
        for col in ("x", "y", "time", "speed"):
            assert df[col].dtype == float, f"Column {col} is not float64"


class TestMissingFile:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist_ann.csv"
        with pytest.raises(FileNotFoundError):
            parse_trajectory_csv(missing)


class TestMalformedBlob:
    def test_warns_but_returns_empty(self, malformed_csv_file: Path) -> None:
        """
        A file with a trajectory blob whose token count is not divisible by 7
        should not crash — it should log a warning and return an empty DataFrame.
        """
        df = parse_trajectory_csv(malformed_csv_file)
        # Malformed row is skipped; file has no valid rows so result is empty
        assert df.empty or len(df) == 0


class TestEmptyFile:
    def test_empty_file_returns_empty_df(self, tmp_path: Path) -> None:
        p = tmp_path / "empty_ann.csv"
        p.write_text("", encoding="utf-8")
        df = parse_trajectory_csv(p)
        assert df.empty
        assert "track_id" in df.columns
