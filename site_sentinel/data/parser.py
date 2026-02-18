"""
Parser for DFS Viewer trajectory CSV files (CONCOR-D dataset format).

The CONCOR-D dataset stores trajectories in a non-standard CSV format produced
by the DFS Viewer (SkyViewer) software. Each row represents one tracked object.
The first 12 columns are metadata (track ID, class, bounding box coordinates,
etc.), and the rest of the row is a semicolon-delimited blob where every 7 values
form one timestep:

    x_utm, y_utm, speed, tangential_acc, lateral_acc, timestamp, heading

This module provides a single, authoritative parser for this format. It was
consolidated from four near-identical copies that had diverged across the pipeline.

Usage:

    from site_sentinel.data.parser import parse_trajectory_csv

    df = parse_trajectory_csv("data/raw/CONCOR-D/.../session_ann.csv")
    # Returns a DataFrame with columns:
    # track_id, object_class, x, y, speed, tangential_acc, lateral_acc, time, heading
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# DFS Viewer CSV format constants
_METADATA_COLS = 12   # columns before the trajectory blob
_STRIDE = 7           # values per timestep in the trajectory blob
_TIMESTEP_FIELDS = [
    "x",
    "y",
    "speed",
    "tangential_acc",
    "lateral_acc",
    "time",
    "heading",
]


def parse_trajectory_csv(path: Path | str) -> pd.DataFrame:
    """
    Parse a single DFS Viewer annotation CSV into a long-format DataFrame.

    Each row in the source file becomes multiple rows in the output — one per
    timestep recorded for that object. The returned DataFrame is sorted by
    (track_id, time) so kinematic features can be computed with a simple diff().

    Args:
        path: Path to the annotation CSV file.

    Returns:
        DataFrame with columns:
            track_id (str), object_class (str), x (float), y (float),
            speed (float), tangential_acc (float), lateral_acc (float),
            time (float), heading (float)
        Returns an empty DataFrame (with the correct columns) if the file
        contains no parseable trajectory data.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    records: list[dict] = []
    skipped_rows = 0

    with path.open(encoding="utf-8", errors="replace") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # Split off exactly _METADATA_COLS fields; the rest is the trajectory blob
            parts = line.split(";", _METADATA_COLS)
            if len(parts) < _METADATA_COLS + 1:
                continue

            track_id = parts[0].strip()
            object_class = parts[1].strip()
            trajectory_blob = parts[_METADATA_COLS]

            tokens = [t for t in trajectory_blob.split(";") if t.strip()]

            if not tokens:
                continue

            if len(tokens) % _STRIDE != 0:
                logger.warning(
                    "%s line %d: skipping malformed trajectory blob "
                    "(got %d tokens, expected a multiple of %d)",
                    path.name,
                    line_num,
                    len(tokens),
                    _STRIDE,
                )
                skipped_rows += 1
                continue

            for i in range(0, len(tokens), _STRIDE):
                group = tokens[i : i + _STRIDE]
                try:
                    row: dict = {
                        "track_id": track_id,
                        "object_class": object_class,
                    }
                    row.update(dict(zip(_TIMESTEP_FIELDS, map(float, group), strict=True)))
                    records.append(row)
                except ValueError:
                    logger.debug(
                        "%s: non-numeric token at stride index %d — skipped", path.name, i
                    )

    if skipped_rows:
        logger.info("%s: skipped %d malformed rows", path.name, skipped_rows)

    if not records:
        logger.warning("%s: no valid trajectory data found", path.name)
        return pd.DataFrame(columns=["track_id", "object_class"] + _TIMESTEP_FIELDS)

    df = pd.DataFrame(records)
    df = df.sort_values(["track_id", "time"]).reset_index(drop=True)
    return df
