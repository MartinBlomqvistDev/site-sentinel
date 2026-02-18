"""
Target variable creation for the Site Sentinel risk prediction pipeline.

The model is trained on two binary labels per frame:

  Y_standard   — is the current TTC already dangerously low?
                 Label = 1 when TTC ≤ ttc_threshold_s at this frame.

  Y_preventive — will the situation become dangerous within the next
                 lead_time_s seconds?
                 Label = 1 when Y_standard will be 1 at any point in
                 the next lead_time_s seconds.

The preventive label is what makes the system useful in practice — it
fires *before* the situation becomes immediately dangerous, giving workers
enough time to react.

Usage:

    from site_sentinel.features.targets import create_dual_targets

    interaction_df = create_dual_targets(
        interaction_df,
        lead_time_s=4.0,
        frame_rate=29.97,
        ttc_threshold_s=2.0,
    )
    # New columns: Y_standard, Y_preventive
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def create_dual_targets(
    df: pd.DataFrame,
    lead_time_s: float,
    frame_rate: float,
    ttc_threshold_s: float = 2.0,
) -> pd.DataFrame:
    """
    Add Y_standard and Y_preventive binary target columns to the interaction DataFrame.

    Args:
        df: Interaction feature DataFrame with a 'ttc' column.
            Must also have 'track_id_vuln' and 'track_id_car' columns if there
            are multiple pairs (to avoid leaking labels across pairs).
        lead_time_s: Lookahead horizon for the preventive label (seconds).
        frame_rate: Recording frame rate in Hz (used to convert lead_time_s to frames).
        ttc_threshold_s: TTC below this threshold triggers the immediate danger label.

    Returns:
        The input DataFrame with two new columns added in place:
            Y_standard   (int, 0 or 1)
            Y_preventive (int, 0 or 1)
    """
    df = df.copy()
    lead_frames = max(1, int(lead_time_s * frame_rate))

    # Immediate danger: is TTC already dangerously low right now?
    df["Y_standard"] = (df["ttc"] <= ttc_threshold_s).astype(int)

    # Preventive danger: will immediate danger occur within the next lead_frames frames?
    # We compute this per (vehicle, worker) pair to prevent labels from one pair
    # bleeding into another.
    pair_cols = [c for c in ("track_id_vuln", "track_id_car") if c in df.columns]

    if pair_cols:
        df["Y_preventive"] = (
            df.groupby(pair_cols)["Y_standard"]
            .transform(lambda s: _rolling_future_max(s, lead_frames))
        )
    else:
        df["Y_preventive"] = _rolling_future_max(df["Y_standard"], lead_frames)

    pos_rate = df["Y_preventive"].mean()
    logger.info(
        "Targets created: Y_preventive positive rate = %.1f%% (lead=%gs, threshold=%gs)",
        pos_rate * 100,
        lead_time_s,
        ttc_threshold_s,
    )

    return df


def _rolling_future_max(series: pd.Series, window: int) -> pd.Series:
    """
    At each position i, return the maximum value in series[i : i + window].

    This is equivalent to 'will this event happen in the next N frames?'
    The last `window` rows will be filled with 0 since we have no future data.
    """
    # Reverse → rolling max → reverse back gives max(series[i : i+window]) at position i.
    # No shift needed — the double-reversal already aligns the window correctly.
    return (
        series[::-1]
        .rolling(window=window, min_periods=1)
        .max()[::-1]
        .fillna(0)
        .astype(int)
    )
