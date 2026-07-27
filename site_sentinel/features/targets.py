"""
Target variable creation for the Site Sentinel risk prediction pipeline.

The model is trained on two binary labels per frame, from two independent
signals rather than one nested inside the other:

  Y_standard   — is the current time-to-collision already dangerously low?
                 Label = 1 when TTC <= ttc_threshold_s at this frame.

  Y_preventive — will a vehicle come physically close to a person within the
                 next lead_time_s seconds?
                 Label = 1 when rel_distance <= risk_distance_m at any point
                 in the next lead_time_s seconds, not counting this frame.

The preventive label is what makes the system useful in practice: it fires
*before* the situation becomes immediately dangerous.

**Why proximity and not TTC for the preventive label.** A near-miss is the
event itself, two bodies nearly touching, and distance measures it directly
where TTC is a kinematic proxy for it. Deriving Y_preventive from Y_standard
instead would make the two targets the same question at two horizons, which
is not what a dual-target model is for.

**A note on the thresholds.** Both happen to be 2.0 and they are not the same
quantity: ``risk_distance_m`` is metres, ``ttc_threshold_s`` is seconds. An
earlier refactor unified them on the shared literal and silently moved the
preventive label onto TTC. The names carry the unit now so that cannot recur.

Usage:

    from site_sentinel.features.targets import create_dual_targets

    interaction_df = create_dual_targets(
        interaction_df,
        lead_time_s=4.0,
        frame_rate=29.97,
        ttc_threshold_s=2.0,
        risk_distance_m=2.0,
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
    risk_distance_m: float = 2.0,
) -> pd.DataFrame:
    """
    Add Y_standard and Y_preventive binary target columns to the interaction DataFrame.

    Args:
        df: Interaction feature DataFrame with 'ttc' and 'rel_distance' columns.
            Must also have 'track_id_vuln' and 'track_id_car' columns if there
            are multiple pairs (to avoid leaking labels across pairs).
        lead_time_s: Lookahead horizon for the preventive label (seconds).
        frame_rate: Recording frame rate in Hz (used to convert lead_time_s to frames).
        ttc_threshold_s: TTC at or below this, in seconds, is immediate danger.
        risk_distance_m: Separation at or below this, in metres, is a proximity event.

    Returns:
        The input DataFrame with two new columns added in place:
            Y_standard   (int, 0 or 1)
            Y_preventive (int, 0 or 1)
    """
    df = df.copy()
    lead_frames = max(1, int(lead_time_s * frame_rate))

    # Immediate danger: is TTC already dangerously low right now?
    df["Y_standard"] = (df["ttc"] <= ttc_threshold_s).astype(int)

    # Proximity event: is a vehicle physically close to a person right now? This
    # is the signal the preventive label looks ahead for, and it is deliberately
    # not Y_standard: two independent hazard definitions rather than one target
    # nested inside the other.
    proximity = (df["rel_distance"] <= risk_distance_m).astype(int)

    # Preventive danger: will a proximity event occur in the next lead_frames
    # frames? Computed per (vehicle, worker) pair via transform, so no label can
    # cross a pair boundary. The original did the shift outside the groupby,
    # which let the tail of each pair read the head of the next one.
    pair_cols = [c for c in ("track_id_vuln", "track_id_car") if c in df.columns]

    if pair_cols:
        df["Y_preventive"] = proximity.groupby([df[c] for c in pair_cols]).transform(
            lambda s: _rolling_future_max(s, lead_frames)
        )
    else:
        df["Y_preventive"] = _rolling_future_max(proximity, lead_frames)

    logger.info(
        "Targets created: Y_standard %.1f%% positive (TTC <= %gs), "
        "Y_preventive %.1f%% positive (within %gm in the next %gs)",
        100 * df["Y_standard"].mean(),
        ttc_threshold_s,
        100 * df["Y_preventive"].mean(),
        risk_distance_m,
        lead_time_s,
    )

    return df


def _rolling_future_max(series: pd.Series, window: int) -> pd.Series:
    """
    At each position i, return the maximum value in series[i+1 : i+window+1].

    This is 'will this event happen in the next N frames?', and the current
    frame is excluded on purpose: a label that includes the frame being scored
    is not a prediction. The last `window` rows fill with 0, since there is no
    future left to look at.
    """
    # Shift first so position i sees i+1 onward, then reverse, rolling max, and
    # reverse back to get max(series[i+1 : i+window+1]) at position i.
    ahead = series.shift(-1)
    return ahead[::-1].rolling(window=window, min_periods=1).max()[::-1].fillna(0).astype(int)
