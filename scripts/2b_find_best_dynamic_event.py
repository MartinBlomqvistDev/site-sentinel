# scripts/2b_find_best_dynamic_event_safe_v4.py
# -----------------------------------------------------------------
# Robust, artifact-proof, risk-scoring for near-misses
# Outputs top 20 events and separately prints top 5 videos + timestamps

import pandas as pd
import numpy as np
import os
import math

# --- CONFIGURATION ---
ROOT_DIRECTORY = "C:/DS24/Site_Sentinel/data/raw/CONCOR-D/3W_SidStP_Trajectory"
OUTPUT_CSV_PATH = "data/analysis_results/top_20_dynamic_events_explainable_filtered_safe.csv"

INTERACTION_DISTANCE_THRESHOLD = 3.0   # meters
NUM_TOP_EVENTS = 20
FPS = 29.97
MAX_SPEED = 50.0                        # m/s
MIN_TIME_SKIP = 10.0                    # seconds to ignore at start of video
MIN_TRACK_DURATION = 1.0                # seconds, remove very short tracks
MAX_REALISTIC_SPEED = 20.0              # m/s (~72 km/h cap for risk scoring)
MIN_VALID_FRAME = 100                   # <--- NEW: skip first 100 frames globally
# ---------------------

def final_parser_safe(filepath):
    """Parse CONCOR-D CSV safely, skipping corrupted or incomplete lines."""
    all_rows = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split(';', 12)
                    if len(parts) < 13:
                        continue
                    meta_parts, trajectory_blob = parts[:12], parts[12]
                    track_id = int(meta_parts[0])
                    object_type = meta_parts[1].strip()
                    trajectory_points = trajectory_blob.split(';')
                    for i in range(0, len(trajectory_points), 7):
                        chunk = trajectory_points[i:i + 7]
                        if len(chunk) == 7:
                            all_rows.append({
                                'trackId': track_id,
                                'class': object_type,
                                'x': float(chunk[0]),
                                'y': float(chunk[1]),
                                'time': float(chunk[5]),
                                'speed': float(chunk[2]) if chunk[2] else 0.0
                            })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"  -> ERROR reading file '{filepath}': {e}")
        return pd.DataFrame()
    return pd.DataFrame(all_rows)

def classify_event(row):
    """Categorize the type of interaction for explainability."""
    try:
        delta_x = row['x_car'] - row['x_vuln']
        delta_y = row['y_car'] - row['y_vuln']
        dot = delta_x * row['velocity_x_car'] + delta_y * row['velocity_y_car']
        car_speed = np.sqrt(row['velocity_x_car']**2 + row['velocity_y_car']**2)
        safe_dist = max(row['rel_distance'], 0.5)
        angle = math.degrees(math.acos(np.clip(dot / (safe_dist * car_speed + 1e-6), -1, 1)))
    except Exception:
        angle = 180

    if row['approach_speed'] > 1.0 and angle < 25:
        return 'collision_course'
    elif row['rel_distance'] < 2.0 and row['speed_car'] > 3.0:
        return 'close_pass'
    elif row['speed_car'] > 1.0 and row['rel_distance'] < 3.0:
        return 'static_exposure'
    elif 25 <= angle < 70:
        return 'crossing_path'
    else:
        return 'same_direction'

def find_best_dynamic_event(df):
    if df.empty or 'trackId' not in df.columns:
        return None

    df = df.sort_values(by=['trackId', 'time']).reset_index(drop=True)

    # --- remove very short tracks ---
    durations = df.groupby("trackId")["time"].agg(lambda x: x.max() - x.min())
    valid_tracks = durations[durations > MIN_TRACK_DURATION].index
    df = df[df['trackId'].isin(valid_tracks)]

    vulnerable = df[df['class'].isin(['Bicycle', 'Pedestrian'])].copy()
    vehicles = df[df['class'] == 'Car'].copy()
    if vulnerable.empty or vehicles.empty:
        return None

    # --- compute frame numbers ---
    df['frame'] = (df['time'] * FPS).round().astype(int)

    # --- REMOVE INITIAL VIDEO ARTIFACTS ---
    df = df[df['frame'] >= MIN_VALID_FRAME].copy()
    if df.empty:
        return None

    # --- compute velocities ---
    for df_track in [vulnerable, vehicles]:
        df_track['delta_t'] = df_track.groupby('trackId')['time'].diff().fillna(0.0).replace(0, np.nan)
        df_track['velocity_x'] = ((df_track['x'] - df_track.groupby('trackId')['x'].shift(1)) / df_track['delta_t']).fillna(0)
        df_track['velocity_y'] = ((df_track['y'] - df_track.groupby('trackId')['y'].shift(1)) / df_track['delta_t']).fillna(0)
        df_track['velocity_x'] = df_track['velocity_x'].clip(-MAX_SPEED, MAX_SPEED)
        df_track['velocity_y'] = df_track['velocity_y'].clip(-MAX_SPEED, MAX_SPEED)
        df_track['speed'] = np.sqrt(df_track['velocity_x']**2 + df_track['velocity_y']**2)
        df_track['frame'] = (df_track['time'] * FPS).round().astype(int)

    merged = pd.merge(vulnerable, vehicles, on='frame', suffixes=('_vuln', '_car'))
    if merged.empty:
        return None

    # --- drop negative times/frames just in case ---
    merged = merged[(merged['frame'] > MIN_VALID_FRAME) & (merged['time_vuln'] >= 0)]

    # --- compute relative distance and approach speed ---
    delta_x = merged['x_car'] - merged['x_vuln']
    delta_y = merged['y_car'] - merged['y_vuln']
    delta_vx = merged['velocity_x_car'] - merged['velocity_x_vuln']
    delta_vy = merged['velocity_y_car'] - merged['velocity_y_vuln']

    merged['rel_distance'] = np.sqrt(delta_x**2 + delta_y**2)
    safe_dist = merged['rel_distance'].clip(lower=0.5)
    merged['approach_speed'] = -(delta_x * delta_vx + delta_y * delta_vy) / safe_dist
    merged['approach_speed'] = merged['approach_speed'].clip(lower=0, upper=MAX_SPEED)

    # --- TTC ---
    merged['ttc'] = np.where(merged['approach_speed'] > 0,
                             merged['rel_distance'] / merged['approach_speed'],
                             np.nan)

    # --- realistic interaction filter ---
    candidates = merged[
        (merged['speed_car'] > 2.0) &
        (merged['speed_vuln'] >= 0.0) &
        (merged['rel_distance'] <= INTERACTION_DISTANCE_THRESHOLD) &
        (merged['rel_distance'] > 0.1)
    ].copy()
    if candidates.empty:
        return None

    # --- CAP SPEEDS FOR RISK SCORE ---
    candidates['approach_speed_capped'] = candidates['approach_speed'].clip(0, MAX_REALISTIC_SPEED)
    candidates['speed_car_capped'] = candidates['speed_car'].clip(0, MAX_REALISTIC_SPEED)

    # --- compute risk score ---
    candidates['risk_directional'] = (1 / candidates['rel_distance']) * candidates['approach_speed_capped']
    candidates['risk_proximity'] = (1 / (candidates['rel_distance']**2)) * candidates['speed_car_capped']
    candidates['risk_score'] = 0.6 * candidates['risk_directional'] + 0.4 * candidates['risk_proximity']

    candidates['event_type'] = candidates.apply(classify_event, axis=1)
    candidates['near_miss'] = candidates['rel_distance'] < 1.5  # flag extreme near-misses

    best_event = candidates.loc[candidates['risk_score'].idxmax()].copy()
    return best_event

def main():
    target_files = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(ROOT_DIRECTORY)
        for f in fn if f.endswith('_ann.csv')
    ]
    all_events = []

    for filepath in target_files:
        filename = os.path.basename(filepath)
        raw_df = final_parser_safe(filepath)
        if raw_df.empty:
            continue
        best_event = find_best_dynamic_event(raw_df)
        if best_event is not None:
            event_data = best_event.to_dict()
            event_data['file'] = filename
            event_data['full_path'] = filepath
            all_events.append(event_data)

    if not all_events:
        print("No dynamic events found.")
        return

    results_df = pd.DataFrame(all_events)
    top_df = results_df.sort_values(by='risk_score', ascending=False).head(NUM_TOP_EVENTS)

    display_cols = ['risk_score', 'rel_distance', 'approach_speed', 'ttc',
                    'event_type', 'near_miss', 'file', 'frame', 'time_vuln',
                    'trackId_vuln', 'trackId_car', 'full_path']
    top_df = top_df[display_cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    top_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\n🏆 Top Dynamic & Explainable Events Found! 🏆")
    print(top_df.to_string())
    print(f"\n✅ Results saved to '{OUTPUT_CSV_PATH}'.")

    # --- Top 5 videos + timestamps ---
    top5_videos = top_df.sort_values('risk_score', ascending=False).head(5)
    print("\n🎯 Top 5 Videos + Near-Miss Timestamps 🎯")
    for idx, row in top5_videos.iterrows():
        near_miss_str = " ⚠️ Near-Miss!" if row['near_miss'] else ""
        print(f"{row['file']}  →  {row['time_vuln']:.2f} s  (frame {row['frame']}){near_miss_str}")

if __name__ == "__main__":
    main()
