# scripts/3_feature_eng.py

import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
TOP_EVENTS_CSV = "data/analysis_results/top_20_dynamic_events.csv"  # Keep the same name
OUTPUT_DIR = "data/analysis_results"
ROOT_DIRECTORY = "C:/DS24/Site_Sentinel/data/raw/CONCOR-D/3W_SidStP_Trajectory"
SELECTED_EVENT_RANK = 1
FRAME_RATE = 29.97
TIME_HORIZON = 1.5  # seconds ahead for preventive features
# ---------------------

def final_parser_v5(filepath):
    all_rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                parts = line.split(';', 12)
                if len(parts) < 13: continue
                meta_parts, trajectory_blob = parts[:12], parts[12]
                track_id, object_type = int(meta_parts[0]), meta_parts[1].strip()
                trajectory_points = trajectory_blob.split(';')
                for i in range(0, len(trajectory_points), 7):
                    chunk = trajectory_points[i:i + 7]
                    if len(chunk) == 7: 
                        all_rows.append({
                            'trackId': track_id,
                            'class': object_type,
                            'x': float(chunk[0]),
                            'y': float(chunk[1]),
                            'time': float(chunk[5])
                        })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(all_rows)


def calculate_motion_features(df):
    print("  Calculating motion features...")
    df = df.sort_values(by=['trackId', 'time']).reset_index(drop=True)
    df['delta_t'] = df.groupby('trackId')['time'].diff()
    df['delta_x'] = df.groupby('trackId')['x'].diff()
    df['delta_y'] = df.groupby('trackId')['y'].diff()
    df['velocity_x'] = (df['delta_x'] / df['delta_t']).replace([np.inf, -np.inf], 0).fillna(0)
    df['velocity_y'] = (df['delta_y'] / df['delta_t']).replace([np.inf, -np.inf], 0).fillna(0)
    df['speed_ms'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    df['delta_speed'] = df.groupby('trackId')['speed_ms'].diff()
    df['accel_ms2'] = (df['delta_speed'] / df['delta_t']).replace([np.inf, -np.inf], 0).fillna(0)
    final_cols = ['time', 'trackId', 'class', 'x', 'y', 'velocity_x', 'velocity_y', 'speed_ms', 'accel_ms2']
    return df[final_cols]


def calculate_all_advanced_features(df, target_actors):
    """
    Calculates relative features, TTC, approach speed, and new preventive features.
    """
    print("  Calculating all advanced features...")
    
    vulnerable_df = df[df['trackId'] == target_actors['vulnerable_id']].copy()
    car_df = df[df['trackId'] == target_actors['car_id']].copy()
    
    merged_df = pd.merge(vulnerable_df, car_df, on='time', suffixes=('_vuln', '_car'), how='outer')
    merged_df = merged_df.sort_values('time').ffill().dropna()

    # --- Standard Relative Features ---
    merged_df['rel_distance'] = np.sqrt(
        (merged_df['x_vuln'] - merged_df['x_car'])**2 + 
        (merged_df['y_vuln'] - merged_df['y_car'])**2
    )
    merged_df['rel_speed'] = np.sqrt(
        (merged_df['velocity_x_vuln'] - merged_df['velocity_x_car'])**2 +
        (merged_df['velocity_y_vuln'] - merged_df['velocity_y_car'])**2
    )

    # --- Approach Speed (gap closing) ---
    delta_x = merged_df['x_car'] - merged_df['x_vuln']
    delta_y = merged_df['y_car'] - merged_df['y_vuln']
    delta_vx = merged_df['velocity_x_car'] - merged_df['velocity_x_vuln']
    delta_vy = merged_df['velocity_y_car'] - merged_df['velocity_y_vuln']

    dot_product_dx_dv = delta_x * delta_vx + delta_y * delta_vy
    merged_df['approach_speed'] = -dot_product_dx_dv / merged_df['rel_distance'].replace(0, np.nan)

    # --- Time-to-Collision (TTC) ---
    dot_product_dv_dv = delta_vx**2 + delta_vy**2
    dot_product_dv_dv_safe = dot_product_dv_dv.replace(0, np.nan)
    ttc = -dot_product_dx_dv / dot_product_dv_dv_safe
    merged_df['ttc'] = ttc.where(ttc > 0, np.nan).fillna(100)

    # --- Preventive / Future Features ---
    merged_df['future_rel_distance'] = merged_df['rel_distance'] - merged_df['rel_speed'] * TIME_HORIZON
    merged_df['future_rel_distance'] = merged_df['future_rel_distance'].clip(lower=0.1)
    merged_df['preventive_risk'] = 1 / merged_df['future_rel_distance']

    # --- Historical Trend / Smoothed Features ---
    window_size = int(FRAME_RATE * 2)  # 2-second window
    merged_df['rel_dist_avg_2s'] = merged_df['rel_distance'].rolling(window=window_size, min_periods=1).mean()
    merged_df['rel_speed_avg_2s'] = merged_df['rel_speed'].rolling(window=window_size, min_periods=1).mean()
    merged_df['future_rel_dist_avg_2s'] = merged_df['future_rel_distance'].rolling(window=window_size, min_periods=1).mean()

    return merged_df


def main():
    if not os.path.exists(TOP_EVENTS_CSV):
        print(f"ERROR: '{TOP_EVENTS_CSV}' not found.")
        return

    top_events = pd.read_csv(TOP_EVENTS_CSV)
    selected_event_info = top_events.iloc[SELECTED_EVENT_RANK]

    print("--- STEP 3 (Final): Ultimate Feature Engineering ---")
    print(f"Selected event file: {selected_event_info['file']}")

    # --- USE THE CORRECT FULL PATH FOR THE CSV ---
    source_csv_path = selected_event_info['full_path']
    clean_df = final_parser_v5(source_csv_path)

    features_df = calculate_motion_features(clean_df)

    target_actors = {
        'vulnerable_id': selected_event_info['trackId_vuln'],
        'car_id': selected_event_info['trackId_car']
    }
    ultimate_features_df = calculate_all_advanced_features(features_df, target_actors)

    base_name = os.path.splitext(os.path.basename(selected_event_info['file']))[0]
    output_filename = f"ultimate_features_{base_name}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ultimate_features_df.to_csv(output_path, index=False)

    print("\n✅ Ultimate Feature Engineering Complete!")
    print(f"Full feature set saved to '{output_path}'")
    print("\nNext step: Re-run your training script (`4d`) on this new 'ultimate' feature file.")


if __name__ == "__main__":
    main()
