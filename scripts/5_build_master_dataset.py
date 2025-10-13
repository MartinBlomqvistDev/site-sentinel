# scripts/5_build_master_dataset.py
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION (Same as before) ---
TOP_EVENTS_CSV = "data/analysis_results/top_20_predictive_events.csv"
OUTPUT_DIR = "data/analysis_results"
OUTPUT_MASTER_CSV = "data/analysis_results/master_training_dataset.csv"
NUM_EVENTS_TO_COMBINE = 5
# ---------------------

# (All helper functions are the same as before)
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
                    if len(chunk) == 7: all_rows.append({'trackId': track_id, 'class': object_type, 'x': float(chunk[0]),'y': float(chunk[1]), 'time': float(chunk[5])})
            except (ValueError, IndexError): continue
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

def calculate_advanced_features(df, target_actors):
    print("  Calculating advanced features...")
    vulnerable_df = df[df['trackId'] == target_actors['vulnerable_id']].copy()
    car_df = df[df['trackId'] == target_actors['car_id']].copy()
    merged_df = pd.merge(vulnerable_df, car_df, on='time', suffixes=('_vuln', '_car'), how='outer')
    merged_df = merged_df.sort_values('time').ffill().dropna()
    merged_df['rel_distance'] = np.sqrt((merged_df['x_vuln'] - merged_df['x_car'])**2 + (merged_df['y_vuln'] - merged_df['y_car'])**2)
    merged_df['rel_speed'] = np.sqrt((merged_df['velocity_x_vuln'] - merged_df['velocity_x_car'])**2 + (merged_df['velocity_y_vuln'] - merged_df['velocity_y_car'])**2)
    delta_x = merged_df['x_car'] - merged_df['x_vuln']
    delta_y = merged_df['y_car'] - merged_df['y_vuln']
    delta_vx = merged_df['velocity_x_car'] - merged_df['velocity_x_vuln']
    delta_vy = merged_df['velocity_y_car'] - merged_df['velocity_y_vuln']
    dot_product_dv_dv = delta_vx**2 + delta_vy**2
    dot_product_dx_dv = delta_x * delta_vx + delta_y * delta_vy
    dot_product_dv_dv = dot_product_dv_dv.replace(0, np.nan)
    ttc = -dot_product_dx_dv / dot_product_dv_dv
    merged_df['ttc'] = ttc.where(ttc > 0, np.nan).fillna(100)
    return merged_df

def main():
    """
    Builds a master training dataset by processing and combining the top N events.
    """
    print("--- STEP 5: Building Master Training Dataset ---")
    if not os.path.exists(TOP_EVENTS_CSV):
        print(f"ERROR: '{TOP_EVENTS_CSV}' not found.")
        return
    top_events = pd.read_csv(TOP_EVENTS_CSV).head(NUM_EVENTS_TO_COMBINE)
    print(f"Will process and combine the top {NUM_EVENTS_TO_COMBINE} events.")
    all_event_dfs = []
    for index, event_info in top_events.iterrows():
        print(f"\n--- Processing Event {index + 1}/{NUM_EVENTS_TO_COMBINE} ---")
        
        # --- CORRECTED LOGIC ---
        source_csv_path = event_info['full_path']
        print(f"File: {os.path.basename(source_csv_path)}")
        # -----------------------

        if not os.path.exists(source_csv_path):
            print(f"  -> FATAL ERROR: File not found at path: {source_csv_path}")
            continue
            
        print("  1. Parsing raw data...")
        clean_df = final_parser_v5(source_csv_path)
        features_df = calculate_motion_features(clean_df)
        
        target_actors = {'vulnerable_id': event_info['trackId_vulnerable'], 'car_id': event_info['trackId_car']}
        advanced_features_df = calculate_advanced_features(features_df, target_actors)
        all_event_dfs.append(advanced_features_df)

    master_df = pd.concat(all_event_dfs, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_df.to_csv(OUTPUT_MASTER_CSV, index=False)
    
    print("\n" + "="*50)
    print("✅ Master Dataset Creation Complete!")
    print(f"Combined {len(all_event_dfs)} events into one file.")
    print(f"Total data points: {len(master_df)}")
    print(f"Master dataset saved to: '{OUTPUT_MASTER_CSV}'")
    print("\nNext step: Re-run the training script on this new master file.")

if __name__ == "__main__":
    main()