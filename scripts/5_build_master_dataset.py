import pandas as pd
import numpy as np
import os
import glob

ROOT_DIRECTORY = "C:/DS24/Site_Sentinel/data/raw/CONCOR-D/3W_SidStP_Trajectory"
OUTPUT_DIR = "data/analysis_results"
OUTPUT_MASTER_CSV = "data/analysis_results/master_training_dataset_full.csv"

FRAME_RATE = 29.97
TIME_HORIZON = 4.0  # preventive horizon for full training

# ------------------ PARSING ------------------

def final_parser_v5(filepath):
    all_rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(';', 12)
                if len(parts) < 13:
                    continue
                meta_parts, trajectory_blob = parts[:12], parts[12]
                track_id, object_type = int(meta_parts[0]), meta_parts[1].strip()
                trajectory_points = trajectory_blob.split(';')
                for i in range(0, len(trajectory_points), 7):
                    chunk = trajectory_points[i:i+7]
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

# ------------------ FEATURES ------------------

def calculate_motion_features(df):
    df = df.sort_values(by=['trackId','time']).reset_index(drop=True)
    df['delta_t'] = df.groupby('trackId')['time'].diff()
    df['delta_x'] = df.groupby('trackId')['x'].diff()
    df['delta_y'] = df.groupby('trackId')['y'].diff()
    df['velocity_x'] = (df['delta_x'] / df['delta_t']).replace([np.inf,-np.inf],0).fillna(0)
    df['velocity_y'] = (df['delta_y'] / df['delta_t']).replace([np.inf,-np.inf],0).fillna(0)
    df['speed_ms'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    df['delta_speed'] = df.groupby('trackId')['speed_ms'].diff()
    df['accel_ms2'] = (df['delta_speed'] / df['delta_t']).replace([np.inf,-np.inf],0).fillna(0)
    return df[['time','trackId','class','x','y','velocity_x','velocity_y','speed_ms','accel_ms2']]

def calculate_all_advanced_features(df, target_actors):
    vuln_df = df[df['trackId']==target_actors['vulnerable_id']].copy()
    car_df  = df[df['trackId']==target_actors['car_id']].copy()
    merged_df = pd.merge(vuln_df, car_df, on='time', suffixes=('_vuln','_car'), how='outer')
    merged_df = merged_df.sort_values('time').ffill().dropna()

    # Relative distance & speed
    merged_df['rel_distance'] = np.sqrt((merged_df['x_vuln']-merged_df['x_car'])**2 + (merged_df['y_vuln']-merged_df['y_car'])**2)
    merged_df['rel_speed'] = np.sqrt((merged_df['velocity_x_vuln']-merged_df['velocity_x_car'])**2 + (merged_df['velocity_y_vuln']-merged_df['velocity_y_car'])**2)

    # Approach speed
    delta_x = merged_df['x_car'] - merged_df['x_vuln']
    delta_y = merged_df['y_car'] - merged_df['y_vuln']
    delta_vx = merged_df['velocity_x_car'] - merged_df['velocity_x_vuln']
    delta_vy = merged_df['velocity_y_car'] - merged_df['velocity_y_vuln']
    dot_product_dx_dv = delta_x*delta_vx + delta_y*delta_vy
    merged_df['approach_speed'] = -dot_product_dx_dv / merged_df['rel_distance'].replace(0,np.nan)

    # Time-to-Collision
    dot_product_dv_dv = delta_vx**2 + delta_vy**2
    ttc = -dot_product_dx_dv / dot_product_dv_dv.replace(0,np.nan)
    merged_df['ttc'] = ttc.where(ttc>0, np.nan).fillna(100)

    # Preventive / future features
    merged_df['future_rel_distance'] = merged_df['rel_distance'] - merged_df['rel_speed']*TIME_HORIZON
    merged_df['future_rel_distance'] = merged_df['future_rel_distance'].clip(lower=0.1)
    merged_df['preventive_risk'] = 1 / merged_df['future_rel_distance']

    # Rolling averages
    window_size = int(FRAME_RATE*2)
    merged_df['rel_dist_avg_2s'] = merged_df['rel_distance'].rolling(window=window_size,min_periods=1).mean()
    merged_df['rel_speed_avg_2s'] = merged_df['rel_speed'].rolling(window=window_size,min_periods=1).mean()
    merged_df['future_rel_dist_avg_2s'] = merged_df['future_rel_distance'].rolling(window=window_size,min_periods=1).mean()

    return merged_df

# ------------------ TRACK ID DETECTION ------------------

def detect_target_actors(df):
    """
    Auto-detect vulnerable and car tracks for **full dataset**.
    """
    if 'car' in df['class'].values:
        car_id = df[df['class']=='car']['trackId'].iloc[0]
    else:
        car_id = df['trackId'].unique()[0]
    vuln_candidates = df[df['trackId']!=car_id]['trackId'].unique()
    vulnerable_id = vuln_candidates[0] if len(vuln_candidates)>0 else df['trackId'].unique()[0]
    return {'vulnerable_id': vulnerable_id, 'car_id': car_id}

# ------------------ MAIN ------------------

def main():
    print("--- STEP 5: Building FULL Master Dataset (129 CSVs) ---")
    
    all_csv_files = glob.glob(os.path.join(ROOT_DIRECTORY,"**","*.csv"), recursive=True)
    print(f"Found {len(all_csv_files)} CSVs to process.")

    all_event_dfs = []
    for idx, csv_path in enumerate(all_csv_files):
        print(f"\n--- Processing CSV {idx+1}/{len(all_csv_files)} --- {os.path.basename(csv_path)}")
        try:
            clean_df = final_parser_v5(csv_path)
            motion_df = calculate_motion_features(clean_df)
            target_actors = detect_target_actors(motion_df)
            ultimate_df = calculate_all_advanced_features(motion_df, target_actors)
            all_event_dfs.append(ultimate_df)
        except Exception as e:
            print(f"⚠️ Skipping {os.path.basename(csv_path)} due to error: {e}")

    master_df = pd.concat(all_event_dfs, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_df.to_csv(OUTPUT_MASTER_CSV, index=False)

    print("\n✅ Master Dataset COMPLETE!")
    print(f"Total CSVs combined: {len(all_event_dfs)}")
    print(f"Total data points: {len(master_df)}")
    print(f"Saved to: '{OUTPUT_MASTER_CSV}'")
    print("\n🔥 Ready to train 4d_train_randomforest.py on the FULL dataset!")

if __name__=="__main__":
    main()
