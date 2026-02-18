import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
DEMO_CSV = r"C:\DS24\Site_Sentinel\data\raw\20190918_1500_Sid_StP_3W_d_1_3_ann_for_demo.csv"
OUTPUT_DIR = r"C:\DS24\Site_Sentinel\data\analysis_results"
FRAME_RATE = 29.97
TIME_HORIZON = 1.5  # seconds ahead for preventive features

TARGET_ACTORS = {
    'vulnerable_id': 1,  # choose based on your demo CSV
    'car_id': 3          # choose based on your demo CSV
}

# --- FUNCTIONS FROM 3_feature_eng.py ---
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
    df = df.sort_values(by=['trackId', 'time']).reset_index(drop=True)
    df['delta_t'] = df.groupby('trackId')['time'].diff()
    df['delta_x'] = df.groupby('trackId')['x'].diff()
    df['delta_y'] = df.groupby('trackId')['y'].diff()
    df['velocity_x'] = (df['delta_x'] / df['delta_t']).replace([np.inf, -np.inf], 0).fillna(0)
    df['velocity_y'] = (df['delta_y'] / df['delta_t']).replace([np.inf, -np.inf], 0).fillna(0)
    df['speed_ms'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    df['delta_speed'] = df.groupby('trackId')['speed_ms'].diff()
    df['accel_ms2'] = (df['delta_speed'] / df['delta_t']).replace([np.inf, -np.inf], 0).fillna(0)
    return df[['time', 'trackId', 'class', 'x', 'y', 'velocity_x', 'velocity_y', 'speed_ms', 'accel_ms2']]


def calculate_all_advanced_features(df, target_actors):
    vulnerable_df = df[df['trackId'] == target_actors['vulnerable_id']].copy()
    car_df = df[df['trackId'] == target_actors['car_id']].copy()
    merged_df = pd.merge(vulnerable_df, car_df, on='time', suffixes=('_vuln', '_car'), how='outer')
    merged_df = merged_df.sort_values('time').ffill().dropna()

    # Relative & preventive features
    merged_df['rel_distance'] = np.sqrt(
        (merged_df['x_vuln'] - merged_df['x_car'])**2 + 
        (merged_df['y_vuln'] - merged_df['y_car'])**2
    )
    merged_df['rel_speed'] = np.sqrt(
        (merged_df['velocity_x_vuln'] - merged_df['velocity_x_car'])**2 +
        (merged_df['velocity_y_vuln'] - merged_df['velocity_y_car'])**2
    )

    delta_x = merged_df['x_car'] - merged_df['x_vuln']
    delta_y = merged_df['y_car'] - merged_df['y_vuln']
    delta_vx = merged_df['velocity_x_car'] - merged_df['velocity_x_vuln']
    delta_vy = merged_df['velocity_y_car'] - merged_df['velocity_y_vuln']
    dot_product_dx_dv = delta_x * delta_vx + delta_y * delta_vy
    merged_df['approach_speed'] = -dot_product_dx_dv / merged_df['rel_distance'].replace(0, np.nan)

    dot_product_dv_dv = delta_vx**2 + delta_vy**2
    ttc = -dot_product_dx_dv / dot_product_dv_dv.replace(0, np.nan)
    merged_df['ttc'] = ttc.where(ttc > 0, np.nan).fillna(100)

    merged_df['future_rel_distance'] = merged_df['rel_distance'] - merged_df['rel_speed'] * TIME_HORIZON
    merged_df['future_rel_distance'] = merged_df['future_rel_distance'].clip(lower=0.1)
    merged_df['preventive_risk'] = 1 / merged_df['future_rel_distance']

    window_size = int(FRAME_RATE * 2)
    merged_df['rel_dist_avg_2s'] = merged_df['rel_distance'].rolling(window=window_size, min_periods=1).mean()
    merged_df['rel_speed_avg_2s'] = merged_df['rel_speed'].rolling(window=window_size, min_periods=1).mean()
    merged_df['future_rel_dist_avg_2s'] = merged_df['future_rel_distance'].rolling(window=window_size, min_periods=1).mean()

    return merged_df


def main():
    clean_df = final_parser_v5(DEMO_CSV)
    features_df = calculate_motion_features(clean_df)
    ultimate_features_df = calculate_all_advanced_features(features_df, TARGET_ACTORS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "ultimate_features_demo.csv")
    ultimate_features_df.to_csv(output_path, index=False)
    print(f"✅ Demo ultimate features saved to '{output_path}'")


if __name__ == "__main__":
    main()
