import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# --- CONFIGURATION (UPDATE PATHS IF NECESSARY) ---
# NOTE: This path assumes the raw tracking data CSV from T21 is here:
INPUT_CSV_PATH = "../data/logs/tracking_data.csv"
OUTPUT_CSV_PATH = "../data/logs/features_engineered.csv"
# Define the size of a safety buffer zone (in arbitrary units, e.g., meters)
SAFETY_DISTANCE_THRESHOLD = 5.0 
# --------------------------------------------------

def calculate_velocity_and_acceleration(df, frame_rate=30):
    """Calculates velocity (speed) and acceleration for each object ID."""
    
    # Calculate difference in position (delta_x, delta_y)
    df['delta_x'] = df.groupby('object_id')['x_center'].diff()
    df['delta_y'] = df.groupby('object_id')['y_center'].diff()
    df['delta_t'] = df.groupby('object_id')['frame_id'].diff() / frame_rate
    
    # Calculate Velocity (Speed) in distance units per second (px/s or meters/s)
    df['velocity_x'] = df['delta_x'] / df['delta_t'].replace({0: np.nan})
    df['velocity_y'] = df['delta_y'] / df['delta_t'].replace({0: np.nan})
    
    # Calculate magnitude of velocity (speed)
    df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    # Calculate Acceleration
    df['accel'] = df.groupby('object_id')['speed'].diff() / df['delta_t'].replace({0: np.nan})
    
    # Fill NaNs created by diff() for the first frame of each object_id
    df = df.fillna(0)
    
    return df

def calculate_relative_features(df):
    """
    Calculates proximity features (distance, relative speed) between 
    a 'Worker' and the nearest 'Vehicle/Machinery'.
    """
    
    df['min_dist_to_hazard'] = np.nan
    df['rel_speed_to_hazard'] = np.nan
    df['risk_zone_status'] = 0

    # Ensure coordinates and speeds are numeric (safety check)
    cols_to_check = ['x_center', 'y_center', 'speed', 'object_class']
    if not all(col in df.columns for col in cols_to_check):
        print("Error: Missing required columns for relative calculations.")
        return df

    # Separate workers (pedestrians) from potential hazards
    workers = df[df['object_class'].isin(['person', 'worker'])]
    hazards = df[df['object_class'].isin(['vehicle', 'truck', 'machinery'])]
    
    if workers.empty or hazards.empty:
        return df # Cannot calculate relative features

    # Loop through each frame where a worker is present
    for frame_id in workers['frame_id'].unique():
        workers_in_frame = workers[workers['frame_id'] == frame_id]
        hazards_in_frame = hazards[hazards['frame_id'] == frame_id]
        
        if hazards_in_frame.empty:
            continue

        for worker_index, worker in workers_in_frame.iterrows():
            worker_pos = np.array([worker['x_center'], worker['y_center']])
            worker_speed = worker['speed']

            min_dist = np.inf
            nearest_hazard_speed = 0

            # Find the nearest hazard to the current worker
            for hazard_index, hazard in hazards_in_frame.iterrows():
                hazard_pos = np.array([hazard['x_center'], hazard['y_center']])
                
                # Calculate distance
                distance = euclidean(worker_pos, hazard_pos)
                
                if distance < min_dist:
                    min_dist = distance
                    nearest_hazard_speed = hazard['speed']
                    
            # Update the DataFrame for the worker (using the original index)
            df.loc[worker_index, 'min_dist_to_hazard'] = min_dist
            
            # Calculate simple relative speed (sum of magnitudes)
            df.loc[worker_index, 'rel_speed_to_hazard'] = worker_speed + nearest_hazard_speed
            
            # Flag if worker is in a high-risk zone
            if min_dist < SAFETY_DISTANCE_THRESHOLD:
                df.loc[worker_index, 'risk_zone_status'] = 1
                
    return df

def main():
    """Main function to load, engineer, and save features."""
    print(f"Loading raw tracking data from {INPUT_CSV_PATH}...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print("ERROR: Input CSV not found. Ensure T21: Data Logger ran successfully.")
        return

    # STEP 1: Calculate basic motion features (speed, acceleration)
    df = calculate_velocity_and_acceleration(df)

    # STEP 2: Calculate relative risk features
    df = calculate_relative_features(df)
    
    # Optional: Implement Time-to-Collision (TTC) logic here (requires advanced vector math)
    # df = calculate_ttc(df) 

    # Filter only the rows where features are relevant (i.e., workers near hazards)
    final_df = df[df['object_class'].isin(['person', 'worker'])].dropna(subset=['min_dist_to_hazard'])

    print(f"Feature engineering complete. Saving {len(final_df)} rows to {OUTPUT_CSV_PATH}")
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("P31 is complete.")


if __name__ == "__main__":
    main()