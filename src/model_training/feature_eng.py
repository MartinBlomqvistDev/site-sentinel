import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import os

# --- CONFIGURATION ---
INPUT_FILENAME = "tracking_data.csv"
OUTPUT_FILENAME = "features_engineered.csv"

# Safety perimeter distance (d) and NGSIM Frame Rate
SAFETY_DISTANCE_THRESHOLD = 5.0 
NGSIM_FRAME_RATE = 10 
# ---------------------

def get_absolute_path(relative_path_from_root):
    """Helper function to construct absolute path from the project root."""
    # Find the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach the project root (Site_Sentinel/)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Construct the final absolute path
    return os.path.join(project_root, relative_path_from_root)

def calculate_velocity_and_acceleration(df, frame_rate=NGSIM_FRAME_RATE):
    """
    Calculates velocity (speed in m/s) and acceleration (m/s^2) for each object ID,
    using the 10 FPS rate.
    """
    
    # Calculate difference in position (delta_x, delta_y)
    df['delta_x'] = df.groupby('object_id')['x_center'].diff()
    df['delta_y'] = df.groupby('object_id')['y_center'].diff()
    df['delta_t'] = df.groupby('object_id')['frame_id'].diff() / frame_rate
    
    # Calculate Velocity (Speed in m/s)
    df['velocity_x'] = df['delta_x'] / df['delta_t'].replace({0: np.nan})
    df['velocity_y'] = df['delta_y'] / df['delta_t'].replace({0: np.nan})
    
    # Calculate magnitude of velocity (speed_ms)
    df['speed_ms'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    # Calculate Acceleration
    df['accel_ms2'] = df.groupby('object_id')['speed_ms'].diff() / df['delta_t'].replace({0: np.nan})
    
    # Fill NaNs created by diff() for the first frame of each object_id
    df = df.fillna(0)
    
    return df

def calculate_relative_features(df):
    """
    Calculates advanced proximity features (distance, relative speed, TTC) between 
    'Personnel' (Worker) and the nearest 'Hazard Vehicle'.
    """
    
    # Initialize new feature columns
    df['min_dist_to_hazard'] = np.nan
    df['rel_speed_to_hazard'] = np.nan
    df['ttc_estimate'] = np.nan 
    df['risk_zone_status'] = 0

    # Filter by the categories established in T21 (Personnel vs. Hazard)
    personnel = df[df['object_class'] == 'personnel']
    hazards = df[df['object_class'] == 'hazard_vehicle']
    
    if personnel.empty or hazards.empty:
        return df 

    # Loop through each frame where personnel is present
    for frame_id in personnel['frame_id'].unique():
        personnel_in_frame = personnel[personnel['frame_id'] == frame_id]
        hazards_in_frame = hazards[hazards['frame_id'] == frame_id]
        
        if hazards_in_frame.empty:
            continue

        for worker_index, worker in personnel_in_frame.iterrows():
            worker_pos = np.array([worker['x_center'], worker['y_center']])
            
            min_dist = np.inf
            nearest_hazard_speed = 0
            
            # Find the nearest hazard
            for hazard_index, hazard in hazards_in_frame.iterrows():
                hazard_pos = np.array([hazard['x_center'], hazard['y_center']])
                distance = euclidean(worker_pos, hazard_pos)
                
                if distance < min_dist:
                    min_dist = distance
                    # CORRECTED: Use the calculated speed_ms column
                    nearest_hazard_speed = hazard['speed_ms'] 
            
            # --- Calculate Core Features ---
            df.loc[worker_index, 'min_dist_to_hazard'] = min_dist
            
            # Simple relative speed (sum of magnitudes)
            # CORRECTED: Use the calculated speed_ms column
            rel_speed = worker['speed_ms'] + nearest_hazard_speed
            df.loc[worker_index, 'rel_speed_to_hazard'] = rel_speed
            
            # Flag if worker is in the 5.0m risk perimeter
            if min_dist < SAFETY_DISTANCE_THRESHOLD:
                df.loc[worker_index, 'risk_zone_status'] = 1
                
            # --- TTC Calculation ---
            if rel_speed > 0.1: 
                ttc = min_dist / rel_speed
                
                # Filter for realistic prediction times (0-10 seconds)
                if ttc > 0.01 and ttc < 10.0:
                    df.loc[worker_index, 'ttc_estimate'] = ttc
                
    return df

def main():
    """Main function to load data, engineer all features, and save the final CSV for training."""
    
    # --- Resolve ABSOLUTE Input/Output Paths ---
    INPUT_FULL_PATH = get_absolute_path(os.path.join('data', 'logs', INPUT_FILENAME))
    OUTPUT_FULL_PATH = get_absolute_path(os.path.join('data', 'logs', OUTPUT_FILENAME))
    # -------------------------------------------
    
    print(f"Loading raw tracking data from {INPUT_FULL_PATH}...")
    try:
        df = pd.read_csv(INPUT_FULL_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Input CSV not found. Checked path: {INPUT_FULL_PATH}")
        return

    # STEP 1: Calculate basic motion features (speed, acceleration)
    df = calculate_velocity_and_acceleration(df, frame_rate=NGSIM_FRAME_RATE)

    # STEP 2: Calculate relative risk features (Distance, Rel. Speed, TTC)
    df = calculate_relative_features(df)
    
    # Filter only the rows corresponding to personnel and drop rows with non-calculable features (NaN)
    final_df = df[df['object_class'] == 'personnel'].dropna(subset=['min_dist_to_hazard', 'ttc_estimate'])
    
    print(f"Feature engineering complete. Saving {len(final_df)} rows to {OUTPUT_FULL_PATH}")
    
    # Ensure directory exists before saving (Safe check)
    os.makedirs(os.path.dirname(OUTPUT_FULL_PATH), exist_ok=True)
    
    final_df.to_csv(OUTPUT_FULL_PATH, index=False)
    
    print("P31 is complete.")


if __name__ == "__main__":
    main()