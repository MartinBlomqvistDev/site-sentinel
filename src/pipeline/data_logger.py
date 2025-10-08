import pandas as pd
import os
import numpy as np
import time

# --- CONFIGURATION (NGSIM Data Processor) ---
NGSIM_FILENAME = "ngsim_trajectories.csv"
OUTPUT_CSV_PATH = "../data/logs/tracking_data.csv" 

# NGSIM TECHNICAL SPECS (CRITICAL for P31)
NGSIM_FRAME_RATE = 10 # 10 Hz reporting frequency
FEET_TO_METERS = 0.3048 # Conversion factor for 5.0m threshold

NGSIM_COLS_TO_LOAD = [
    'Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Class'        
]
# ---------------------

def map_class_to_category(ngsim_v_class):
    """
    Maps NGSIM's Vehicle_Class to 'personnel' or 'hazard_vehicle'.
    Assuming 1-4 are vehicles, 0, 5, 6 are VRUs/Pedestrians (for Site Sentinel safety targets).
    """
    if ngsim_v_class in [1, 2, 3, 4]: 
        return 'hazard_vehicle'
    elif ngsim_v_class in [0, 5, 6]: 
        return 'personnel'
    else:
        return 'hazard_vehicle' 

def get_absolute_path(relative_path):
    """Helper function to construct absolute path from the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return os.path.join(project_root, relative_path)


def inject_synthetic_worker(df, start_x=10.0, start_y=50.0):
    """
    Creates a synthetic 'personnel' object (Worker) with dynamic, irregular movement
    across all frames of the dataset. 
    """
    
    # Get the total number of frames in the dataset
    total_frames = df['frame_id'].max()
    
    # Assign a unique, high ID to the worker
    worker_id = df['object_id'].max() + 1 
    
    worker_data = []
    current_x = start_x
    current_y = start_y
    
    # --- Define Movement Characteristics ---
    # The worker will walk around this central point:
    CENTER_X = start_x 
    CENTER_Y = start_y
    # Max displacement from center
    MAX_DISPLACEMENT = 3.0 * FEET_TO_METERS # approx 0.9m displacement
    
    for frame_id in range(1, total_frames + 1):
        # 1. Apply small, random movement (irregular times)
        # Randomly adjust position, but keep movement small to simulate slow walking/pausing
        
        # Adjust position slightly (random step size between -0.05m and 0.05m)
        step_x = (np.random.rand() * 0.1 - 0.05) * FEET_TO_METERS
        step_y = (np.random.rand() * 0.1 - 0.05) * FEET_TO_METERS
        
        current_x += step_x
        current_y += step_y
        
        # 2. Keep the worker within the simulated work area boundary
        # If the worker moves too far, pull them back towards the center
        if abs(current_x - CENTER_X) > MAX_DISPLACEMENT:
            current_x = CENTER_X + (MAX_DISPLACEMENT * np.sign(CENTER_X - current_x))
            
        if abs(current_y - CENTER_Y) > MAX_DISPLACEMENT:
            current_y = CENTER_Y + (MAX_DISPLACEMENT * np.sign(CENTER_Y - current_y))

        # 3. Log the data point
        worker_data.append({
            'object_id': worker_id,
            'frame_id': frame_id,
            'object_class': 'personnel',
            'x_center': current_x,
            'y_center': current_y
        })
        
    # Convert worker data to DataFrame
    df_worker = pd.DataFrame(worker_data)
    
    # Ensure columns match the existing df (for concatenation)
    df_worker = df_worker.reindex(columns=df.columns) 
    
    # Combine original data with synthetic worker data
    df = pd.concat([df, df_worker], ignore_index=True)
    
    print(f"Injected 1 synthetic worker across {total_frames} frames.")
    return df


def run_data_processor():
    """Reads NGSIM data, cleans it, sorts, converts units, and saves the final tracking CSV."""
    
    # --- Resolve Input and Output Paths ---
    # INPUT: Absolute path from the root (ensures FileNotFoundError is fixed)
    input_ngsim_full_path = os.path.join(get_absolute_path('data'), 'raw', NGSIM_FILENAME)
    
    # OUTPUT: Absolute path to the logs folder (solves the final OSError/non-existent directory)
    logs_dir = os.path.join(get_absolute_path('data'), 'logs')
    output_full_path = os.path.join(logs_dir, os.path.basename(OUTPUT_CSV_PATH))
    # ------------------------------------
    
    print(f"Loading raw NGSIM trajectory data from {input_ngsim_full_path}...")
    
    try:
        # Load data using standard CSV read
        df = pd.read_csv(input_ngsim_full_path, usecols=NGSIM_COLS_TO_LOAD)
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file not found. Checked path: {input_ngsim_full_path}")
        return None
    except Exception as e:
        print(f"Error reading file. Details: {e}")
        return None

    # Step 1: Rename Columns
    df = df.rename(columns={
        'Vehicle_ID': 'object_id', 'Frame_ID': 'frame_id', 'Local_X': 'x_center',
        'Local_Y': 'y_center', 'v_Class': 'v_class'
    })
    
    # Step 2: Critical Sorting (Ensures correct velocity calculation in P31)
    print("Sorting data by object_id and frame_id (time)...")
    df = df.sort_values(by=['object_id', 'frame_id']).reset_index(drop=True)
    
    # Step 3: Unit Conversion (Feet to Meters)
    print("Converting coordinates from US Survey Feet to Meters...")
    df['x_center'] = df['x_center'] * FEET_TO_METERS
    df['y_center'] = df['y_center'] * FEET_TO_METERS

    # Step 4: Map Classes
    df['object_class'] = df['v_class'].apply(map_class_to_category)

    # Step 5: Inject Synthetic Worker Data (NEW CRITICAL STEP)
    # The worker is placed near the road: You must manually check the raw NGSIM coordinates
    # to find a relevant starting point near the main traffic flow.
    # Placeholder values for X/Y are used here:
    df = inject_synthetic_worker(df, start_x=10.0, start_y=50.0) 
    
    # Step 6: Final Filtering (Includes the synthetic worker data)
    final_df = df[df['object_class'].isin(['personnel', 'hazard_vehicle'])].copy()
    
    # Step 7: Critical Sorting (Ensures the new worker data is sorted correctly)
    print("Sorting data by object_id and frame_id (time)...")
    df = df.sort_values(by=['object_id', 'frame_id']).reset_index(drop=True)
    
    # --- Save the Cleaned Data ---
    # Ensure the output directory exists before saving (Solves the OSError)
    os.makedirs(logs_dir, exist_ok=True) 
    
    final_df[['frame_id', 'object_id', 'object_class', 'x_center', 'y_center']].to_csv(output_full_path, index=False)
    
    print("\n--- T21 Data Logger Complete (NGSIM Processor) ---")
    print(f"Total entries logged: {len(final_df)}")
    print(f"CSV saved to: {output_full_path}")
    
    return final_df

if __name__ == "__main__":
    run_data_processor()