# explore_and_clean_v4.py

import pandas as pd
import os

# --- CONFIGURATION ---
TARGET_CSV_PATH = "C:/DS24/Site_Sentinel/data/raw/CONCOR-D/3W_SidStP_Trajectory/20190918_0850/20190918_0850_Sid_StP_3W_d_1_1_ann.csv"
# ---------------------

def robust_parser_v4(filepath):
    """
    A more robust parser that reads the CSV line by line and includes
    a guard clause to skip blank or malformed lines.
    """
    print("Running robust parser (v4) for complex CSV structure...")
    
    all_rows = []
    line_number = 0
    
    with open(filepath, 'r') as f:
        header_line = f.readline()
        line_number += 1
        
        # ... (rest of the robust parser logic is the same)
        trajectory_col_name = 'Trajectory(x [m]; y [m]; Speed [km/h]; Tan. Acc. [ms-2]; Lat. Acc. [ms-2]; Time [s]; Angle [rad]; )'
        
        for line in f:
            line_number += 1
            parts = line.strip().split(';')
            
            # --- NEW GUARD CLAUSE ---
            # If the line is blank or doesn't have at least 2 parts (ID and Type), skip it.
            if len(parts) < 2:
                print(f"Skipping malformed or blank line at position: {line_number}")
                continue
            # ------------------------

            try:
                track_id = int(parts[0])
                object_type = parts[1]
                
                # Find where the trajectory data string begins
                trajectory_data_string = ""
                # This logic is complex because the number of metadata columns before the trajectory is not fixed.
                # We find the start of the trajectory data by looking for the first parenthesis.
                for i, part in enumerate(parts):
                    if '(' in part:
                        trajectory_data_string = ";".join(parts[i:])
                        break
                
                if not trajectory_data_string:
                    continue

                # Clean up the trajectory string
                trajectory_data_string = trajectory_data_string.split(')', 1)[0]
                if '(' in trajectory_data_string:
                    trajectory_data_string = trajectory_data_string.split('(', 1)[1]
                
                trajectory_points = trajectory_data_string.strip().split('; ')
                
                for i in range(0, len(trajectory_points), 7):
                    chunk = trajectory_points[i:i + 7]
                    if len(chunk) == 7:
                        all_rows.append({
                            'trackId': track_id, 'class': object_type, 'x': float(chunk[0]),
                            'y': float(chunk[1]), 'speed_kmh': float(chunk[2]),
                            'tan_accel': float(chunk[3]), 'lat_accel': float(chunk[4]),
                            'time': float(chunk[5]), 'angle': float(chunk[6]),
                        })

            except (ValueError, IndexError) as e:
                # This will catch errors on a specific line but allow the script to continue
                print(f"Warning: Could not parse line {line_number}. Error: {e}. Skipping.")
                continue
                        
    return pd.DataFrame(all_rows)

def main():
    if not os.path.exists(TARGET_CSV_PATH):
        print(f"FATAL ERROR: File not found at {TARGET_CSV_PATH}")
        return

    try:
        clean_df = robust_parser_v4(TARGET_CSV_PATH)
        
        if clean_df.empty:
            print("Processing resulted in an empty DataFrame. Please check the input file and parser logic.")
            return

        print("\n✅ Data successfully parsed with robust method!")
        print("-" * 30)

        print("1. First 5 rows of the clean dataset:")
        print(clean_df.head())
        print("-" * 30)

        print("2. Basic statistics:")
        fps = 25
        clean_df['frame'] = (clean_df['time'] * fps).round().astype(int)
        
        num_frames = clean_df['frame'].nunique()
        num_tracks = clean_df['trackId'].nunique()
        print(f"   - Total Frames (calculated @ {fps}fps): {num_frames}")
        print(f"   - Unique Object Tracks: {num_tracks}")
        print("-" * 30)

        print("3. Object classes present in this clip:")
        print(clean_df['class'].value_counts())
        print("-" * 30)
        
        output_path = "data/cleaned_trajectory.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        clean_df.to_csv(output_path, index=False)
        print(f"✅ Clean data has been saved to '{output_path}' for future use!")

    except Exception as e:
        print(f"A critical error occurred during main execution: {e}")

if __name__ == "__main__":
    main()