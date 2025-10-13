# scripts/1_data_parser.py

import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# This script is designed to process ONE raw file at a time.
# Update this path to the specific raw CSV you want to clean.
RAW_CSV_PATH = "C:/DS24/Site_Sentinel/data/raw/CONCOR-D/3W_SidStP_Trajectory/20220629_1530/20220629_1530_Sid_StP_3W_d_1_18_ann.csv"

# The output will be a clean, long-format CSV.
OUTPUT_DIR = "data/analysis_results"
# ---------------------

def final_parser_v5(filepath):
    """
    The definitive parser for CONCOR-D CSV files.
    """
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
                            'trackId': track_id, 'class': object_type, 'x': float(chunk[0]),
                            'y': float(chunk[1]), 'time': float(chunk[5])
                        })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(all_rows)

def main():
    """
    Main function to parse a single raw data file and save it in a clean format.
    """
    print("--- STEP 1: Data Parser ---")
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"ERROR: Raw data file not found at '{RAW_CSV_PATH}'. Please check the path.")
        return

    print(f"Parsing raw data from: {os.path.basename(RAW_CSV_PATH)}...")
    clean_df = final_parser_v5(RAW_CSV_PATH)
    
    if clean_df.empty:
        print("ERROR: Parsing resulted in an empty DataFrame. Check the file format.")
        return

    # Create a descriptive name for the clean output file
    base_name = os.path.splitext(os.path.basename(RAW_CSV_PATH))[0]
    output_filename = f"cleaned_{base_name}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    
    print("\n✅ Parsing Complete!")
    print(f"Saved {len(clean_df)} rows of clean data to '{output_path}'")
    print("\nNext step: `2_event_analyzer.py`")

if __name__ == "__main__":
    main()