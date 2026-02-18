# scripts/2_event_analyzer.py
import pandas as pd
import numpy as np
import os
import time

# --- CONFIGURATION ---
ROOT_DIRECTORY = "C:/DS24/Site_Sentinel/data/raw/CONCOR-D/3W_SidStP_Trajectory"
OUTPUT_CSV_PATH = "data/analysis_results/top_20_predictive_events.csv"
PLAUSIBILITY_THRESHOLD = 0.5 
MINIMUM_EVENT_TIME_S = 30.0 
NUM_TOP_EVENTS = 20
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
                    if len(chunk) == 7: all_rows.append({'trackId': track_id, 'class': object_type, 'x': float(chunk[0]),'y': float(chunk[1]), 'time': float(chunk[5])})
            except (ValueError, IndexError): continue
    return pd.DataFrame(all_rows)

def find_best_predictive_event(df, fps=25):
    df['frame'] = (df['time'] * fps).round().astype(int)
    vulnerable_users = df[df['class'].isin(['Bicycle', 'Pedestrian'])].copy()
    vehicles = df[df['class'] == 'Car'].copy()
    if vulnerable_users.empty or vehicles.empty: return None
    merged_df = pd.merge(vulnerable_users, vehicles, on='frame', suffixes=('_vulnerable', '_car'))
    if merged_df.empty: return None
    merged_df['distance'] = np.sqrt((merged_df['x_vulnerable'] - merged_df['x_car'])**2 + (merged_df['y_vulnerable'] - merged_df['y_car'])**2)
    meaningful_events = merged_df[merged_df['time_vulnerable'] > MINIMUM_EVENT_TIME_S]
    plausible_events = meaningful_events[meaningful_events['distance'] > PLAUSIBILITY_THRESHOLD]
    if plausible_events.empty: return None
    return plausible_events.loc[plausible_events['distance'].idxmin()]

def main():
    if not os.path.isdir(ROOT_DIRECTORY):
        print(f"FATAL ERROR: Root directory not found at {ROOT_DIRECTORY}")
        return
    print(f"Scanning all files to find the best PREDICTIVE events (closest interaction after {MINIMUM_EVENT_TIME_S}s)...")
    target_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(ROOT_DIRECTORY) for f in fn if f.endswith('_ann.csv')]

    print(f"Found {len(target_files)} files to analyze...")
    
    all_candidate_events, start_time = [], time.time()

    for i, filepath in enumerate(target_files):
        filename = os.path.basename(filepath)
        print(f"Processing file {i+1}/{len(target_files)}: {filename}")
        try:
            raw_df = final_parser_v5(filepath)
            if raw_df.empty: continue
            best_event_in_file = find_best_predictive_event(raw_df)
            if best_event_in_file is not None:
                event_data = best_event_in_file.to_dict()
                event_data['file'] = filename
                # --- THIS IS THE CRITICAL FIX ---
                event_data['full_path'] = filepath 
                # --------------------------------
                all_candidate_events.append(event_data)
                print(f"  -> Found candidate event: {event_data['distance']:.2f}m at {event_data['time_vulnerable']:.2f}s.")
        except Exception as e:
            print(f"  -> ERROR processing file: {e}")
            continue

    end_time = time.time()
    
    if not all_candidate_events:
        print("No suitable events were found in any of the files.")
        return

    results_df = pd.DataFrame(all_candidate_events)
    top_events_df = results_df.sort_values(by='distance').head(NUM_TOP_EVENTS)
    
    # --- ADDED 'full_path' TO THE OUTPUT FOR THE NEXT SCRIPT ---
    display_columns = ['distance', 'file', 'frame', 'time_vulnerable', 'trackId_vulnerable', 'class_vulnerable', 'trackId_car', 'full_path']
    top_events_df = top_events_df[display_columns].reset_index(drop=True)

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("="*50)
    print(f"\n🏆 Top {NUM_TOP_EVENTS} Best Predictive Events Found! 🏆")
    print("-" * 50)
    print(top_events_df.to_string())
    print("-" * 50)
    
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    top_events_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Results have been saved to '{OUTPUT_CSV_PATH}'.")

if __name__ == "__main__":
    main()