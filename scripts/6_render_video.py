# scripts/6_render_video.py
#
# FINAL VERSION 5:
# - Uses the CORRECT parser for the complex DFS Viewer CSV format.
# - Loads and uses 'transform_params.json'.
# - Uses the hardcoded TARGET_PERSON_ID and TARGET_CAR_ID.
# - Transforms coordinates correctly before homography projection.
# - Includes debug print for frame matching (commented out by default).
# - Fixes merge_asof tolerance error.
# - Fixes ValueError for array truthiness.
# - Lowers threshold for drawing predictive lines to 0.5 pixels.
# - COLOR-CODES predictive lines for target objects based on risk.
#

import pandas as pd
import numpy as np
import os
import joblib
import cv2
import json
from sklearn.linear_model import LinearRegression # Needed by parser/transform

# --- CONFIGURATION ---
RAW_TRAJECTORY_CSV = "data/full_trajectories_PIXELS.csv" # Or your UTM file, parser uses both cols
MODEL_PATH = "models/rf_master_predictor_dual_lead_tuned.pkl"
VIDEO_PATH = r"C:\DS24\Site_Sentinel\data\raw\20190918_1500_Sid_StP_3W_d_1_3_cal.mp4"
OUTPUT_VIDEO_PATH = "data/analysis_results/final_demo_polished_color_lines.mp4" # New name reflecting change
HOMOGRAPHY_PATH = "data/analysis_results/homography_matrix.npy"
PARAMS_PATH = "data/analysis_results/transform_params.json"

# Main event and window
MAIN_EVENT_START_TIME = 4*60 + 7  # 04:07 in seconds
WINDOW_BEFORE = 40  # seconds before
WINDOW_AFTER = 30   # seconds after

# Target IDs
TARGET_PERSON_ID = 114
TARGET_CAR_ID = 192

TIME_HORIZON = 4.0  # seconds ahead for preventive risk
FRAME_RATE = 29.97 # Default, script tries to read from video

# --- Robust CSV PARSING ---
NUM_FIXED_COLS = 12
NUM_TRAJ_COLS_PER_STEP = 9
IDX_UTM_X = 0; IDX_UTM_Y = 1; IDX_TIME = 5; IDX_PIXEL_X = 7; IDX_PIXEL_Y = 8
CSV_METADATA_LINES = 80 # Adjust if needed for your specific export

def parse_complex_dfs_csv(filepath):
    """
    Parses the complex CSV format from DFS Viewer.
    Returns a DataFrame with one row per time step per track.
    """
    all_points = []
    skipped_rows = 0
    parsed_lines = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            print(f"Skipping {CSV_METADATA_LINES} metadata lines...")
            for i in range(CSV_METADATA_LINES):
                line = f.readline();
                if not line: print(f"❌ ERROR: File ended unexpectedly at metadata line {i+1}."); return pd.DataFrame()
            header_line = f.readline()
            if not header_line: print("❌ ERROR: Could not read header line."); return pd.DataFrame()
            header_line = header_line.strip(); print("Header line read.")
            for line_num, line in enumerate(f, start=CSV_METADATA_LINES + 2):
                line = line.strip();
                if not line: continue
                parts = line.split(';'); parsed_lines += 1
                try:
                    track_id = int(parts[0])
                    obj_type = parts[1].strip() if len(parts) > 1 else "Unknown"
                    traj_data = parts[NUM_FIXED_COLS:]
                    for i in range(0, len(traj_data), NUM_TRAJ_COLS_PER_STEP):
                        chunk = traj_data[i : i + NUM_TRAJ_COLS_PER_STEP]
                        if len(chunk) == NUM_TRAJ_COLS_PER_STEP:
                            try:
                                utm_x = float(chunk[IDX_UTM_X]); utm_y = float(chunk[IDX_UTM_Y])
                                time = float(chunk[IDX_TIME])
                                pixel_x = float(chunk[IDX_PIXEL_X]); pixel_y = float(chunk[IDX_PIXEL_Y])
                                all_points.append({
                                    'trackId': track_id, 'class': obj_type, 'time': time,
                                    'x': utm_x, 'y': utm_y, 'pixel_x': pixel_x, 'pixel_y': pixel_y
                                })
                            except (ValueError, IndexError): continue # Skip bad chunk
                except (ValueError, IndexError): skipped_rows += 1; continue # Skip bad line
        if skipped_rows > 0: print(f"⚠️ Skipped {skipped_rows} malformed lines.")
    except FileNotFoundError: print(f"❌ ERROR: File not found: {filepath}"); return pd.DataFrame()
    except Exception as e: print(f"❌ ERROR parsing {filepath}: {e}"); return pd.DataFrame()
    if not all_points: print("❌ ERROR: No data points parsed."); return pd.DataFrame()
    df = pd.DataFrame(all_points)
    # Ensure correct dtypes
    for col in ['trackId', 'time', 'x', 'y', 'pixel_x', 'pixel_y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['trackId', 'time', 'x', 'y', 'pixel_x', 'pixel_y'])
    df['trackId'] = df['trackId'].astype(int)
    return df

# --- MOTION FEATURES ---
def calculate_motion_features(df):
    if df.empty: return df
    df = df.sort_values(by=['trackId','time']).reset_index(drop=True)
    df['delta_t'] = df.groupby('trackId')['time'].diff()
    df['delta_x'] = df.groupby('trackId')['x'].diff()
    df['delta_y'] = df.groupby('trackId')['y'].diff()
    # Calculate velocity safely
    df['velocity_x'] = (df['delta_x'] / df['delta_t'].replace(0, np.nan)).replace([np.inf,-np.inf],0).fillna(0)
    df['velocity_y'] = (df['delta_y'] / df['delta_t'].replace(0, np.nan)).replace([np.inf,-np.inf],0).fillna(0)
    df['speed_ms'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    # Calculate acceleration safely
    df['delta_speed'] = df.groupby('trackId')['speed_ms'].diff()
    df['accel_ms2'] = (df['delta_speed'] / df['delta_t'].replace(0, np.nan)).replace([np.inf,-np.inf],0).fillna(0)
    # Return all necessary columns including original coordinates and velocities
    return df[['time','trackId','class','x','y','velocity_x','velocity_y','speed_ms','accel_ms2', 'pixel_x', 'pixel_y']] # Keep pixels if needed elsewhere

# --- ADVANCED FEATURES ---
def calculate_all_advanced_features(df, target_actors, frame_rate_local): # Pass frame_rate
    if df.empty: return pd.DataFrame()
    vuln_df = df[df['trackId']==target_actors['vulnerable_id']].copy()
    car_df  = df[df['trackId']==target_actors['car_id']].copy()
    if vuln_df.empty or car_df.empty:
        print(f"❌ WARNING: Data missing for targets (Vuln: {target_actors['vulnerable_id']}, Car: {target_actors['car_id']}).")
        return pd.DataFrame()
    vuln_df = vuln_df.sort_values('time'); car_df = car_df.sort_values('time')
    time_tolerance = 1.0 / frame_rate_local
    # Ensure merge columns exist and handle potential missing velocities/speeds/accels
    req_cols = ['time', 'x', 'y', 'velocity_x', 'velocity_y', 'speed_ms', 'accel_ms2']
    vuln_df = vuln_df.dropna(subset=req_cols)
    car_df = car_df.dropna(subset=req_cols)
    if vuln_df.empty or car_df.empty:
         print(f"❌ WARNING: Data missing required columns (speed/velocity/accel) for targets after dropna.")
         return pd.DataFrame()

    merged_df = pd.merge_asof(vuln_df, car_df, on='time', suffixes=('_vuln','_car'),
                              tolerance=time_tolerance, direction='nearest')
    # Drop rows where the merge failed to find a match within tolerance
    merged_df = merged_df.dropna(subset=['trackId_vuln', 'trackId_car']).reset_index(drop=True)

    if merged_df.empty: print("❌ WARNING: No overlapping data for targets after merge_asof."); return pd.DataFrame()

    # Calculations... (Ensure required columns from merge exist)
    merged_df['rel_distance'] = np.sqrt((merged_df['x_vuln']-merged_df['x_car'])**2 + (merged_df['y_vuln']-merged_df['y_car'])**2)
    merged_df['rel_speed'] = np.sqrt((merged_df['velocity_x_vuln']-merged_df['velocity_x_car'])**2 + (merged_df['velocity_y_vuln']-merged_df['velocity_y_car'])**2)
    delta_x = merged_df['x_car'] - merged_df['x_vuln']; delta_y = merged_df['y_car'] - merged_df['y_vuln']
    delta_vx = merged_df['velocity_x_car'] - merged_df['velocity_x_vuln']; delta_vy = merged_df['velocity_y_car'] - merged_df['velocity_y_vuln']
    dot_product_dx_dv = delta_x*delta_vx + delta_y*delta_vy
    merged_df['approach_speed'] = -dot_product_dx_dv / merged_df['rel_distance'].replace(0,np.nan)
    dot_product_dv_dv = delta_vx**2 + delta_vy**2
    ttc = -dot_product_dx_dv / dot_product_dv_dv.replace(0,np.nan)
    merged_df['ttc'] = ttc.where(ttc>0, np.nan).fillna(100)
    merged_df['future_rel_distance'] = (merged_df['rel_distance'] - merged_df['rel_speed']*TIME_HORIZON).clip(lower=0.1)
    merged_df['preventive_risk'] = 1 / merged_df['future_rel_distance']
    window_size = max(1, int(frame_rate_local*2))
    merged_df['rel_dist_avg_2s'] = merged_df['rel_distance'].rolling(window=window_size,min_periods=1).mean()
    merged_df['rel_speed_avg_2s'] = merged_df['rel_speed'].rolling(window=window_size,min_periods=1).mean()
    merged_df['future_rel_dist_avg_2s'] = merged_df['future_rel_distance'].rolling(window=window_size,min_periods=1).mean()
    # Add required columns for model
    merged_df['speed_ms_vuln'] = merged_df['speed_ms_vuln']
    merged_df['speed_ms_car'] = merged_df['speed_ms_car']
    merged_df['accel_ms2_vuln'] = merged_df['accel_ms2_vuln']
    merged_df['accel_ms2_car'] = merged_df['accel_ms2_car']
    return merged_df

# --- TRANSFORM FUNCTION ---
def apply_dynamic_transform(df, params):
    df = df.copy();
    if df.empty or 'x' not in df.columns or 'y' not in df.columns: return df # Check required cols
    x_mean, y_mean = params['x_mean'], params['y_mean']
    y_centered_max, theta = params['y_centered_max'], params['theta']
    df['x_centered'] = df['x'] - x_mean; df['y_centered'] = df['y'] - y_mean
    df['y_centered_inv'] = y_centered_max - df['y_centered']
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    df['x_transformed'] = df['x_centered'] * cos_t - df['y_centered_inv'] * sin_t
    df['y_transformed'] = df['x_centered'] * sin_t + df['y_centered_inv'] * cos_t
    return df

# --- VIDEO RENDERING UTILS ---
def rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=10):
    x1, y1 = int(pt1[0]), int(pt1[1]); x2, y2 = int(pt2[0]), int(pt2[1]) # Ensure integer coords
    if x1 >= x2 or y1 >= y2: return
    radius = max(1, min(radius, int((x2-x1)/2), int((y2-y1)/2)))
    # Draw lines and arcs
    cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180,0,90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270,0,90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius), 90,0,90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius), 0,0,90, color, thickness)

# --------------------------- MAIN FUNCTION ---------------------------
def main():
    print("--- STEP 6: Final Video Rendering (Robust Parser v5 - Color Lines) ---")

    # --- Load trajectory CSV ---
    all_objects_df = parse_complex_dfs_csv(RAW_TRAJECTORY_CSV)
    if all_objects_df.empty: print(f"❌ ERROR: No data from {RAW_TRAJECTORY_CSV}."); return
    print(f"✅ Parsed {len(all_objects_df)} pts from {all_objects_df['trackId'].nunique()} tracks.")

    # --- Calculate motion features ---
    motion_df = calculate_motion_features(all_objects_df)

    # --- Video Setup & FPS ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print(f"❌ ERROR: Cannot open video {VIDEO_PATH}"); return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = FRAME_RATE; print(f"⚠️ WARNING: Using default FPS {fps}.")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    # --- Define Target Actors ---
    target_actors = {'vulnerable_id': TARGET_PERSON_ID, 'car_id': TARGET_CAR_ID}
    print(f"✅ Targets: Person={TARGET_PERSON_ID}, Car={TARGET_CAR_ID}")

    # --- Calculate Interaction Features ---
    interaction_df = calculate_all_advanced_features(motion_df, target_actors, fps)
    if interaction_df.empty: print("⚠️ WARNING: No interaction features calculated.")

    # --- Load Model, Homography, Params ---
    try:
        model_dict = joblib.load(MODEL_PATH)
        if 'preventive' in model_dict: model = model_dict['preventive']
        elif 'prevention' in model_dict: model = model_dict['prevention']
        else: first_key = next(iter(model_dict)); model = model_dict[first_key]; print(f"⚠️ WARN: Using model key '{first_key}'.")
        print("✅ Model loaded.")
        expected_features_raw = getattr(model, 'feature_names_in_', None)
        expected_features = list(expected_features_raw) if expected_features_raw is not None else None
    except Exception as e: print(f"❌ ERROR loading model: {e}"); return
    try: H = np.load(HOMOGRAPHY_PATH); print("✅ Homography loaded.")
    except Exception as e: print(f"❌ ERROR loading homography: {e}"); return
    try:
        with open(PARAMS_PATH, 'r') as f: params = json.load(f)
        print("✅ Params loaded.")
    except Exception as e: print(f"❌ ERROR loading params: {e}"); return

    # --- Apply dynamic transformation ---
    all_objects_df_transformed = apply_dynamic_transform(motion_df, params)
    print("✅ Dynamic transform applied.")

    # --- Prepare data for rendering loop ---
    all_objects_df_transformed['frame'] = (all_objects_df_transformed['time'] * fps).round().astype(int)
    if not interaction_df.empty: interaction_df['frame'] = (interaction_df['time'] * fps).round().astype(int)
    all_objects_df_transformed = all_objects_df_transformed.dropna(subset=['frame'])
    objects_by_frame = {f: g.to_dict('records') for f, g in all_objects_df_transformed.groupby('frame')}
    interaction_by_frame = {}
    if not interaction_df.empty:
        interaction_df = interaction_df.dropna(subset=['frame'])
        interaction_by_frame = {f: g.iloc[0].to_dict() for f, g in interaction_df.groupby('frame')}

    start_frame = max(0, int((MAIN_EVENT_START_TIME - WINDOW_BEFORE)*fps))
    end_frame = int((MAIN_EVENT_START_TIME + WINDOW_AFTER)*fps)
    total_frames_to_render = end_frame - start_frame + 1
    print(f"Rendering frames {start_frame} to {end_frame}...")

    # --- Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened(): print(f"❌ ERROR opening VideoWriter."); cap.release(); return

    # --- Define features for prediction ---
    base_features = [
        'rel_distance', 'rel_speed', 'speed_ms_vuln', 'speed_ms_car',
        'accel_ms2_vuln', 'accel_ms2_car', 'ttc', 'approach_speed',
        'rel_dist_avg_2s', 'rel_speed_avg_2s', 'future_rel_dist_avg_2s'
    ]
    available_features = [f for f in base_features if f in interaction_df.columns]
    features_to_predict = expected_features if expected_features is not None else available_features
    if expected_features is not None and set(available_features) != set(expected_features):
         print(f"⚠️ WARNING: Feature mismatch. Model expects {expected_features}. Data has {available_features}.")
         features_to_predict = [f for f in expected_features if f in available_features]
         if not features_to_predict: print("❌ ERROR: No common features."); features_to_predict = []
    print(f"Using features for prediction: {features_to_predict}")

    # ---=== VIDEO RENDERING LOOP ===---
    frame_id = -1
    processed_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: print("End of video stream."); break
        frame_id += 1
        if frame_id < start_frame: continue
        if frame_id > end_frame: print("Reached end frame."); break

        # --- Risk Prediction ---
        interaction_data = interaction_by_frame.get(frame_id)
        risk_prob, proximity_risk, preventive_risk = 0.0, 0.0, 0.0
        rel_distance, rel_speed = None, None
        if interaction_data and features_to_predict: # Check list not empty
            try:
                current_features_dict = {f: interaction_data.get(f, 0) for f in features_to_predict}
                current_features_df = pd.DataFrame([current_features_dict])[features_to_predict]
                if expected_features and set(features_to_predict) != set(expected_features):
                     current_features_df = current_features_df.reindex(columns=expected_features, fill_value=0)
                if current_features_df.isnull().values.any():
                    print(f"⚠️ NaN found in features frame {frame_id}, skipping prediction.")
                    risk_prob = 0.0
                else:
                    risk_prob = model.predict_proba(current_features_df)[0][1]
                rel_distance = interaction_data.get('rel_distance')
                rel_speed = interaction_data.get('rel_speed')
                if rel_distance is not None and np.isfinite(rel_distance) and rel_distance > 0.1:
                    proximity_risk = 1 / (rel_distance + 0.1)
                    if rel_speed is not None and np.isfinite(rel_speed):
                        future_rel_distance = max(0.1, rel_distance - (rel_speed * TIME_HORIZON))
                        preventive_risk = 1 / (future_rel_distance + 0.1)
                    else: preventive_risk = proximity_risk
                else: proximity_risk = 0.0; preventive_risk = 0.0;
            except Exception as e: print(f"⚠️ Error prediction frame {frame_id}: {e}"); risk_prob=0.0; proximity_risk=0.0; preventive_risk=0.0;
        boosted_risk = min(1.0, max(0.0, (risk_prob*0.6) + (proximity_risk*0.2) + (preventive_risk*0.2)))

        # --- Draw Overlay ---
        overlay_img = frame.copy()
        cv2.rectangle(overlay_img, (0,0), (frame_width,100), (0,0,0), -1)
        frame = cv2.addWeighted(overlay_img, 0.6, frame, 0.4, 0)
        bar_width = int(frame_width * boosted_risk)
        risk_color = (0,255,0); # Default Green
        if boosted_risk > 0.8: risk_color = (0,0,255) # Red
        elif boosted_risk > 0.6: risk_color = (0,165,255) # Orange
        cv2.rectangle(frame, (0,90), (bar_width,100), risk_color, -1)
        status_text = (f"NEAR MISS / IMMINENT Risk: {boosted_risk:.0%}" if boosted_risk > 0.8 else
                       f"Approaching Risk: {boosted_risk:.0%}" if boosted_risk > 0.6 else
                       f"SAFE (Risk: {boosted_risk:.0%})")
        current_time_sec = frame_id / fps; time_text = f"{int(current_time_sec // 60):02d}:{current_time_sec % 60:05.2f}"
        cv2.putText(frame, f"Frame: {frame_id} Time: {time_text}", (frame_width - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, status_text, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, risk_color, 3)

        # --- Draw Objects ---
        all_frame_objects = objects_by_frame.get(frame_id, [])

        for row in all_frame_objects: # row is now a dictionary
            try:
                obj_id = int(row['trackId'])
                # Default style
                box_color, box_size, line_thick, is_target = (255,180,0), 15, 1, False
                # Highlight targets
                if obj_id == TARGET_PERSON_ID: box_color, box_size, line_thick, is_target = (0, 255, 0), 25, 2, True
                elif obj_id == TARGET_CAR_ID: box_color, box_size, line_thick, is_target = (0, 165, 255), 25, 2, True

                # Get TRANSFORMED coordinates for current position
                if 'x_transformed' not in row or 'y_transformed' not in row: continue
                x_tf, y_tf = float(row['x_transformed']), float(row['y_transformed'])
                if np.isnan(x_tf) or np.isnan(y_tf): continue
                pt = np.array([[[x_tf, y_tf]]], dtype=np.float32)
                px_py = cv2.perspectiveTransform(pt, H);
                if px_py is None: continue
                px, py = px_py[0][0]
                if not (-frame_width < px < frame_width*2 and -frame_height < py < frame_height*2): continue

                # Calculate and project FUTURE position
                vx, vy = float(row.get('velocity_x', 0.0)), float(row.get('velocity_y', 0.0))
                if 'x' not in row or 'y' not in row: continue # Need original UTM
                fx_m_utm, fy_m_utm = float(row['x']) + (vx * TIME_HORIZON), float(row['y']) + (vy * TIME_HORIZON)
                fx_c, fy_c = fx_m_utm - params['x_mean'], fy_m_utm - params['y_mean']
                fy_c_inv = params['y_centered_max'] - fy_c
                cos_t, sin_t = np.cos(-params['theta']), np.sin(-params['theta'])
                fx_tf = fx_c * cos_t - fy_c_inv * sin_t; fy_tf = fx_c * sin_t + fy_c_inv * cos_t
                fpx, fpy = px, py # Default
                if not (np.isnan(fx_tf) or np.isnan(fy_tf)):
                    fpt = np.array([[[fx_tf, fy_tf]]], dtype=np.float32)
                    fpx_py = cv2.perspectiveTransform(fpt, H)
                    if fpx_py is not None:
                        fpx, fpy = fpx_py[0][0]
                        fpx, fpy = np.clip(fpx, 0, frame_width - 1), np.clip(fpy, 0, frame_height - 1)

                # --- Draw elements ---
                if 0 <= px < frame_width and 0 <= py < frame_height:
                    # Draw box
                    pt1 = (int(px-box_size), int(py-box_size))
                    pt2 = (int(px+box_size), int(py+box_size))
                    rounded_rectangle(frame, pt1, pt2, box_color, line_thick)
                    if is_target: cv2.putText(frame, f"ID:{obj_id}", (int(px), int(py-box_size-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    # --- COLOR-CODE THE LINE BASED ON RISK (for targets) ---
                    line_color = box_color # Default to box color
                    if is_target:
                        line_color = risk_color # Use the same color as the risk bar
                    # --------------------------------------------------------

                    # Draw line if movement predicted, using the determined line_color
                    # Using the lowered threshold 0.5
                    should_draw_line = abs(px - fpx) > 0.5 or abs(py - fpy) > 0.5
                    if should_draw_line:
                         # Ensure line coordinates are integers
                        cv2.line(frame, (int(px), int(py)), (int(fpx), int(fpy)), line_color, line_thick) # Use line_color

            except Exception as e: print(f"💥 ERROR drawing obj {row.get('trackId', 'N/A')} frame {frame_id}: {e}"); continue

        out.write(frame); processed_count += 1
        # Optional: Display frame during rendering
        # cv2.imshow("Rendering", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break # Press 'q' to quit

        if processed_count % 100 == 0 or processed_count == 1:
            print(f"Processed {processed_count}/{total_frames_to_render} frames...")

    # --- Cleanup ---
    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"\n✅ Final demo saved to '{OUTPUT_VIDEO_PATH}' ({processed_count} frames written).")

if __name__=="__main__":
    main()