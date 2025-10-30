# scripts/6_render_video.py
#
# VERSION 15: Temporal Smoothing & Reduced False Alarms
# - Added risk history tracking with configurable window
# - Hysteresis thresholds for state transitions
# - Approach speed gating for IMMINENT state
# - All tunable parameters at top of script
#

import pandas as pd
import numpy as np
import os
import joblib
import cv2
import json
from sklearn.linear_model import LinearRegression
from itertools import combinations
from math import sqrt, hypot

# ==================== CONFIGURATION ====================
# --- DATA & MODEL PATHS ---
RAW_TRAJECTORY_CSV = "data/full_trajectories_PIXELS.csv"
MODEL_PATH = "models/rf_master_predictor_dual_lead_tuned.pkl"
VIDEO_PATH = r"C:\DS24\Site_Sentinel\data\raw\20190918_1500_Sid_StP_3W_d_1_3_cal.mp4"
OUTPUT_VIDEO_PATH = "data/analysis_results/final_demo_smooth_v15.mp4"
HOMOGRAPHY_PATH = "data/analysis_results/homography_matrix.npy"
PARAMS_PATH = "data/analysis_results/transform_params.json"

# --- TIME WINDOW ---
MAIN_EVENT_START_TIME = 4*60 + 7  # 04:07 in seconds
WINDOW_BEFORE = 10  # seconds before event
WINDOW_AFTER = 10   # seconds after event

# --- TARGET TRACKING ---
TARGET_PERSON_ID = 114  # Main worker to monitor

# --- PREDICTION & PHYSICS ---
TIME_HORIZON = 4.0  # Prediction horizon (seconds)
FRAME_RATE = 25  # Default FPS
INTERACTION_DISTANCE_THRESHOLD = 35.0  # meters
SAFETY_BUBBLE_RADIUS_PIXELS = 50.0  # pixels

# --- OBJECT CLASSES ---
VEHICLE_CLASSES = ['Car', 'Medium Vehicle', 'Heavy Vehicle', 'Bus', 'Motorcycle']
VULNERABLE_CLASSES = ['Pedestrian', 'Bicycle']

# --- RISK CALCULATION WEIGHTS ---
AI_MODEL_WEIGHT = 0.80       # Trust in ML model
PROXIMITY_WEIGHT = 0.10      # Current distance influence
PREVENTIVE_WEIGHT = 0.10     # Future prediction influence

# --- TEMPORAL SMOOTHING CONFIG ---
RISK_HISTORY_WINDOW = 15     # frames (~0.6s at 25 FPS)
IMMINENT_PERSISTENCE_FRAMES = 10  # frames that must be >0.8 for IMMINENT

# --- RISK THRESHOLDS (with hysteresis) ---
# Entering thresholds (from lower state)
APPROACHING_ENTER_THRESHOLD = 0.50
IMMINENT_ENTER_THRESHOLD = 0.85
IMMINENT_MIN_APPROACH_SPEED = 0.5  # m/s - must be closing in

# Exiting thresholds (to lower state)
APPROACHING_EXIT_THRESHOLD = 0.40
IMMINENT_EXIT_THRESHOLD = 0.65

# --- VISUALIZATION CONFIG ---
LINE_THRESHOLD = 0.5  # pixels - minimum movement to draw prediction line
CSV_METADATA_LINES = 80  # Lines to skip in CSV

# --- DRAWING SIZES ---
DEFAULT_BOX_SIZE = 15
DEFAULT_LINE_THICK = 1
VULNERABLE_BOX_SIZE = 20
VULNERABLE_LINE_THICK = 1
WORKER_BOX_SIZE = 25
WORKER_LINE_THICK = 2
ALERT_BOX_SIZE = 25
ALERT_LINE_THICK = 2

# --- COLORS (BGR format) ---
COLOR_SAFE = (0, 255, 0)        # Green
COLOR_APPROACHING = (0, 165, 255)  # Orange
COLOR_IMMINENT = (0, 0, 255)    # Red
COLOR_DEFAULT = (255, 180, 0)   # Teal
# ======================================================

# --- CSV PARSING CONSTANTS ---
NUM_FIXED_COLS = 12
NUM_TRAJ_COLS_PER_STEP = 9
IDX_UTM_X = 0
IDX_UTM_Y = 1
IDX_TIME = 5
IDX_PIXEL_X = 7
IDX_PIXEL_Y = 8

def parse_complex_dfs_csv(filepath):
    """Parse the complex DFS trajectory CSV format."""
    all_points = []
    skipped_rows = 0
    parsed_lines = 0
    print(f"Parsing {filepath}...")
   
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            print(f"Skipping {CSV_METADATA_LINES} metadata lines...")
            for i in range(CSV_METADATA_LINES):
                line = f.readline()
           
            header_line = f.readline()
            print("Header line read.")
           
            for line_num, line in enumerate(f, start=CSV_METADATA_LINES + 2):
                line = line.strip()
                parts = line.split(';')
                parsed_lines += 1
               
                try:
                    track_id = int(parts[0])
                    obj_type = parts[1].strip()
                    traj_data = parts[NUM_FIXED_COLS:]
                except:
                    skipped_rows += 1
                    continue
               
                for i in range(0, len(traj_data), NUM_TRAJ_COLS_PER_STEP):
                    chunk = traj_data[i : i + NUM_TRAJ_COLS_PER_STEP]
                    if len(chunk) == NUM_TRAJ_COLS_PER_STEP:
                        try:
                            utm_x = float(chunk[IDX_UTM_X])
                            utm_y = float(chunk[IDX_UTM_Y])
                            time = float(chunk[IDX_TIME])
                            pixel_x = float(chunk[IDX_PIXEL_X])
                            pixel_y = float(chunk[IDX_PIXEL_Y])
                           
                            all_points.append({
                                'trackId': track_id,
                                'class': obj_type,
                                'time': time,
                                'x': utm_x,
                                'y': utm_y,
                                'pixel_x': pixel_x,
                                'pixel_y': pixel_y
                            })
                        except:
                            continue
   
    except Exception as e:
        print(f"❌ ERROR parsing {filepath}: {e}")
        return pd.DataFrame()
   
    if skipped_rows > 0:
        print(f"⚠️ Skipped {skipped_rows} lines.")
   
    if not all_points:
        print("❌ ERROR: No data points parsed.")
        return pd.DataFrame()
   
    df = pd.DataFrame(all_points)
    for col in ['trackId', 'time', 'x', 'y', 'pixel_x', 'pixel_y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
   
    df = df.dropna().astype({'trackId': int})
    return df

def calculate_motion_features(df):
    """Calculate velocity, speed, and acceleration for each track."""
    if df.empty:
        return df
   
    df = df.sort_values(by=['trackId', 'time']).reset_index(drop=True)
   
    df['delta_t'] = df.groupby('trackId')['time'].diff()
    df['delta_x'] = df.groupby('trackId')['x'].diff()
    df['delta_y'] = df.groupby('trackId')['y'].diff()
   
    df['velocity_x'] = (df['delta_x'] / df['delta_t'].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
    df['velocity_y'] = (df['delta_y'] / df['delta_t'].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
    df['speed_ms'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
   
    df['delta_speed'] = df.groupby('trackId')['speed_ms'].diff()
    df['accel_ms2'] = (df['delta_speed'] / df['delta_t'].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
   
    cols_to_return = ['time', 'trackId', 'class', 'x', 'y', 'velocity_x', 'velocity_y', 'speed_ms', 'accel_ms2']
    if 'pixel_x' in df.columns:
        cols_to_return.extend(['pixel_x', 'pixel_y'])
   
    return df[cols_to_return]

def calculate_multi_pair_features(frame_objects_list, frame_rate_local):
    """Calculate interaction features for vehicle-vulnerable pairs."""
    interaction_features_list = []
   
    if len(frame_objects_list) < 2:
        return interaction_features_list
   
    req_cols = ['trackId', 'class', 'x', 'y', 'velocity_x', 'velocity_y', 'speed_ms', 'accel_ms2']
   
    for obj1, obj2 in combinations(frame_objects_list, 2):
        # Validate both objects have required data
        if not all(key in obj1 and key in obj2 and
                  (isinstance(obj1.get(key), (int, float)) and np.isfinite(obj1.get(key, np.nan))) and
                  (isinstance(obj2.get(key), (int, float)) and np.isfinite(obj2.get(key, np.nan)))
                  for key in req_cols if key not in ['trackId', 'class']):
            continue
       
        # Identify vehicle-vulnerable pairs
        vehicle, vulnerable = None, None
        if obj1['class'] in VEHICLE_CLASSES and obj2['class'] in VULNERABLE_CLASSES:
            vehicle, vulnerable = obj1, obj2
        elif obj2['class'] in VEHICLE_CLASSES and obj1['class'] in VULNERABLE_CLASSES:
            vehicle, vulnerable = obj2, obj1
        else:
            continue
       
        # Calculate distance
        dist = sqrt((vehicle['x'] - vulnerable['x'])**2 + (vehicle['y'] - vulnerable['y'])**2)
       
        if dist < INTERACTION_DISTANCE_THRESHOLD:
            pair_features = {
                'vehicle_id': vehicle['trackId'],
                'vulnerable_id': vulnerable['trackId'],
                'time': vehicle['time']
            }
           
            # Copy features
            for key in ['x', 'y', 'velocity_x', 'velocity_y', 'speed_ms', 'accel_ms2']:
                pair_features[key + '_car'] = vehicle.get(key, 0.0)
                pair_features[key + '_vuln'] = vulnerable.get(key, 0.0)
           
            # Relative metrics
            pair_features['rel_distance'] = dist
            rel_speed_val = sqrt((pair_features['velocity_x_car'] - pair_features['velocity_x_vuln'])**2 +
                                (pair_features['velocity_y_car'] - pair_features['velocity_y_vuln'])**2)
            pair_features['rel_speed'] = rel_speed_val
           
            # Approach speed calculation
            delta_x = pair_features['x_car'] - pair_features['x_vuln']
            delta_y = pair_features['y_car'] - pair_features['y_vuln']
            delta_vx = pair_features['velocity_x_car'] - pair_features['velocity_x_vuln']
            delta_vy = pair_features['velocity_y_car'] - pair_features['velocity_y_vuln']
           
            dot_product_dx_dv = delta_x * delta_vx + delta_y * delta_vy
            approach_speed_val = -dot_product_dx_dv / dist if dist > 0.1 else 0
            pair_features['approach_speed'] = approach_speed_val
           
            # TTC calculation
            dot_product_dv_dv = delta_vx**2 + delta_vy**2
            ttc = -dot_product_dx_dv / dot_product_dv_dv if dot_product_dv_dv > 1e-6 else np.inf
            pair_features['ttc'] = ttc if ttc > 0 and ttc < 1000 else 100
           
            # Future distance prediction
            if approach_speed_val > 0:
                future_dist = dist - approach_speed_val * TIME_HORIZON
            else:
                future_dist = dist
           
            pair_features['future_rel_distance'] = max(0.1, future_dist)
            pair_features['preventive_risk'] = 1.0 / pair_features['future_rel_distance']
           
            # Averaged features (simplified for single frame)
            pair_features['rel_dist_avg_2s'] = dist
            pair_features['rel_speed_avg_2s'] = rel_speed_val
            pair_features['future_rel_dist_avg_2s'] = pair_features['future_rel_distance']
           
            interaction_features_list.append(pair_features)
   
    return interaction_features_list

def apply_dynamic_transform(df, params):
    """Apply coordinate transformation using stored parameters."""
    df = df.copy()
   
    if df.empty or 'x' not in df.columns or 'y' not in df.columns:
        return df
   
    x_mean, y_mean = params['x_mean'], params['y_mean']
    y_centered_max, theta = params['y_centered_max'], params['theta']
   
    df['x_centered'] = df['x'] - x_mean
    df['y_centered'] = df['y'] - y_mean
    df['y_centered_inv'] = y_centered_max - df['y_centered']
   
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    df['x_transformed'] = df['x_centered'] * cos_t - df['y_centered_inv'] * sin_t
    df['y_transformed'] = df['x_centered'] * sin_t + df['y_centered_inv'] * cos_t
   
    return df

def rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=10):
    """Draw a rounded rectangle."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
   
    if x1 >= x2 or y1 >= y2:
        return
   
    radius = max(1, min(radius, int((x2-x1)/2), int((y2-y1)/2)))
   
    cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
   
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

def main():
    print("--- STEP 6: Final Video Rendering (Temporal Smoothing v15) ---")
    print(f"Configuration:")
    print(f"  Risk Weights: AI={AI_MODEL_WEIGHT}, Proximity={PROXIMITY_WEIGHT}, Preventive={PREVENTIVE_WEIGHT}")
    print(f"  Smoothing Window: {RISK_HISTORY_WINDOW} frames")
    print(f"  Thresholds: Approaching={APPROACHING_ENTER_THRESHOLD}, Imminent={IMMINENT_ENTER_THRESHOLD}")
    print(f"  Approach Speed Gate: {IMMINENT_MIN_APPROACH_SPEED} m/s\n")

    # Load trajectory CSV
    all_objects_df = parse_complex_dfs_csv(RAW_TRAJECTORY_CSV)
    if all_objects_df.empty:
        print(f"❌ ERROR: No data from {RAW_TRAJECTORY_CSV}.")
        return
    print(f"✅ Parsed {len(all_objects_df)} pts from {all_objects_df['trackId'].nunique()} tracks.")

    # Calculate motion features
    motion_df = calculate_motion_features(all_objects_df)

    # Video Setup & FPS
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open video {VIDEO_PATH}")
        return
   
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = FRAME_RATE
        print(f"⚠️ WARNING: Using default FPS {fps}.")
   
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_video_frames}")

    # Load Model, Homography, Params
    try:
        model_dict = joblib.load(MODEL_PATH)
        model = model_dict.get('preventive', model_dict.get('prevention', next(iter(model_dict.values()))))
        print("✅ Model loaded.")
        expected_features_raw = getattr(model, 'feature_names_in_', None)
        expected_features = list(expected_features_raw) if expected_features_raw is not None else None
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return
   
    try:
        H = np.load(HOMOGRAPHY_PATH)
        print("✅ Homography loaded.")
    except Exception as e:
        print(f"❌ ERROR loading homography: {e}")
        return
   
    try:
        with open(PARAMS_PATH, 'r') as f:
            params = json.load(f)
        print("✅ Params loaded.")
    except Exception as e:
        print(f"❌ ERROR loading params: {e}")
        return

    # Apply dynamic transformation
    all_objects_df_transformed = apply_dynamic_transform(motion_df, params)
    print("✅ Dynamic transform applied.")

    # Prepare data for rendering loop
    all_objects_df_transformed['frame'] = (all_objects_df_transformed['time'] * fps).round().astype(int)
    all_objects_df_transformed = all_objects_df_transformed.dropna(subset=['frame'])
    objects_by_frame = {f: g.to_dict('records') for f, g in all_objects_df_transformed.groupby('frame')}

    # Calculate start/end frames
    start_frame = max(0, int((MAIN_EVENT_START_TIME - WINDOW_BEFORE) * fps))
    end_frame = min(total_video_frames, int((MAIN_EVENT_START_TIME + WINDOW_AFTER) * fps))
    total_frames_to_render = max(0, end_frame - start_frame + 1)
    print(f"Rendering frames {start_frame} to {end_frame} ({total_frames_to_render} frames / ~{total_frames_to_render/fps:.1f} s)...")

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"❌ ERROR opening VideoWriter.")
        cap.release()
        return

    # Define features for prediction
    base_interaction_features = ['rel_distance', 'rel_speed', 'speed_ms_car', 'speed_ms_vuln',
                                 'accel_ms2_car', 'accel_ms2_vuln', 'ttc', 'approach_speed',
                                 'rel_dist_avg_2s', 'rel_speed_avg_2s', 'future_rel_dist_avg_2s']
   
    features_to_predict = expected_features if expected_features is not None else base_interaction_features
   
    if expected_features:
        missing = [f for f in expected_features if f not in base_interaction_features]
        if missing:
            print(f"❌ ERROR: Model expects features not calculated: {missing}")
            features_to_predict = [f for f in expected_features if f in base_interaction_features]
        if not features_to_predict:
            print("❌ ERROR: No common features.")
            features_to_predict = []
   
    print(f"Using features for prediction: {features_to_predict}")

    # STATE TRACKING
    risk_history = []
    current_warning_state = "SAFE"

    # VIDEO RENDERING LOOP
    frame_id = -1
    processed_count = 0
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
       
        frame_id += 1
       
        if frame_id < start_frame:
            continue
        if frame_id > end_frame:
            print("Reached end frame.")
            break

        current_frame_objects = objects_by_frame.get(frame_id, [])
        future_pixel_positions = {}

        # Pass 1: Calculate future positions for ALL objects
        for obj_data in current_frame_objects:
            try:
                obj_id = obj_data['trackId']
                if 'x_transformed' not in obj_data or 'y_transformed' not in obj_data:
                    continue
               
                x_tf, y_tf = float(obj_data['x_transformed']), float(obj_data['y_transformed'])
                if np.isnan(x_tf) or np.isnan(y_tf):
                    continue
               
                pt = np.array([[[x_tf, y_tf]]], dtype=np.float32)
                px_py = cv2.perspectiveTransform(pt, H)
                px, py = px_py[0][0]
               
                vx, vy = float(obj_data.get('velocity_x', 0.0)), float(obj_data.get('velocity_y', 0.0))
                if 'x' not in obj_data or 'y' not in obj_data:
                    continue
               
                fx_m_utm = float(obj_data['x']) + (vx * TIME_HORIZON)
                fy_m_utm = float(obj_data['y']) + (vy * TIME_HORIZON)
               
                fx_c = fx_m_utm - params['x_mean']
                fy_c = fy_m_utm - params['y_mean']
                fy_c_inv = params['y_centered_max'] - fy_c
               
                cos_t, sin_t = np.cos(-params['theta']), np.sin(-params['theta'])
                fx_tf = fx_c * cos_t - fy_c_inv * sin_t
                fy_tf = fx_c * sin_t + fy_c_inv * cos_t
               
                fpx, fpy = px, py
                if not (np.isnan(fx_tf) or np.isnan(fy_tf)):
                    fpt = np.array([[[fx_tf, fy_tf]]], dtype=np.float32)
                    fpx_py = cv2.perspectiveTransform(fpt, H)
                    if fpx_py is not None:
                        fpx, fpy = fpx_py[0][0]
                        fpx = np.clip(fpx, 0, frame_width - 1)
                        fpy = np.clip(fpy, 0, frame_height - 1)
               
                future_pixel_positions[obj_id] = (fpx, fpy)
           
            except Exception as e:
                print(f"⚠️ Error calculating future pos obj {obj_data.get('trackId','N/A')} frame {frame_id}: {e}")
                continue

        # Pass 2: Calculate pair risks
        interaction_pairs_features = calculate_multi_pair_features(current_frame_objects, fps)
        vehicle_risk_to_worker = {}
        max_risk_for_worker = 0.0
        worker_threatened_by_intrusion = False
        intruding_vehicles = set()

        worker_future_pos = future_pixel_positions.get(TARGET_PERSON_ID)

        for pair_data in interaction_pairs_features:
            risk_prob, proximity_risk, preventive_risk = 0.0, 0.0, 0.0
            boosted_risk = 0.0
           
            if features_to_predict:
                try:
                    current_features_dict = {f: pair_data.get(f, 0) for f in features_to_predict}
                    current_features_df = pd.DataFrame([current_features_dict])[features_to_predict]
                   
                    if expected_features and set(features_to_predict) != set(expected_features):
                        current_features_df = current_features_df.reindex(columns=expected_features, fill_value=0)
                   
                    if not current_features_df.isnull().values.any():
                        risk_prob = model.predict_proba(current_features_df)[0][1]
                   
                    rel_distance = pair_data.get('rel_distance')
                    preventive_risk_val = pair_data.get('preventive_risk')
                   
                    if rel_distance is not None and np.isfinite(rel_distance) and rel_distance > 0.1:
                        proximity_risk = (1 / (rel_distance + 0.1)) * np.exp(-rel_distance / 10.0)
                   
                    if preventive_risk_val is not None and np.isfinite(preventive_risk_val):
                        preventive_risk = preventive_risk_val
               
                except Exception as e:
                    print(f"⚠️ Error predicting pair frame {frame_id}: {e}")
                    risk_prob = 0.0
                    proximity_risk = 0.0
                    preventive_risk = 0.0

            boosted_risk = min(1.0, max(0.0,
                (risk_prob * AI_MODEL_WEIGHT) +
                (proximity_risk * PROXIMITY_WEIGHT) +
                (preventive_risk * PREVENTIVE_WEIGHT)
            ))

            veh_id = pair_data['vehicle_id']
            vuln_id = pair_data['vulnerable_id']

            if vuln_id == TARGET_PERSON_ID:
                approach_speed_val = pair_data.get('approach_speed', 0)
               
                if approach_speed_val > IMMINENT_MIN_APPROACH_SPEED:
                    vehicle_risk_to_worker[veh_id] = max(vehicle_risk_to_worker.get(veh_id, 0.0), boosted_risk)
                else:
                    capped_risk = min(boosted_risk, APPROACHING_ENTER_THRESHOLD)
                    vehicle_risk_to_worker[veh_id] = max(vehicle_risk_to_worker.get(veh_id, 0.0), capped_risk)
               
                max_risk_for_worker = max(max_risk_for_worker, vehicle_risk_to_worker[veh_id])
               
                vehicle_future_pos = future_pixel_positions.get(veh_id)
                if worker_future_pos and vehicle_future_pos:
                    future_dist_px = hypot(
                        worker_future_pos[0] - vehicle_future_pos[0],
                        worker_future_pos[1] - vehicle_future_pos[1]
                    )
                    if future_dist_px < SAFETY_BUBBLE_RADIUS_PIXELS:
                        worker_threatened_by_intrusion = True
                        intruding_vehicles.add(veh_id)

        # TEMPORAL SMOOTHING
        risk_history.append(max_risk_for_worker)
        if len(risk_history) > RISK_HISTORY_WINDOW:
            risk_history.pop(0)
       
        smoothed_risk = np.mean(risk_history) if risk_history else 0.0
        high_risk_frame_count = sum(1 for r in risk_history if r > 0.8)
       
        # STATE MACHINE WITH HYSTERESIS
        if current_warning_state == "IMMINENT":
            if smoothed_risk < IMMINENT_EXIT_THRESHOLD:
                current_warning_state = "APPROACHING"
       
        elif current_warning_state == "APPROACHING":
            if high_risk_frame_count >= IMMINENT_PERSISTENCE_FRAMES and smoothed_risk > IMMINENT_ENTER_THRESHOLD:
                current_warning_state = "IMMINENT"
            elif smoothed_risk < APPROACHING_EXIT_THRESHOLD:
                current_warning_state = "SAFE"
       
        else:  # SAFE
            if smoothed_risk > APPROACHING_ENTER_THRESHOLD:
                current_warning_state = "APPROACHING"

        # Draw Overlay Based on State
        if current_warning_state == "IMMINENT":
            indicator_color = COLOR_IMMINENT
            risk_text = "WORKER: IMMINENT"
        elif current_warning_state == "APPROACHING":
            indicator_color = COLOR_APPROACHING
            risk_text = "WORKER: APPROACHING"
        else:
            indicator_color = COLOR_SAFE
            risk_text = f"WORKER RISK: {smoothed_risk:.0%}"
       
        text_x, text_y = frame_width - 300, 40
        (text_width, text_height), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10),
                     (text_x + text_width + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, risk_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, indicator_color, 2)

        # Determine visual warning state
        show_warning_state = worker_threatened_by_intrusion or (current_warning_state != "SAFE")

        # Draw Objects
        for row in current_frame_objects:
            try:
                obj_id = int(row['trackId'])
                obj_class = row.get('class', 'Unknown')

                # Color Logic
                box_color, line_color, box_size, line_thick = COLOR_DEFAULT, COLOR_DEFAULT, DEFAULT_BOX_SIZE, DEFAULT_LINE_THICK
                show_individual_warning_text = False
                is_main_worker = (obj_id == TARGET_PERSON_ID)
                is_intruding_vehicle = (obj_id in intruding_vehicles)

                if is_main_worker:
                    if current_warning_state == "IMMINENT":
                        box_color, line_color = COLOR_IMMINENT, COLOR_IMMINENT
                        box_size, line_thick = ALERT_BOX_SIZE, ALERT_LINE_THICK
                        show_individual_warning_text = True
                    elif current_warning_state == "APPROACHING":
                        box_color, line_color = COLOR_APPROACHING, COLOR_APPROACHING
                        box_size, line_thick = ALERT_BOX_SIZE, ALERT_LINE_THICK
                        show_individual_warning_text = True
                    else:
                        box_color, line_color = COLOR_SAFE, COLOR_SAFE
                        box_size, line_thick = WORKER_BOX_SIZE, WORKER_LINE_THICK
               
                elif is_intruding_vehicle:
                    risk_level = vehicle_risk_to_worker.get(obj_id, 0.0)
                    if risk_level > 0.8:
                        box_color, line_color = COLOR_IMMINENT, COLOR_IMMINENT
                    elif risk_level > 0.6:
                        box_color, line_color = COLOR_APPROACHING, COLOR_APPROACHING
                    else:
                        box_color, line_color = COLOR_APPROACHING, COLOR_APPROACHING
                    box_size, line_thick = ALERT_BOX_SIZE, ALERT_LINE_THICK
               
                elif obj_class in VULNERABLE_CLASSES:
                    box_color, line_color = COLOR_SAFE, COLOR_SAFE
                    box_size, line_thick = VULNERABLE_BOX_SIZE, VULNERABLE_LINE_THICK

                # Drawing
                if 'x_transformed' not in row or 'y_transformed' not in row:
                    continue
               
                x_tf, y_tf = float(row['x_transformed']), float(row['y_transformed'])
                if np.isnan(x_tf) or np.isnan(y_tf):
                    continue
               
                pt = np.array([[[x_tf, y_tf]]], dtype=np.float32)
                px_py = cv2.perspectiveTransform(pt, H)
                if px_py is None:
                    continue
                px, py = px_py[0][0]
               
                if not (-frame_width < px < frame_width*2 and -frame_height < py < frame_height*2):
                    continue

                fpx, fpy = future_pixel_positions.get(obj_id, (px, py))

                if 0 <= px < frame_width and 0 <= py < frame_height:
                    pt1 = (int(px-box_size), int(py-box_size))
                    pt2 = (int(px+box_size), int(py+box_size))
                    rounded_rectangle(frame, pt1, pt2, box_color, line_thick)
                   
                    if is_main_worker or is_intruding_vehicle:
                        cv2.putText(frame, f"ID:{obj_id}", (int(px), int(py-box_size-5)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    if show_individual_warning_text and is_main_worker:
                        cv2.putText(frame, "WARNING!", (int(px - box_size*2), int(py + box_size + 15)),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_IMMINENT, 2)

                    # Draw prediction lines
                    should_draw_line = abs(px - fpx) > LINE_THRESHOLD or abs(py - fpy) > LINE_THRESHOLD
                    if should_draw_line and (is_main_worker or is_intruding_vehicle):
                        cv2.line(frame, (int(px), int(py)), (int(fpx), int(fpy)), line_color, line_thick)
           
            except Exception as e:
                print(f"⚠️ ERROR drawing obj {row.get('trackId', 'N/A')} frame {frame_id}: {e}")
                continue

        out.write(frame)
        processed_count += 1
       
        if processed_count % 100 == 0 or processed_count == 1:
            est_time_left = (total_frames_to_render - processed_count) / fps if fps > 0 else 0
            print(f"Processed {processed_count}/{total_frames_to_render} frames... (Est. time left: {est_time_left:.0f}s)")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Final demo saved to '{OUTPUT_VIDEO_PATH}' ({processed_count} frames written).")

if __name__ == "__main__":
    main()