# scripts/6_render_video.py

import pandas as pd
import numpy as np
import os
import joblib
import cv2

# --- CONFIGURATION ---
FEATURES_CSV = "data/analysis_results/ultimate_features_20220629_1530_Sid_StP_3W_d_1_18_ann.csv"
ALL_FEATURES_CSV = "data/analysis_results/features_20220629_1530_Sid_StP_3W_d_1_18_ann.csv"
MODEL_PATH = "models/rf_ultimate_predictor_tuned.pkl" 
ROOT_VIDEO_DIR = "data/raw"
VIDEO_FILENAME = "20220629_1530_Sid_StP_3W_d_1_18_org.MP4"
OUTPUT_VIDEO_PATH = "data/analysis_results/final_demo_homography.mp4"

# NEW: Path to the matrix we just created
HOMOGRAPHY_PATH = "data/analysis_results/homography_matrix.npy"
# ---------------------

def main():
    """
    Renders the predictive video using a precise homography transformation.
    """
    print("--- STEP 6: Final Video Rendering (with Homography) ---")

    print("Loading assets...")
    interaction_df = pd.read_csv(FEATURES_CSV)
    all_objects_df = pd.read_csv(ALL_FEATURES_CSV)
    model = joblib.load(MODEL_PATH)
    
    # --- NEW: Load the Homography Matrix ---
    if not os.path.exists(HOMOGRAPHY_PATH):
        print(f"FATAL ERROR: Homography matrix not found at '{HOMOGRAPHY_PATH}'.")
        print("Please run '6a_get_homography.py' first.")
        return
    H = np.load(HOMOGRAPHY_PATH)
    print("Homography matrix loaded successfully.")
    # -------------------------------------

    video_path = os.path.join(ROOT_VIDEO_DIR, VIDEO_FILENAME)
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    interaction_df['frame'] = (interaction_df['time'] * source_fps).round().astype(int)
    all_objects_df['frame'] = (all_objects_df['time'] * source_fps).round().astype(int)
    
    features_to_use = ['rel_distance', 'rel_speed', 'speed_ms_vuln', 'speed_ms_car', 
                       'accel_ms2_vuln', 'accel_ms2_car', 'ttc',
                       'approach_speed', 'rel_dist_avg_2s', 'rel_speed_avg_2s']

    target_vuln_id = interaction_df['trackId_vuln'].iloc[0]
    target_car_id = interaction_df['trackId_car'].iloc[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, source_fps, (frame_width, frame_height))
    print(f"Video opened. Preparing to render...")
    
    frame_id = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        
        # ... (Risk prediction logic is the same) ...
        interaction_data = interaction_df[interaction_df['frame'] == frame_id]
        risk_prob = 0.0
        if not interaction_data.empty:
            current_features = interaction_data[features_to_use].fillna(100)
            risk_prob = model.predict_proba(current_features)[0][1]
        if risk_prob > 0.75: color, status_text = ((0, 0, 255), f"PREDICTED RISK: {risk_prob:.0%} (HIGH)")
        elif risk_prob > 0.5: color, status_text = ((0, 165, 255), f"PREDICTED RISK: {risk_prob:.0%} (MEDIUM)")
        else: color, status_text = ((0, 255, 0), f"STATUS: SAFE (Risk: {risk_prob:.0%})")
        overlay = frame.copy()
        cv2.rectangle(overlay, (40, 20), (850, 80), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, status_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        all_frame_objects = all_objects_df[all_objects_df['frame'] == frame_id]

        for _, row in all_frame_objects.iterrows():
            obj_id = int(row['trackId'])
            box_color = (255, 180, 0)
            box_size = 15
            
            if obj_id == target_vuln_id or obj_id == target_car_id:
                box_color = color
                box_size = 25
            
            # --- NEW: Use Homography for Precise Transformation ---
            point_meters = np.array([[[row['x'], row['y']]]], dtype=np.float32)
            point_pixels = cv2.perspectiveTransform(point_meters, H)
            px, py = int(point_pixels[0][0][0]), int(point_pixels[0][0][1])

            future_x_m = row['x'] + (row['velocity_x'] * 1.5)
            future_y_m = row['y'] + (row['velocity_y'] * 1.5)
            future_point_meters = np.array([[[future_x_m, future_y_m]]], dtype=np.float32)
            future_point_pixels = cv2.perspectiveTransform(future_point_meters, H)
            future_px, future_py = int(future_point_pixels[0][0][0]), int(future_point_pixels[0][0][1])
            # ----------------------------------------------------

            cv2.rectangle(frame, (px-box_size, py-box_size), (px+box_size, py+box_size), box_color, 2)
            cv2.putText(frame, f"ID:{obj_id}", (px, py-box_size-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.line(frame, (px, py), (future_px, future_py), box_color, 2)

        out.write(frame)

        if frame_id % 100 == 0: print(f"Processed frame {frame_id}...")

    cap.release(); out.release(); cv2.destroyAllWindows()

    print(f"\n✅ Video rendering complete!")
    print(f"Final demo saved to '{OUTPUT_VIDEO_PATH}'")

if __name__ == "__main__":
    main()