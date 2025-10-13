# scripts/6_render_video.py

import pandas as pd
import numpy as np
import os
import joblib
import cv2

# --- CONFIGURATION ---
# The feature file for the single event we want to visualize
FEATURES_CSV = "data/analysis_results/advanced_features_20220629_1530_Sid_StP_3W_d_1_18_ann.csv"

# Our champion model, trained on the master dataset
MODEL_PATH = "models/rf_master_predictor_tuned.pkl" 
ROOT_VIDEO_DIR = "data/raw"

# IMPORTANT: Verify this filename matches the video you downloaded for the 1.01m event
VIDEO_FILENAME = "20220629_1530_Sid_StP_3W_d_1_18_cal.mp4" 

OUTPUT_VIDEO_PATH = "data/analysis_results/final_demo_master_model.mp4"
FRAME_RATE = 25
# ---------------------

def main():
    """
    Loads the final tuned model and assets to render the predictive video.
    """
    print("--- STEP 6: Final Video Rendering ---")

    print("Loading assets...")
    if not os.path.exists(FEATURES_CSV):
        print(f"ERROR: Advanced features file not found at '{FEATURES_CSV}'")
        return
    
    features_df = pd.read_csv(FEATURES_CSV)
    model = joblib.load(MODEL_PATH)
    
    video_path = os.path.join(ROOT_VIDEO_DIR, VIDEO_FILENAME)
    if not os.path.exists(video_path):
        print(f"FATAL ERROR: Video file not found at '{video_path}'.")
        print(f"Please make sure your video is named '{VIDEO_FILENAME}' and is in the '{ROOT_VIDEO_DIR}' folder.")
        return

    features_df['frame'] = (features_df['time'] * FRAME_RATE).round().astype(int)
    features_to_use = ['rel_distance', 'rel_speed', 'speed_ms_vuln', 'speed_ms_car', 'accel_ms2_vuln', 'accel_ms2_car', 'ttc']

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FRAME_RATE, (frame_width, frame_height))
    print(f"Video opened. Resolution: {frame_width}x{frame_height}. Preparing to render...")

    frame_id = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        
        interaction_data = features_df[features_df['frame'] == frame_id]
        
        risk_prob = 0.0
        if not interaction_data.empty:
            current_features = interaction_data[features_to_use].fillna(100)
            risk_prob = model.predict_proba(current_features)[0][1]

        if risk_prob > 0.75: color, status_text = ((0, 0, 255), f"PREDICTED RISK: {risk_prob:.0%} (HIGH)")
        elif risk_prob > 0.5: color, status_text = ((0, 165, 255), f"PREDICTED RISK: {risk_prob:.0%} (MEDIUM)")
        else: color, status_text = ((0, 255, 0), f"STATUS: SAFE (Risk: {risk_prob:.0%})")

        # Draw a semi-transparent background for the text
        cv2.rectangle(frame, (40, 20), (850, 80), (0,0,0), -1)
        cv2.addWeighted(frame, 0.5, frame, 0.5, 0)
        
        cv2.putText(frame, status_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # This visualization part uses a simple scaling to draw the meter-based coordinates onto the pixel-based video.
        # It is an abstract representation, not a perfect real-world projection.
        for _, row in interaction_data.iterrows():
            # For the vulnerable user (bicycle)
            px_v = int(row['x_vuln'] * 20 - 8000)
            py_v = int(row['y_vuln'] * 20 - 113000)
            cv2.rectangle(frame, (px_v-20, py_v-20), (px_v+20, py_v+20), color, 3)
            cv2.putText(frame, f"ID:{int(row['trackId_vuln'])}", (px_v, py_v-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # For the car
            px_c = int(row['x_car'] * 20 - 8000)
            py_c = int(row['y_car'] * 20 - 113000)
            cv2.rectangle(frame, (px_c-25, py_c-25), (px_c+25, py_c+25), color, 3)
            cv2.putText(frame, f"ID:{int(row['trackId_car'])}", (px_c, py_c-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)

        if frame_id % 100 == 0: print(f"Processed frame {frame_id}...")

    cap.release(); out.release(); cv2.destroyAllWindows()

    print(f"\n✅ Video rendering complete!")
    print(f"Final demo saved to '{OUTPUT_VIDEO_PATH}'")

if __name__ == "__main__":
    main()