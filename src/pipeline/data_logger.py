import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
import time

# --- CONFIGURATION ---
# Path to the final, short demo clip (local access)
INPUT_CLIP_PATH = "../data/final_test_clip/demo_clip.mp4" 

# Output CSV file for raw tracking data
OUTPUT_CSV_PATH = "../data/logs/tracking_data.csv"

# Pre-trained YOLOv8 model weights (using 'n' for speed in MVP)
YOLO_MODEL_PATH = "yolov8n.pt" 
# ---------------------

def initialize_tracker():
    """
    NOTE: Tracking libraries (DeepSORT/ByteTrack) are complex to install via pip.
    For the MVP, we use the basic tracking built into the ultralytics package 
    to quickly get tracking IDs.
    """
    # Load YOLO model
    model = YOLO(YOLO_MODEL_PATH)
    return model

def run_data_logger(model, input_path):
    """Processes video, runs YOLO tracking, and logs results."""
    
    # Resolve local path for the input clip
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_full_path = os.path.join(base_dir, input_path)
    output_full_path = os.path.join(base_dir, OUTPUT_CSV_PATH)
    
    cap = cv2.VideoCapture(input_full_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file at {input_full_path}. Ensure it exists.")
        return None

    frame_data = []
    frame_count = 0
    start_time = time.time()
    
    # Headers for the CSV file
    columns = ['frame_id', 'object_id', 'object_class', 'x_center', 'y_center']

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- YOLOv8 Tracking ---
        # Run inference with tracking. The 'persist=True' maintains tracking state.
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        frame_count += 1
        
        # Check if tracking IDs were assigned
        if results and results[0].boxes.id is not None:
            
            boxes = results[0].boxes
            
            # Convert object IDs (tensors) and coordinates to numpy arrays
            track_ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xywh.cpu().numpy() # x_center, y_center, width, height
            clss = boxes.cls.cpu().numpy().astype(int)
            
            # Map class IDs to names (simplified: just use 'car' and 'person' for MVP)
            class_names = [model.names[int(c)] for c in clss]
            
            for i, track_id in enumerate(track_ids):
                yolo_class_name = class_names[i]
                
                # --- NEW MAPPING STEP ---
                category = map_class_to_category(yolo_class_name) 
                
                # We only log Personnel and Hazard Vehicles, ignoring static items
                if category not in ['personnel', 'hazard_vehicle']:
                    continue

                # Get coordinates
                x_center, y_center, _, _ = xyxy[i] 
                
                frame_data.append({
                    'frame_id': frame_count,
                    'object_id': track_id,
                    'object_class': category, # LOGGING THE MAPPED CATEGORY!
                    'x_center': x_center,
                    'y_center': y_center
                })
        
        # Limit processing time for initial test
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames. Elapsed time: {time.time() - start_time:.2f}s")
            
        # Optional: Uncomment if you want to see the video live while logging
        # annotated_frame = results[0].plot()
        # cv2.imshow("Tracking Log", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(frame_data, columns=columns)
    
    # Save to CSV
    df.to_csv(output_full_path, index=False)
    print("\n--- Data Logger Complete ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Total entries logged: {len(df)}")
    print(f"CSV saved to: {output_full_path}")
    
    return df


def map_class_to_category(yolo_class_name):
    """
    Maps detailed YOLO class names (from annotations) to the two broad categories 
    needed for Feature Engineering (Personnel or Hazard).
    """
    
    # 1. Define Personnel Categories
    personnel_classes = [
        'person', 'worker', 'flagger', 
        'safety_vest', 'hard_hat'
    ]
    
    # 2. Define Hazard/Vehicle Categories
    hazard_classes = [
        'car', 'vehicle', 'truck', 'bus', 
        'excavator', 'heavy_machinery', 'machinery'
    ]

    # 3. Perform Mapping
    if yolo_class_name in personnel_classes:
        return 'personnel'
    elif yolo_class_name in hazard_classes:
        return 'hazard_vehicle'
    else:
        # Fallback for cones, barrels, signs, etc. which are not tracked hazards
        return 'other_static'
    

if __name__ == "__main__":
    # NOTE: You must have run D12 (YOLO Fine-Tuning) OR be using a pre-trained model (yolov8n.pt)
    # And you must have your demo_clip.mp4 ready in data/final_test_clip/
    
    model = initialize_tracker()
    run_data_logger(model, INPUT_CLIP_PATH)