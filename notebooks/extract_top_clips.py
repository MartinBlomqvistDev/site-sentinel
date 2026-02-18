# scripts/extract_top_clips.py

import os
import cv2
import pandas as pd
import numpy as np

# --- CONFIG ---
TOP_EVENTS_CSV = "data/analysis_results/top_20_dynamic_events.csv"
CLIP_LENGTH_FRAMES = 150  # total frames to extract (centered on event frame)
OUTPUT_DIR = "data/clips"
FORCED_FPS = 30000 / 1001  # 29.97 FPS

# Visualization
BBOX_COLOR = {
    'Car': (0, 0, 255),
    'Bicycle': (0, 255, 0),
    'Pedestrian': (255, 0, 0)
}
LINE_COLOR = (255, 255, 0)
TARGET_COLOR = (0, 255, 255)  # cyan for target objects
TARGET_THICKNESS = 4
NORMAL_THICKNESS = 2
TARGET_BBOX_SIZE = (40, 40)
NORMAL_BBOX_SIZE = (30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX

YEARS = ["2019", "2020", "2022"]  # automatically handle 2020 as _cal

# Create folders
for year in YEARS:
    os.makedirs(os.path.join(OUTPUT_DIR, year), exist_ok=True)

# Read top events
events_df = pd.read_csv(TOP_EVENTS_CSV)
print(f"Extracting clips for {len(events_df)} events...")

for idx, row in events_df.iterrows():
    csv_file = os.path.basename(row['file'])
    event_frame = int(row['frame'])
    year = csv_file[:4]

    # Map to correct video name
    if year in ["2019", "2020"]:
        video_file = csv_file.replace("_ann.csv", "_cal.MP4")
    else:  # 2022
        video_file = csv_file.replace("_ann.csv", "_org.MP4")

    video_path = os.path.join("data/raw", year, video_file)
    if not os.path.exists(video_path):
        print(f"⚠️  Video not found: {video_path}, skipping.")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️  Can't open video: {video_path}, skipping.")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(event_frame - CLIP_LENGTH_FRAMES // 2, 0)
    end_frame = min(event_frame + CLIP_LENGTH_FRAMES // 2, total_frames-1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_filename = f"{year}_event{idx+1}_{video_file.replace('.MP4','')}_frame{event_frame}.mp4"
    out_path = os.path.join(OUTPUT_DIR, year, out_filename)
    out = cv2.VideoWriter(out_path, fourcc, FORCED_FPS, (width, height))

    # Load annotation CSV
    ann_path = os.path.join("data/raw", year, csv_file)
    if not os.path.exists(ann_path):
        print(f"⚠️  Annotation file not found: {ann_path}, skipping visualization.")
        cap.release()
        out.release()
        continue
    ann_df = pd.read_csv(ann_path)

    # Identify target objects
    target_trackIds = [row['trackId_vuln'], row['trackId_car']]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for f in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        frame_objs = ann_df[ann_df['frame'] == f]
        for _, o in frame_objs.iterrows():
            x, y = int(o['x']), int(o['y'])
            cls = o['class']
            track_id = o['trackId']

            # Determine bbox visual
            if track_id in target_trackIds:
                color = TARGET_COLOR
                thickness = TARGET_THICKNESS
                size = TARGET_BBOX_SIZE
            else:
                color = BBOX_COLOR.get(cls, (255, 255, 255))
                thickness = NORMAL_THICKNESS
                size = NORMAL_BBOX_SIZE

            cv2.rectangle(frame,
                          (x - size[0]//2, y - size[1]//2),
                          (x + size[0]//2, y + size[1]//2),
                          color, thickness)

            # Draw predicted line to next frame
            next_obj = ann_df[(ann_df['trackId'] == track_id) & (ann_df['frame'] == f + 1)]
            if not next_obj.empty:
                nx, ny = int(next_obj['x'].values[0]), int(next_obj['y'].values[0])
                cv2.line(frame, (x, y), (nx, ny), LINE_COLOR, 2)

        out.write(frame)

    cap.release()
    out.release()
    clip_seconds = (end_frame - start_frame + 1) / FORCED_FPS
    print(f"✅ Extracted clip: {out_path} ({clip_seconds:.2f}s)")

print("All done!")
