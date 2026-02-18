import pandas as pd
import re

file_path = "data/raw/20190918_1500_Sid_StP_3W_d_1_3_ann_for_demo.csv"

# Read as raw text (no splitting)
with open(file_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

header_line = lines[0]
print("✅ Header line length:", len(header_line))
header_parts = [h.strip() for h in header_line.split(";") if h.strip()]
print(f"✅ Found {len(header_parts)} header fields")

data_lines = lines[1:]
print(f"🔍 Loaded {len(data_lines)} data lines")

# Prepare extracted rows
records = []

for i, line in enumerate(data_lines, start=1):
    # Split only the first ~12 metadata fields, rest is trajectory
    parts = line.split(";", 12)
    if len(parts) < 13:
        continue
    meta = parts[:12]
    trajectory_raw = parts[12]
    records.append(meta + [trajectory_raw])

# Create DataFrame
meta_headers = header_parts[:12] + ["Trajectory_raw"]
df = pd.DataFrame(records, columns=meta_headers)

print(f"✅ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")

# Save cleaned version
out_path = "data/for_demo_clean.csv"
df.to_csv(out_path, index=False)
print(f"✅ Saved cleaned file with {len(df)} rows to {out_path}")
