import pandas as pd
import re
from pathlib import Path

# Input/output paths
input_file = Path("data/for_demo_clean.csv")
output_file = Path("data/for_demo_xy.csv")

print(f"🔍 Loading {input_file} ...")
df = pd.read_csv(input_file)

# We'll store clean coordinate results here
coords = []

for i, row in df.iterrows():
    traj = str(row["Trajectory_raw"])
    # Extract numeric sequences separated by semicolons or spaces
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", traj)
    if len(nums) < 10:
        continue  # Skip incomplete
    try:
        # Image X = second-to-last value, Image Y = last value
        image_x = float(nums[-2])
        image_y = float(nums[-1])
        coords.append({
            "track_id": row.get("Track ID", i),
            "type": row.get("Type", "Unknown"),
            "x": image_x,
            "y": image_y
        })
    except ValueError:
        continue

# Convert to DataFrame
df_out = pd.DataFrame(coords)
print(f"✅ Extracted {len(df_out)} coordinate pairs")

# Save
df_out.to_csv(output_file, index=False)
print(f"✅ Saved {output_file}")
