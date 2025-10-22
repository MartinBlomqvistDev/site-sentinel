# scripts/6a_final_automatic_calibration.py
#
# Läser det komplexa CSV-formatet från DFS Viewer där
# trajektoriedata finns i upprepade kolumner på samma rad.
# Beräknar automatiskt homografi och transformationsparametrar.
#

import pandas as pd
import numpy as np
import cv2
import json
import os
from sklearn.linear_model import LinearRegression

print("--- FINAL AUTOMATIC HOMOGRAPHY CALIBRATION ---")

# --- Konfiguration ---
# ANVÄND DIN NYA, FULLSTÄNDIGA FIL HÄR
INPUT_CSV_PATH = "data/full_trajectories_PIXELS.csv" # Eller vad du sparade den som

OUTPUT_DIR = "data/analysis_results"
OUTPUT_H_PATH = os.path.join(OUTPUT_DIR, "homography_matrix.npy")
OUTPUT_PARAMS_PATH = os.path.join(OUTPUT_DIR, "transform_params.json")

# Antal "fasta" kolumner innan trajektoridatan börjar
# Baserat på din kopierade rad:
# Track ID; Type; ... Avg. Speed [km/h]; Trajectory(...
# Det är 12 fasta kolumner (index 0 till 11)
NUM_FIXED_COLS = 12

# Antal kolumner per tidpunkt i trajektoridatan
# Baserat på din kopierade rad:
# x [m]; y [m]; Speed; Tan. Acc.; Lat. Acc.; Time; Angle; Image x; Image y; ) <-- Sista ')' räknas, 9 kolumner
NUM_TRAJ_COLS_PER_STEP = 9

# Index inom varje trajektorigrupp
IDX_UTM_X = 0 # 'x [m]'
IDX_UTM_Y = 1 # 'y [m]'
IDX_TIME = 5  # 'Time [s]'
IDX_PIXEL_X = 7 # 'Image x [px]'
IDX_PIXEL_Y = 8 # 'Image y [px]'

# --- 1. Läs in och Parsa den komplexa CSV-filen ---
all_points = []
try:
    # Läs hela filen som text, hoppa över metadata
    with open(INPUT_CSV_PATH, 'r', encoding='utf-8') as f:
        # Hoppa över metadata-rader (antar 80, justera om nödvändigt)
        for _ in range(80):
            f.readline()
        header_line = f.readline().strip() # Läs rubrikraden
        
        # Läs resten av raderna
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(';')
            
            try:
                track_id = int(parts[0])
                obj_type = parts[1].strip()
                
                # Extrahera trajektoridelen
                traj_data = parts[NUM_FIXED_COLS:]
                
                # Iterera genom trajektoridatan i block om 9
                for i in range(0, len(traj_data), NUM_TRAJ_COLS_PER_STEP):
                    chunk = traj_data[i : i + NUM_TRAJ_COLS_PER_STEP]
                    
                    if len(chunk) == NUM_TRAJ_COLS_PER_STEP:
                        # Försök extrahera värden, hoppa över om något misslyckas
                        try:
                            utm_x = float(chunk[IDX_UTM_X])
                            utm_y = float(chunk[IDX_UTM_Y])
                            time = float(chunk[IDX_TIME])
                            pixel_x = float(chunk[IDX_PIXEL_X])
                            pixel_y = float(chunk[IDX_PIXEL_Y])
                            
                            # Lägg till som en punkt (en rad) i vår lista
                            all_points.append({
                                'track_id': track_id,
                                'type': obj_type,
                                'time': time,
                                'utm_x': utm_x,
                                'utm_y': utm_y,
                                'pixel_x': pixel_x,
                                'pixel_y': pixel_y
                            })
                        except (ValueError, IndexError):
                            # Ignorera felaktiga datablock inom en rad
                            continue 
                            
            except (ValueError, IndexError):
                # Ignorera hela rader som inte kan parsas (t.ex. felaktigt Track ID)
                print(f"⚠️ Varnning: Hoppade över felaktig rad: {line[:50]}...")
                continue

    # Skapa en DataFrame från alla extraherade punkter
    df = pd.DataFrame(all_points)
    
    if df.empty:
        print(f"❌ Fel: Ingen giltig trajektoriedata kunde extraheras från {INPUT_CSV_PATH}.")
        exit()
        
    print(f"✅ Parsade {len(df)} datapunkter från {df['track_id'].nunique()} spår.")
    print(df.head())

except FileNotFoundError:
    print(f"❌ Fel: Filen hittades inte: {INPUT_CSV_PATH}")
    exit()
except Exception as e:
    print(f"❌ Fel vid inläsning eller parsning av CSV: {e}")
    exit()

# --- 2. Applicera STABIL Dynamisk Transform på UTM-punkter ---
df_transformed = df.copy()
df_transformed['x_dyn'] = df_transformed['utm_x']
df_transformed['y_dyn'] = df_transformed['utm_y']

x_mean = df_transformed['x_dyn'].mean()
y_mean = df_transformed['y_dyn'].mean()
df_transformed['x_centered'] = df_transformed['x_dyn'] - x_mean
df_transformed['y_centered'] = df_transformed['y_dyn'] - y_mean
y_centered_max = df_transformed['y_centered'].max()
df_transformed['y_centered_inv'] = y_centered_max - df_transformed['y_centered']

X = df_transformed['x_centered'].values.reshape(-1,1)
y = df_transformed['y_centered_inv'].values
valid_mask = ~np.isnan(X).flatten() & ~np.isnan(y)

if valid_mask.sum() < 2:
    print("❌ Fel: Inte tillräckligt med data för linjär regression.")
    exit()

model = LinearRegression().fit(X[valid_mask], y[valid_mask])
slope = model.coef_[0]
theta = np.arctan(slope)

print(f"Beräknad dynamisk rotationsvinkel: {-np.rad2deg(theta):.2f} grader")

cos_t = np.cos(-theta)
sin_t = np.sin(-theta)
df_transformed['x_transformed'] = df_transformed['x_centered'] * cos_t - df_transformed['y_centered_inv'] * sin_t
df_transformed['y_transformed'] = df_transformed['x_centered'] * sin_t + df_transformed['y_centered_inv'] * cos_t

transform_params = {
    'x_mean': x_mean,
    'y_mean': y_mean,
    'y_centered_max': y_centered_max,
    'theta': theta
}
print("✅ Transformationsparametrar beräknade.")

# --- 3. Beräkna Homografi Automatiskt ---
# Källpunkter: De dynamiskt transformerade UTM-koordinaterna
pts_source = df_transformed[['x_transformed', 'y_transformed']].to_numpy().astype(np.float32)

# Målpunkter: Pixelkoordinaterna
pts_destination = df_transformed[['pixel_x', 'pixel_y']].to_numpy().astype(np.float32)

# Filtrera bort eventuella NaN som kan ha uppstått
valid_pts_mask = ~np.isnan(pts_source).any(axis=1) & ~np.isnan(pts_destination).any(axis=1)
pts_source = pts_source[valid_pts_mask]
pts_destination = pts_destination[valid_pts_mask]

if len(pts_source) < 4:
    print(f"❌ Fel: För få giltiga punktpar ({len(pts_source)}) för att beräkna homografi.")
    exit()

H, mask = cv2.findHomography(pts_source, pts_destination, cv2.RANSAC, 5.0)

if H is None:
    print("❌ FEL: Homografi-beräkningen misslyckades!")
    exit()

print(f"✅ Homografi-matris beräknad AUTOMATISKT (använde {mask.sum()}/{len(mask)} punkter).")

# --- 4. Spara BÅDA filerna ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(OUTPUT_H_PATH, H)
print(f"✅ Homografi-matris sparad till '{OUTPUT_H_PATH}'")
with open(OUTPUT_PARAMS_PATH, 'w') as f:
    json.dump(transform_params, f, indent=4)
print(f"✅ Transformationsparametrar sparade till '{OUTPUT_PARAMS_PATH}'")

print("\n--- KLAR ---")
print("Automatisk kalibrering är slutförd med den nya datan.")
print("Du kan nu köra ditt visualiseringsskript (t.ex. 6b eller 6).")