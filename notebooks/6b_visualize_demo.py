# scripts/6b_visualize_demo.py
#
# Detta är det KORREKTA, STABILA visualiseringsskriptet.
# Det laddar BÅDE homografimatrisen och transformationsparametrarna
# från 'data/analysis_results/' för att korrekt mappa datan till videon.
#

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json # Behövs för att ladda params
import os # För att kontrollera sökvägar

# --- KONFIGURATION ---
CSV_PATH = "data/for_demo_clean.csv" # Laddar från data/
VIDEO_PATH = "data/raw/20190918_1500_Sid_StP_3W_d_1_3_cal.MP4"

# Laddar från analysis_results mappen
HOMOGRAPHY_PATH = "data/analysis_results/homography_matrix.npy"
PARAMS_PATH = "data/analysis_results/transform_params.json"

# --- Ladda homografi och parametrar ---
try:
    H = np.load(HOMOGRAPHY_PATH)
    print(f"✅ Laddade homografimatris från {HOMOGRAPHY_PATH}")
except FileNotFoundError:
    print(f"❌ FEL: Homografi-fil hittades inte: {HOMOGRAPHY_PATH}")
    print("Se till att du har kört '6a_get_homography.py' (det automatiska skriptet) först.")
    exit()
    
try:
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
    print(f"✅ Laddade transformationsparametrar från {PARAMS_PATH}")
except FileNotFoundError:
    print(f"❌ FEL: Parameterfil hittades inte: {PARAMS_PATH}")
    print("Se till att du har kört '6a_get_homography.py' (det automatiska skriptet) först.")
    exit()

# --- Ladda data ---
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Laddade {len(df)} rader från {CSV_PATH}")
except FileNotFoundError:
    print(f"❌ FEL: CSV-fil hittades inte: {CSV_PATH}")
    exit()


# --- (FIXED) Parsa Trajectory_raw till x/y ---
def parse_trajectory(raw_str):
    """
    Parsar säkert den råa trajektoriesträngen till x- och y-koordinater.
    Returnerar (np.nan, np.nan) om parsning misslyckas.
    """
    if not isinstance(raw_str, str) or not raw_str:
        return np.nan, np.nan
    try:
        vals = [s for s in raw_str.split(';') if s.strip()]
        if len(vals) > 2:
            x = float(vals[1]) # UTM X
            y = float(vals[2]) # UTM Y
            return x, y
        else:
            return np.nan, np.nan
    except (ValueError, IndexError):
        return np.nan, np.nan

df[['x','y']] = df['Trajectory_raw'].apply(lambda s: pd.Series(parse_trajectory(s)))
print("✅ Parsade x/y (UTM) från Trajectory_raw")

# --- Hantera misslyckad parsning ---
original_count = len(df)
df = df.dropna(subset=['x', 'y'])
new_count = len(df)
if original_count > new_count:
    print(f"⚠️ Tog bort {original_count - new_count} rader med saknade koordinater.")
if df.empty:
    print("❌ Fel: Ingen giltig trajektoriedata kvar efter parsning. Avslutar.")
    exit()

# --- Applicera STABIL Dynamisk Transform ---
def apply_dynamic_transform(df, params):
    """
    Applicerar de förberäknade transformationsparametrarna (från 6a)
    på den nya dataframen.
    """
    df = df.copy()
    
    # 1. Ladda parametrar
    x_mean = params['x_mean']
    y_mean = params['y_mean']
    y_centered_max = params['y_centered_max']
    theta = params['theta']
    
    # 2. Applicera centrering
    df['x_centered'] = df['x'] - x_mean
    df['y_centered'] = df['y'] - y_mean
    
    # 3. Applicera Y-invertering
    df['y_centered_inv'] = y_centered_max - df['y_centered']
    
    # 4. Applicera rotation
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    df['x_transformed'] = df['x_centered'] * cos_t - df['y_centered_inv'] * sin_t
    df['y_transformed'] = df['x_centered'] * sin_t + df['y_centered_inv'] * cos_t
    
    print("✅ Applicerade dynamisk transformering med laddade parametrar.")
    return df

df = apply_dynamic_transform(df, params)

# --- Applicera homografi ---
# Vi transformerar 'x_transformed' och 'y_transformed' kolonnerna
coords = df[['x_transformed','y_transformed']].to_numpy().astype(np.float32)
coords_reshaped = coords.reshape(-1, 1, 2)

# Detta mappar från (x_transformed, y_transformed) -> (x_pixel, y_pixel)
coords_hom = cv2.perspectiveTransform(coords_reshaped, H).reshape(-1, 2)
df['x_hom'] = coords_hom[:,0]
df['y_hom'] = coords_hom[:,1]
print("✅ Applicerade homografi på koordinaterna.")

# --- Överlägg på video ---
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret:
    print(f"❌ Fel: Kunde inte läsa bildruta från video: {VIDEO_PATH}")
    frame = np.zeros((720,1280,3), dtype=np.uint8)

overlay = frame.copy()
point_count = 0
for x, y in coords_hom:
    # Säkerställ att koordinaterna är giltiga heltal innan de ritas
    if np.isfinite(x) and np.isfinite(y):
        # Rita bara om de är innanför bildrutans gränser
        if 0 < x < frame.shape[1] and 0 < y < frame.shape[0]:
            cv2.circle(overlay, (int(x), int(y)), 5, (0,0,255), -1)
            point_count += 1

print(f"✅ Skapade överläggsbild med {point_count} synliga punkter.")
if point_count == 0:
    print("⚠️ VARNING: Inga punkter ritades. Detta betyder att din homografi fortfarande är felaktig.")

cv2.namedWindow("Homography Overlay", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Homography Overlay", 1280, 720)
cv2.imshow("Homography Overlay", overlay)
print("Visar bild: Röda prickar = projicerade trajektoriepunkter. Tryck valfri tangent för att stänga.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Plotta top-down-vy (i Pixelkoordinater) ---
plt.figure(figsize=(12,9))
for typ in df['Type'].unique():
    subset = df[df['Type']==typ]
    # Detta plottar de *slutgiltiga pixelkoordinaterna*
    plt.scatter(subset['x_hom'], subset['y_hom'], label=typ, s=10, alpha=0.7)
    
plt.title("Trajektorier efter Homografi (i Pixelkoordinater)")
plt.xlabel("X [pixlar]")
plt.ylabel("Y [pixlar]")
plt.legend()
plt.axis('equal')
plt.grid(True)
# Invertera Y-axeln för att matcha bildkoordinater (0,0 uppe till vänster)
plt.gca().invert_yaxis()
print("Visar trajektorieplott. Stäng plottfönstret för att avsluta.")
plt.show()

print("✅ Skriptet är färdigt.")