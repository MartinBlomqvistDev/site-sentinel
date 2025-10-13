# scripts/4b_train_lstm.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import time

# --- CONFIGURATION ---
TOP_EVENTS_CSV = "data/analysis_results/top_20_predictive_events.csv"
OUTPUT_DIR = "data/analysis_results"
ROOT_DIRECTORY = "C:/DS24/Site_Sentinel/data/raw/CONCOR-D/3W_SidStP_Trajectory"
OUTPUT_MODEL_PATH = "models/lstm_risk_predictor_tuned.keras"

# --- MODEL PARAMETERS ---
RISK_DISTANCE_THRESHOLD = 2.0
PREDICTION_LEAD_TIME_S = 4.0
FRAME_RATE = 25
SELECTED_EVENT_RANK = 1

# --- LSTM-SPECIFIC PARAMETERS ---
SEQUENCE_LENGTH = 25
EPOCHS = 20
BATCH_SIZE = 32
# ---------------------

def final_parser_v5(filepath):
    """
    The definitive parser for CONCOR-D CSV files.
    """
    all_rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                parts = line.split(';', 12)
                if len(parts) < 13: continue
                meta_parts, trajectory_blob = parts[:12], parts[12]
                track_id, object_type = int(meta_parts[0]), meta_parts[1].strip()
                trajectory_points = trajectory_blob.split(';')
                for i in range(0, len(trajectory_points), 7):
                    chunk = trajectory_points[i:i + 7]
                    if len(chunk) == 7:
                        all_rows.append({
                            'trackId': track_id, 'class': object_type, 'x': float(chunk[0]),
                            'y': float(chunk[1]), 'time': float(chunk[5])
                        })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(all_rows)

def calculate_motion_features(df):
    """
    Calculates velocity and acceleration for each object track.
    """
    print("Calculating motion features...")
    df = df.sort_values(by=['trackId', 'time']).reset_index(drop=True)
    
    df['delta_t'] = df.groupby('trackId')['time'].diff()
    df['delta_x'] = df.groupby('trackId')['x'].diff()
    df['delta_y'] = df.groupby('trackId')['y'].diff()
    
    df['velocity_x'] = (df['delta_x'] / df['delta_t']).replace([np.inf, -np.inf], 0).fillna(0)
    df['velocity_y'] = (df['delta_y'] / df['delta_t']).replace([np.inf, -np.inf], 0).fillna(0)
    df['speed_ms'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    
    df['delta_speed'] = df.groupby('trackId')['speed_ms'].diff()
    df['accel_ms2'] = (df['delta_speed'] / df['delta_t']).replace([np.inf, -np.inf], 0).fillna(0)
    
    final_cols = ['time', 'trackId', 'class', 'x', 'y', 'velocity_x', 'velocity_y', 'speed_ms', 'accel_ms2']
    return df[final_cols]

def calculate_advanced_features(df, target_actors):
    """
    Calculates relative features (distance, speed) and Time-to-Collision (TTC).
    """
    print("Calculating relative and advanced features (TTC)...")
    vulnerable_df = df[df['trackId'] == target_actors['vulnerable_id']].copy()
    car_df = df[df['trackId'] == target_actors['car_id']].copy()

    merged_df = pd.merge(vulnerable_df, car_df, on='time', suffixes=('_vuln', '_car'), how='outer')
    merged_df = merged_df.sort_values('time').ffill().dropna()

    merged_df['rel_distance'] = np.sqrt((merged_df['x_vuln'] - merged_df['x_car'])**2 + (merged_df['y_vuln'] - merged_df['y_car'])**2)
    merged_df['rel_speed'] = np.sqrt((merged_df['velocity_x_vuln'] - merged_df['velocity_x_car'])**2 + (merged_df['velocity_y_vuln'] - merged_df['velocity_y_car'])**2)

    delta_x = merged_df['x_car'] - merged_df['x_vuln']
    delta_y = merged_df['y_car'] - merged_df['y_vuln']
    delta_vx = merged_df['velocity_x_car'] - merged_df['velocity_x_vuln']
    delta_vy = merged_df['velocity_y_car'] - merged_df['velocity_y_vuln']

    dot_product_dv_dv = delta_vx**2 + delta_vy**2
    dot_product_dx_dv = delta_x * delta_vx + delta_y * delta_vy
    dot_product_dv_dv = dot_product_dv_dv.replace(0, np.nan)

    ttc = -dot_product_dx_dv / dot_product_dv_dv
    merged_df['ttc'] = ttc.where(ttc > 0, np.nan).fillna(100)

    return merged_df

def create_target_variable(df):
    """Creates the predictive target variable (Y)."""
    print(f"Creating target variable (Y)...")
    lead_frames = int(PREDICTION_LEAD_TIME_S * FRAME_RATE)
    df['immediate_risk'] = (df['rel_distance'] <= RISK_DISTANCE_THRESHOLD).astype(int)
    df['Y_risk_in_future'] = df['immediate_risk'].rolling(window=lead_frames, min_periods=1).max().shift(-lead_frames).fillna(0)
    return df

def main():
    """Main function to orchestrate the LSTM model training and cross-validation process."""
    print("--- STEP 4b (ALL-IN): LSTM Training with Data Scaling & Advanced Features ---")
    
    top_events = pd.read_csv(TOP_EVENTS_CSV)
    selected_event_info = top_events.iloc[SELECTED_EVENT_RANK]
    
    filename_parts = selected_event_info['file'].split('_')
    date_time_folder = f"{filename_parts[0]}_{filename_parts[1]}"
    source_csv_path = os.path.join(ROOT_DIRECTORY, date_time_folder, selected_event_info['file'])
    
    # Define paths
    base_name = os.path.splitext(selected_event_info['file'])[0]
    features_filename = f"features_{base_name}.csv"
    features_path = os.path.join(OUTPUT_DIR, features_filename)
    advanced_features_filename = f"advanced_features_{base_name}.csv"
    advanced_features_path = os.path.join(OUTPUT_DIR, advanced_features_filename)

    if not os.path.exists(advanced_features_path):
        print(f"Advanced feature file not found. Generating it now...")
        if not os.path.exists(features_path):
            print(f"Base feature file not found. Parsing and creating it...")
            clean_df = final_parser_v5(source_csv_path)
            features_df = calculate_motion_features(clean_df)
            features_df.to_csv(features_path, index=False)
        else:
            print(f"Loading base features from '{features_path}'...")
            features_df = pd.read_csv(features_path)
        
        target_actors = {'vulnerable_id': selected_event_info['trackId_vulnerable'], 'car_id': selected_event_info['trackId_car']}
        advanced_df = calculate_advanced_features(features_df, target_actors)
        advanced_df.to_csv(advanced_features_path, index=False)
        print(f"   -> Advanced features saved to '{advanced_features_path}'")
    else:
        print(f"Loading existing advanced feature set from '{advanced_features_path}'...")
        advanced_df = pd.read_csv(advanced_features_path)

    model_df = create_target_variable(advanced_df)
    
    features_to_use = ['rel_distance', 'rel_speed', 'speed_ms_vuln', 'speed_ms_car', 'accel_ms2_vuln', 'accel_ms2_car', 'ttc']
    X_data = model_df[features_to_use].fillna(100)
    Y_data = model_df.loc[X_data.index, 'Y_risk_in_future']
    
    if Y_data.sum() == 0:
        print("\nWARNING: No positive risk labels were generated.")
        return

    print(f"\nData prepared for sequencing. Found {int(Y_data.sum())} positive risk labels out of {len(Y_data)} total.")

    print("\n--- Starting 5-Fold Stratified Cross-Validation for LSTM ---")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    precision_scores, recall_scores, f1_scores = [], [], []

    for fold, (train_index, test_index) in enumerate(kf.split(X_data, Y_data)):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        
        X_train_fold, X_test_fold = X_data.iloc[train_index], X_data.iloc[test_index]
        Y_train_fold, Y_test_fold = Y_data.iloc[train_index], Y_data.iloc[test_index]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        train_generator = TimeseriesGenerator(X_train_scaled, Y_train_fold.to_numpy(), length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
        test_generator = TimeseriesGenerator(X_test_scaled, Y_test_fold.to_numpy(), length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
        
        Y_test_actual = Y_test_fold.iloc[SEQUENCE_LENGTH:]

        if len(Y_test_actual) == 0: continue

        model = Sequential([
            LSTM(units=64, activation='tanh', input_shape=(SEQUENCE_LENGTH, len(features_to_use)), return_sequences=True),
            Dropout(0.3),
            LSTM(units=32, activation='tanh'),
            Dropout(0.3),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        class_weight = {0: 1.0, 1: (len(Y_data) - Y_data.sum()) / Y_data.sum()}
        
        model.fit(train_generator, epochs=EPOCHS, verbose=1, class_weight=class_weight, shuffle=False)
        
        preds_proba = model.predict(test_generator)
        preds = (preds_proba > 0.5).astype(int)

        p_score = precision_score(Y_test_actual, preds, zero_division=0)
        r_score = recall_score(Y_test_actual, preds, zero_division=0)
        f1 = f1_score(Y_test_actual, preds, zero_division=0)
        precision_scores.append(p_score); recall_scores.append(r_score); f1_scores.append(f1)
        print(f"Fold {fold + 1} -> Precision: {p_score:.4f}, Recall: {r_score:.4f}, F1-Score: {f1:.4f}")

    print("\n--- LSTM Cross-Validation Results Summary ---")
    print(f"Average Precision: {np.mean(precision_scores):.4f} (+/- {np.std(precision_scores):.4f})")
    print(f"Average Recall:    {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
    print(f"Average F1-Score:  {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

if __name__ == "__main__":
    main()