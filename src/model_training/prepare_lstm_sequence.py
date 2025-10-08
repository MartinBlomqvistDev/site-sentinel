import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf # Required for future LSTM training

# --- CONFIGURATION ---
# Input is the engineered features CSV (Target variable must be present)
INPUT_FEATURES_FILENAME = "lstm_features_with_y.csv" 
OUTPUT_DATA_PATH = "lstm_sequence_data.npz"

# LSTM Sequence Parameters
SEQUENCE_LENGTH = 15  # 1.5 seconds history (15 frames at 10 FPS)
# ---------------------

def get_absolute_path(relative_path_from_root):
    """Helper function to construct absolute path from the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return os.path.join(project_root, relative_path_from_root)


def create_sequences(df, sequence_length):
    """
    Converts a DataFrame of object features into sequences for LSTM training.
    Output: X_sequences (3D array) and Y_target (1D array).
    """
    
    # Feature columns (MUST match the XGBoost feature set from P32)
    feature_cols = ['min_dist_to_hazard', 'rel_speed_to_hazard', 'ttc_estimate', 'speed_ms', 'accel_ms2']
    
    X_data, Y_data = [], []
    
    # CRITICAL FILTERING: Filter data to isolate 'personnel' and drop any remaining NaNs.
    df_filtered = df[df['object_class'] == 'personnel'].dropna(subset=feature_cols + ['Y_PREDICT_RISK']).copy()

    # Group by object_id to process each continuous trajectory separately
    for object_id, group in df_filtered.groupby('object_id'):
        
        # Select features and targets for the current object
        features = group[feature_cols].values
        # Target (risk status) created by the P32 script
        targets = group['Y_PREDICT_RISK'].values 
        
        # Normalize the features to the 0-1 range (CRITICAL for LSTMs)
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)
        
        # Create sequences: use 'sequence_length' frames to predict the outcome
        for i in range(len(features_scaled) - sequence_length):
            # X_seq: 15 frames of past features
            X_seq = features_scaled[i:(i + sequence_length), :] 
            # Y_target: The risk status of the frame immediately following the sequence
            Y_target = targets[i + sequence_length]
            
            X_data.append(X_seq)
            Y_data.append(Y_target)
            
    return np.array(X_data), np.array(Y_data)

def main():
    """Main function to load features, create sequences, and save as NPZ."""
    
    INPUT_FULL_PATH = get_absolute_path(os.path.join('data', 'logs', INPUT_FEATURES_FILENAME))
    OUTPUT_FULL_PATH = get_absolute_path(os.path.join('data', 'logs', OUTPUT_DATA_PATH))

    print(f"Loading engineered features for LSTM from {INPUT_FULL_PATH}...")
    try:
        df = pd.read_csv(INPUT_FULL_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Input CSV not found. Checked path: {INPUT_FULL_PATH}")
        return

    # NOTE: df must now contain the 'Y_PREDICT_RISK' column from P32 execution.
    
    X_sequences, Y_targets = create_sequences(df, SEQUENCE_LENGTH)

    print("\n--- Sequence Preparation Complete ---")
    print(f"Shape of X (Sequences): {X_sequences.shape}")
    print(f"Shape of Y (Targets): {Y_targets.shape}")
    
    if X_sequences.shape[0] == 0:
        print("ERROR: No sequences were created. Check sequence length or data filtering.")
        return

    # Step 2: Save the prepared sequences in compressed NPZ format
    os.makedirs(os.path.dirname(OUTPUT_FULL_PATH), exist_ok=True)
    np.savez_compressed(OUTPUT_FULL_PATH, X=X_sequences, Y=Y_targets)
    print(f"Data saved successfully to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    main()