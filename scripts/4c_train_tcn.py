# scripts/4c_train_tcn.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tcn import TCN # Import the TCN layer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# --- CONFIGURATION ---
ADVANCED_FEATURES_CSV = "data/analysis_results/advanced_features_20220629_1530_Sid_StP_3W_d_1_18_ann.csv"
OUTPUT_MODEL_PATH = "models/tcn_risk_predictor.keras"

# --- MODEL PARAMETERS ---
RISK_DISTANCE_THRESHOLD = 2.0
PREDICTION_LEAD_TIME_S = 4.0
FRAME_RATE = 25

# --- TCN-SPECIFIC PARAMETERS ---
SEQUENCE_LENGTH = 25 # 1 second of history
EPOCHS = 15
BATCH_SIZE = 32
# ---------------------

def create_target_variable(df):
    """Creates the predictive target variable (Y)."""
    print(f"Creating target variable (Y)...")
    lead_frames = int(PREDICTION_LEAD_TIME_S * FRAME_RATE)
    df['immediate_risk'] = (df['rel_distance'] <= RISK_DISTANCE_THRESHOLD).astype(int)
    df['Y_risk_in_future'] = df['immediate_risk'].rolling(window=lead_frames, min_periods=1).max().shift(-lead_frames).fillna(0)
    return df

def main():
    """Main function to orchestrate the TCN model training and cross-validation process."""
    print("--- STEP 4c (ALL-IN): TCN Model Training ---")
    
    if not os.path.exists(ADVANCED_FEATURES_CSV):
        print(f"ERROR: Advanced features file not found at '{ADVANCED_FEATURES_CSV}'.")
        return
        
    print(f"Loading advanced feature set from '{ADVANCED_FEATURES_CSV}'...")
    model_df = pd.read_csv(ADVANCED_FEATURES_CSV)
    model_df = create_target_variable(model_df)
    
    features_to_use = ['rel_distance', 'rel_speed', 'speed_ms_vuln', 'speed_ms_car', 'accel_ms2_vuln', 'accel_ms2_car', 'ttc']
    X_data = model_df[features_to_use].fillna(100)
    Y_data = model_df.loc[X_data.index, 'Y_risk_in_future']
    
    if Y_data.sum() == 0: return

    print(f"\nData prepared for sequencing. Found {int(Y_data.sum())} positive risk labels out of {len(Y_data)} total.")

    print("\n--- Starting 5-Fold Stratified Cross-Validation for TCN ---")
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

        # --- NEW TCN MODEL ARCHITECTURE ---
        model = Sequential([
            TCN(
                input_shape=(SEQUENCE_LENGTH, len(features_to_use)),
                nb_filters=64,
                kernel_size=3,
                dilations=[1, 2, 4, 8],
                use_skip_connections=True,
                dropout_rate=0.2
            ),
            Dense(1, activation='sigmoid')
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

    print("\n--- TCN Cross-Validation Results Summary ---")
    print(f"Average Precision: {np.mean(precision_scores):.4f} (+/- {np.std(precision_scores):.4f})")
    print(f"Average Recall:    {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
    print(f"Average F1-Score:  {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

if __name__ == "__main__":
    main()