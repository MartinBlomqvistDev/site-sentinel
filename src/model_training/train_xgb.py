import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib 
import os 
import numpy as np

# --- CONFIGURATION ---
INPUT_FEATURES_FILENAME = "features_engineered.csv"
MODEL_OUTPUT_FILENAME = "xgb_risk_predictor.pkl"
LSTM_TEMP_FILENAME = "lstm_features_with_y.csv" # <-- FIL FÖR P34 ATT LÄSA

PREDICTION_LEAD_TIME_SECONDS = 4
FRAME_RATE = 10 # NGSIM data runs at 10 FPS (10 Hz)
# ---------------------

def get_absolute_path(relative_path_from_root):
    """Helper function to construct absolute path from the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return os.path.join(project_root, relative_path_from_root)

def create_target_variable(df):
    """
    Creates the Target Variable (Y): Flags if an object will be near a hazard.
    (Includes fix for the Pandas MultiIndex error.)
    """
    
    df['immediate_risk'] = df['risk_zone_status'] 
    
    lead_frames = int(PREDICTION_LEAD_TIME_SECONDS * FRAME_RATE)
    
    # 1. Calculate the rolling series
    risk_prediction_series = df.groupby('object_id')['immediate_risk'] \
                                .rolling(window=lead_frames, closed='right', min_periods=1) \
                                .max().shift(-lead_frames).fillna(0)
                                
    # 2. CRITICAL FIX: Drop the MultiIndex level to align with the main DataFrame
    risk_prediction_series = risk_prediction_series.droplevel(0) 

    # 3. Assign the calculated series back
    df['Y_PREDICT_RISK'] = risk_prediction_series.astype(int)
    
    return df

def train_and_save_model():
    """Loads features, performs K-Fold CV, trains the final model, and saves results for P34."""
    
    # --- Resolve ABSOLUTE Input/Output Paths ---
    INPUT_FULL_PATH = get_absolute_path(os.path.join('data', 'logs', INPUT_FEATURES_FILENAME))
    MODEL_SAVE_PATH = get_absolute_path(os.path.join('models', MODEL_OUTPUT_FILENAME))
    LSTM_TEMP_PATH = get_absolute_path(os.path.join('data', 'logs', LSTM_TEMP_FILENAME)) 
    # -------------------------------------------
    
    print(f"Loading engineered features from {INPUT_FULL_PATH}...")
    try:
        df = pd.read_csv(INPUT_FULL_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Feature CSV not found. Checked path: {INPUT_FULL_PATH}")
        return

    # Step 1: Create the target variable (Y)
    df = create_target_variable(df)
    
    # --- CRITICAL ADDITION: Save for LSTM (P34) ---
    # Save the DataFrame with the new Y column before filtering, so P34 can read it.
    os.makedirs(os.path.dirname(LSTM_TEMP_PATH), exist_ok=True)
    df.to_csv(LSTM_TEMP_PATH, index=False)
    print(f"Target variable saved for LSTM input to {LSTM_TEMP_FILENAME}")
    # ----------------------------------------------
    
    # Filter data for XGBoost training
    df = df[df['object_class'] == 'personnel'].dropna(subset=['min_dist_to_hazard', 'ttc_estimate', 'speed_ms', 'accel_ms2'])
    
    features = ['min_dist_to_hazard', 'rel_speed_to_hazard', 'ttc_estimate', 'speed_ms', 'accel_ms2'] 
    
    X = df[features]
    Y = df['Y_PREDICT_RISK']
    
    # Final Check for data sparsity
    if df.empty or len(Y.unique()) < 2:
        print("\nFATAL ERROR: Dataset is too sparse or lacks positive risk examples after filtering.")
        return

    # --- IMPLEMENTATION OF 5-FOLD STRATIFIED CROSS-VALIDATION (P33) ---
    # ... (K-Fold CV logic remains the same as previously defined) ...
    print("\n--- Starting 5-Fold Stratified Cross-Validation ---")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    precision_scores = []
    recall_scores = []
    
    scale_pos_weight = len(Y[Y == 0]) / len(Y[Y == 1]) 
    
    base_model = XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight
    )
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X, Y)):
        print(f"Training Fold {fold_idx + 1}/5...")
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        fold_model = base_model
        fold_model.fit(X_train, Y_train)
        
        Y_pred = fold_model.predict(X_test)
        
        precision_scores.append(precision_score(Y_test, Y_pred, zero_division=0))
        recall_scores.append(recall_score(Y_test, Y_pred, zero_division=0))

    # Step 3 & 4: Train Final Model and Report
    print("\nTraining Final Model on Full Dataset for Deployment...")
    final_model = base_model
    final_model.fit(X, Y)

    print("\n--- Model Evaluation (5-Fold Results) ---")
    print(f"Average Precision (CRITICAL): {np.mean(precision_scores):.4f} (Std Dev: {np.std(precision_scores):.4f})")
    print(f"Average Recall: {np.mean(recall_scores):.4f} (Std Dev: {np.std(recall_scores):.4f})")
    print(f"Final F1-Score (Trained on Full Data): {f1_score(Y, final_model.predict(X), zero_division=0):.4f}")


    # Step 5: Save the final model artifact
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"\nFinal Model (Full Data) saved successfully to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_and_save_model()