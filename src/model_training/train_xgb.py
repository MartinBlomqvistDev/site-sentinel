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
    Creates the Target Variable (Y): Flags if an object will be near a hazard 
    within PREDICTION_LEAD_TIME_SECONDS. 
    """
    
    # 1. Flag the risk observed in the CURRENT moment
    df['immediate_risk'] = df['risk_zone_status'] 
    
    # 2. Calculate the number of future frames to look ahead (4 seconds * 10 FPS = 40 frames)
    lead_frames = int(PREDICTION_LEAD_TIME_SECONDS * FRAME_RATE)
    
    # 3. Create the Y_PREDICT_RISK column:
    # Logic looks 'lead_frames' into the future to see if 'immediate_risk' becomes 1.
    risk_prediction_series = df.groupby('object_id')['immediate_risk'] \
                                .rolling(window=lead_frames, closed='right', min_periods=1) \
                                .max().shift(-lead_frames).fillna(0)
                                
    # CRITICAL FIX: Drop the 'object_id' level from the MultiIndex to align with the main DataFrame
    risk_prediction_series = risk_prediction_series.droplevel(0) 

    # Assign the calculated series back to the DataFrame
    df['Y_PREDICT_RISK'] = risk_prediction_series.astype(int)
    
    return df

def train_and_save_model():
    """Loads features, performs K-Fold Cross-Validation, trains the final model, and saves the result."""
    
    # --- Resolve ABSOLUTE Input/Output Paths ---
    INPUT_FULL_PATH = get_absolute_path(os.path.join('data', 'logs', INPUT_FEATURES_FILENAME))
    MODEL_SAVE_PATH = get_absolute_path(os.path.join('models', MODEL_OUTPUT_FILENAME))
    # -------------------------------------------
    
    print(f"Loading engineered features from {INPUT_FULL_PATH}...")
    try:
        df = pd.read_csv(INPUT_FULL_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Feature CSV not found. Checked path: {INPUT_FULL_PATH}")
        return

    # Step 1: Create the target variable (Y)
    df = create_target_variable(df)
    
    # CRITICAL FILTERING: Filter for 'personnel' and drop NaNs introduced by Feature Engineering
    df = df[df['object_class'] == 'personnel'].dropna(subset=['min_dist_to_hazard', 'ttc_estimate'])
    
    # Step 2: Prepare data for XGBoost (CORRECTED COLUMN NAMES)
    features = ['min_dist_to_hazard', 'rel_speed_to_hazard', 'ttc_estimate', 'speed_ms', 'accel_ms2'] 
    
    # Check for empty or single-class dataset after filtering
    if df.empty or len(df['Y_PREDICT_RISK'].unique()) < 2:
        print("\nFATAL ERROR: Dataset is too sparse or lacks positive risk examples after filtering.")
        print(f"Rows remaining: {len(df)}. Unique Target Values: {df['Y_PREDICT_RISK'].nunique()}")
        return

    X = df[features]
    Y = df['Y_PREDICT_RISK']
    
    # --- IMPLEMENTATION OF 5-FOLD STRATIFIED CROSS-VALIDATION (P33) ---
    print("\n--- Starting 5-Fold Stratified Cross-Validation ---")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    precision_scores = []
    recall_scores = []
    
    # Calculate scale_pos_weight based on the full dataset (safer for CV loops)
    scale_pos_weight = len(Y[Y == 0]) / len(Y[Y == 1]) 
    
    base_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight
    )
    
    # Loop through each fold for training and scoring
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X, Y)):
        print(f"Training Fold {fold_idx + 1}/5...")
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        fold_model = base_model
        fold_model.fit(X_train, Y_train)
        
        Y_pred = fold_model.predict(X_test)
        
        # Store scores (Precision is the most critical metric for safety)
        precision_scores.append(precision_score(Y_test, Y_pred, zero_division=0))
        recall_scores.append(recall_score(Y_test, Y_pred, zero_division=0))

    # Step 3: Train Final Model on Full Dataset
    print("\nTraining Final Model on Full Dataset for Deployment...")
    final_model = base_model
    final_model.fit(X, Y)

    # Step 4: Final Reporting (P33)
    
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