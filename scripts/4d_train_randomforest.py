# scripts/4d_train_randomforest.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# --- CONFIGURATION ---
ADVANCED_FEATURES_CSV = "data/analysis_results/master_training_dataset.csv"
OUTPUT_MODEL_PATH = "models/rf_master_predictor_tuned.pkl"

# Model parameters
RISK_DISTANCE_THRESHOLD = 2.0
PREDICTION_LEAD_TIME_S = 4.0
FRAME_RATE = 25
# ---------------------

def create_target_variable(df):
    """Creates the predictive target variable (Y)."""
    print(f"Creating target variable (Y)...")
    lead_frames = int(PREDICTION_LEAD_TIME_S * FRAME_RATE)
    df['immediate_risk'] = (df['rel_distance'] <= RISK_DISTANCE_THRESHOLD).astype(int)
    # This groupby is crucial for the master dataset to prevent windows from crossing between events
    df['Y_risk_in_future'] = df.groupby(['trackId_vuln', 'trackId_car'])['immediate_risk'].rolling(window=lead_frames, min_periods=1).max().shift(-lead_frames).fillna(0).reset_index(level=[0,1], drop=True)
    return df

def main():
    """Main function to orchestrate the Random Forest model training and tuning."""
    print("--- STEP 4d (ALL-IN): Random Forest Tuning with SMOTE and Advanced Features ---")
    
    if not os.path.exists(ADVANCED_FEATURES_CSV):
        print(f"ERROR: Advanced features file not found. Run '5_build_master_dataset.py' first.")
        return
        
    print(f"Loading advanced feature set from '{ADVANCED_FEATURES_CSV}'...")
    model_df = pd.read_csv(ADVANCED_FEATURES_CSV)
    
    model_df = create_target_variable(model_df)
    
    features_to_use = ['rel_distance', 'rel_speed', 'speed_ms_vuln', 'speed_ms_car', 'accel_ms2_vuln', 'accel_ms2_car', 'ttc']
    X = model_df[features_to_use].fillna(100)
    Y = model_df.loc[X.index, 'Y_risk_in_future']
    
    if Y.sum() == 0:
        print("\nWARNING: No positive risk labels were generated.")
        return

    print(f"\nData prepared for modeling. Found {int(Y.sum())} positive risk labels out of {len(Y)} total.")

    print("\n--- Starting 5-Fold CV with SMOTE and Hyperparameter Tuning ---")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    precision_scores, recall_scores, f1_scores = [], [], []

    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    best_params_from_cv = {}

    for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        print("Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=42)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

        print("Running RandomizedSearchCV for hyperparameters...")
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=20, cv=3, scoring='f1', random_state=42, n_jobs=-1, verbose=1)
        random_search.fit(X_train_resampled, Y_train_resampled)
        
        best_model = random_search.best_estimator_
        best_params_from_cv = random_search.best_params_
        print(f"Best params for this fold: {best_params_from_cv}")

        preds = best_model.predict(X_test)
        p_score = precision_score(Y_test, preds, zero_division=0)
        r_score = recall_score(Y_test, preds, zero_division=0)
        f1 = f1_score(Y_test, preds, zero_division=0)
        
        precision_scores.append(p_score); recall_scores.append(r_score); f1_scores.append(f1)
        print(f"Fold {fold + 1} -> Precision: {p_score:.4f}, Recall: {r_score:.4f}, F1-Score: {f1:.4f}")

    print("\n--- FINAL Cross-Validation Results Summary ---")
    print(f"Average Precision: {np.mean(precision_scores):.4f} (+/- {np.std(precision_scores):.4f})")
    print(f"Average Recall:    {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
    print(f"Average F1-Score:  {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    
    # Final model training and saving
    print("\nTraining final TUNED Random Forest model on all data for deployment...")
    
    # Re-sample the entire dataset for final training
    X_resampled_full, Y_resampled_full = SMOTE(random_state=42).fit_resample(X, Y)
    
    # Use the best parameters found during the last fold of CV as a good starting point
    final_model = RandomForestClassifier(random_state=42, class_weight='balanced', **best_params_from_cv)
    final_model.fit(X_resampled_full, Y_resampled_full)

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    joblib.dump(final_model, OUTPUT_MODEL_PATH)
    
    print("\n✅ Final Tuned Random Forest model training complete!")
    print(f"Model saved to '{OUTPUT_MODEL_PATH}'")
    print(f"Using final parameters: {best_params_from_cv}")

if __name__ == "__main__":
    main()