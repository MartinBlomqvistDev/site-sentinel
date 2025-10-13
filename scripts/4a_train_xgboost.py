# scripts/4a_train_xgboost.py

import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# --- CONFIGURATION ---
TOP_EVENTS_CSV = "data/analysis_results/top_20_predictive_events.csv"
OUTPUT_DIR = "data/analysis_results"
OUTPUT_MODEL_PATH = "models/xgb_risk_predictor_tuned.pkl"

# Model parameters
RISK_DISTANCE_THRESHOLD = 2.0
PREDICTION_LEAD_TIME_S = 4.0
FRAME_RATE = 25
SELECTED_EVENT_RANK = 1
# ---------------------

def create_target_variable(df):
    """Creates the predictive target variable (Y)."""
    print(f"Creating target variable (Y)...")
    lead_frames = int(PREDICTION_LEAD_TIME_S * FRAME_RATE)
    df['immediate_risk'] = (df['rel_distance'] <= RISK_DISTANCE_THRESHOLD).astype(int)
    df['Y_risk_in_future'] = df['immediate_risk'].rolling(window=lead_frames, min_periods=1).max().shift(-lead_frames).fillna(0)
    return df

def main():
    """Main function to orchestrate the advanced model training and tuning."""
    print("--- STEP 4a (ALL-IN): XGBoost Tuning with SMOTE and Advanced Features ---")
    
    if not os.path.exists(TOP_EVENTS_CSV):
        print(f"ERROR: '{TOP_EVENTS_CSV}' not found. Run '2_event_analyzer.py' first.")
        return
        
    # --- CORRECTED LOGIC: Dynamically build the input filename ---
    top_events = pd.read_csv(TOP_EVENTS_CSV)
    selected_event_info = top_events.iloc[SELECTED_EVENT_RANK]
    base_name = os.path.splitext(selected_event_info['file'])[0]
    advanced_features_filename = f"advanced_features_{base_name}.csv"
    advanced_features_path = os.path.join(OUTPUT_DIR, advanced_features_filename)
    # -----------------------------------------------------------

    if not os.path.exists(advanced_features_path):
        print(f"ERROR: Advanced features file not found at '{advanced_features_path}'.")
        print("Please run '3_feature_eng.py' first.")
        return
        
    print(f"Loading advanced feature set from '{advanced_features_path}'...")
    model_df = pd.read_csv(advanced_features_path)
    
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
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        print("Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=42)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

        print("Running RandomizedSearchCV for hyperparameters...")
        xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
        random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=20, cv=3, scoring='f1', random_state=42, n_jobs=-1, verbose=1)
        random_search.fit(X_train_resampled, Y_train_resampled)
        
        best_model = random_search.best_estimator_
        print(f"Best params for this fold: {random_search.best_params_}")

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
    
    print("\nTraining final TUNED model on all data for deployment...")
    X_resampled_full, Y_resampled_full = SMOTE(random_state=42).fit_resample(X, Y)
    final_search = RandomizedSearchCV(XGBClassifier(objective='binary:logistic', eval_metric='logloss'), param_distributions=param_grid, n_iter=20, cv=3, scoring='f1', random_state=42, n_jobs=-1, verbose=1)
    final_search.fit(X_resampled_full, Y_resampled_full)
    final_model = final_search.best_estimator_

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    joblib.dump(final_model, OUTPUT_MODEL_PATH)
    
    print("\n✅ Final Tuned XGBoost model training complete!")
    print(f"Model saved to '{OUTPUT_MODEL_PATH}'")
    print(f"Best overall parameters found: {final_search.best_params_}")

if __name__ == "__main__":
    main()