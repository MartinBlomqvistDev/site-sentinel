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
MASTER_FEATURES_CSV = "data/analysis_results/master_training_dataset_full.csv"  # full master CSV
OUTPUT_MODEL_PATH = "models/rf_master_predictor_dual_lead_tuned.pkl"

FRAME_RATE = 29.97
PREDICTION_LEAD_TIME_S = 4.0  # <- correct preventive horizon
RISK_DISTANCE_THRESHOLD = 2.0
# ---------------------

def create_dual_lead_targets(df):
    """
    Creates both preventive (future, 4s ahead) and standard (immediate TTC) target variables.
    Maintains trackId pair grouping to avoid event leakage.
    """
    lead_frames = int(PREDICTION_LEAD_TIME_S * FRAME_RATE)
    
    # Preventive risk
    df['immediate_risk'] = (df['rel_distance'] <= RISK_DISTANCE_THRESHOLD).astype(int)
    df['Y_preventive'] = (
        df.groupby(['trackId_vuln','trackId_car'])['immediate_risk']
          .rolling(window=lead_frames, min_periods=1)
          .max()
          .shift(-lead_frames)
          .fillna(0)
          .reset_index(level=[0,1], drop=True)
          .astype(int)
    )

    # Standard risk (instant TTC danger)
    df['Y_standard'] = (df['ttc'] <= 2.0).astype(int)

    return df

def main():
    print("--- STEP 4d: Dual-Lead Random Forest Training (FULL Master Dataset) ---")

    if not os.path.exists(MASTER_FEATURES_CSV):
        print(f"ERROR: Master dataset file not found at '{MASTER_FEATURES_CSV}'. Build it first.")
        return

    df = pd.read_csv(MASTER_FEATURES_CSV)
    df = create_dual_lead_targets(df)

    # Feature selection (all advanced / preventive features)
    features = ['rel_distance','rel_speed','speed_ms_vuln','speed_ms_car',
                'accel_ms2_vuln','accel_ms2_car','ttc',
                'approach_speed','rel_dist_avg_2s','rel_speed_avg_2s','future_rel_dist_avg_2s']

    # Ensure all features exist
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)

    # Targets
    y_pre = df['Y_preventive']
    y_std = df['Y_standard']

    if y_pre.sum() == 0 or y_std.sum() == 0:
        print("ERROR: No positive labels detected. Check thresholds or master dataset.")
        return

    print(f"Data loaded: {len(df)} rows. Preventive positives: {y_pre.sum()}, Standard positives: {y_std.sum()}")

    # 5-fold Stratified CV
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'n_estimators': [100,200,300,400],
        'max_depth': [5,10,15,None],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1,2,4],
        'max_features': ['sqrt','log2']
    }

    def train_rf(X, y, label_name):
        print(f"\n--- Training {label_name} model ---")
        precision_scores, recall_scores, f1_scores = [], [], []
        best_params_fold = {}

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            print(f"\nFold {fold+1}/5")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            search = RandomizedSearchCV(
                rf, param_distributions=param_grid, n_iter=20,
                cv=3, scoring='f1', random_state=42, n_jobs=-1, verbose=0
            )
            search.fit(X_train_res, y_train_res)

            best_model = search.best_estimator_
            best_params_fold = search.best_params_
            preds = best_model.predict(X_test)

            p = precision_score(y_test, preds, zero_division=0)
            r = recall_score(y_test, preds, zero_division=0)
            f = f1_score(y_test, preds, zero_division=0)
            precision_scores.append(p); recall_scores.append(r); f1_scores.append(f)
            print(f"Fold {fold+1} -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
        
        print(f"\n{label_name} CV Summary -> Precision: {np.mean(precision_scores):.4f} (+/- {np.std(precision_scores):.4f})")
        print(f"{label_name} CV Summary -> Recall:    {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
        print(f"{label_name} CV Summary -> F1:        {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

        # Train final model on full resampled dataset
        X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
        final_rf = RandomForestClassifier(random_state=42, class_weight='balanced', **best_params_fold)
        final_rf.fit(X_res, y_res)

        return final_rf, best_params_fold

    # Train both models
    rf_pre, pre_params = train_rf(X, y_pre, "Preventive Risk (4s)")
    rf_std, std_params = train_rf(X, y_std, "Standard Risk (Instant TTC)")

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    joblib.dump({'preventive': rf_pre, 'standard': rf_std}, OUTPUT_MODEL_PATH)

    print("\n✅ Dual-Lead Random Forest models trained and saved!")
    print(f"Model path: {OUTPUT_MODEL_PATH}")
    print(f"Preventive params: {pre_params}")
    print(f"Standard params: {std_params}")

if __name__ == "__main__":
    main()
