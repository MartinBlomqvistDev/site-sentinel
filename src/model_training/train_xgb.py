import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib 

# --- CONFIGURATION ---
INPUT_FEATURES_PATH = "../data/logs/features_engineered.csv"
MODEL_OUTPUT_PATH = "../models/xgb_risk_predictor.pkl"

# How many seconds INTO THE FUTURE do we want to predict a risk?
# 4 seconds provides a robust lead time for human intervention.
PREDICTION_LEAD_TIME_SECONDS = 4
FRAME_RATE = 30 # Frames per second (must match T21)
# ---------------------

def create_target_variable(df):
    """
    Creates the Target Variable (Y): Flags if an object will be near a hazard 
    within PREDICTION_LEAD_TIME_SECONDS. 
    """
    
    # 1. Flag the risk observed in the CURRENT moment (used for lookahead)
    df['immediate_risk'] = df['risk_zone_status'] 
    
    # 2. Calculate the number of future frames we must look ahead
    lead_frames = int(PREDICTION_LEAD_TIME_SECONDS * FRAME_RATE)
    
    # 3. Create the Y_PREDICT_RISK column: 
    # This advanced logic looks 'lead_frames' into the future to see if 'immediate_risk' becomes 1.
    df['Y_PREDICT_RISK'] = df.groupby('object_id')['immediate_risk'] \
                                .rolling(window=lead_frames, closed='right', min_periods=1) \
                                .max().shift(-lead_frames).fillna(0)
    
    # Convert to integer for binary classification (0 or 1)
    df['Y_PREDICT_RISK'] = df['Y_PREDICT_RISK'].astype(int)
    
    return df

def train_and_save_model():
    """Loads features, trains the XGBoost model, and saves the result."""
    print(f"Loading engineered features from {INPUT_FEATURES_PATH}...")
    try:
        df = pd.read_csv(INPUT_FEATURES_PATH)
    except FileNotFoundError:
        print("ERROR: Feature CSV not found. Ensure P31 ran successfully.")
        return

    # Step 1: Create the target variable (Y)
    df = create_target_variable(df)
    
    # Filter only the rows corresponding to workers (the target of safety prediction)
    # CRITICAL: Drop rows with NaN values created by initial feature engineering (e.g., TTC)
    df = df[df['object_class'].isin(['person', 'worker'])].dropna()

    # Step 2: Prepare data for XGBoost
    # Include the powerful new TTC feature.
    features = ['min_dist_to_hazard', 'rel_speed_to_hazard', 'ttc_estimate', 'speed', 'accel']
    
    # Ensure X and Y are filtered identically to match indices after dropping NaNs
    X = df[features]
    Y = df['Y_PREDICT_RISK']
    
    # Split into training and testing sets (CRITICAL to avoid data leakage)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

    # Step 3: Train the XGBoost Model
    print("Starting XGBoost training...")
    # scale_pos_weight is crucial for handling severe class imbalance (many safe frames vs. few risky ones)
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1]) 
    )
    
    model.fit(X_train, Y_train)
    print("Training complete.")

    # Step 4: Evaluation (P33 Preview)
    Y_pred = model.predict(X_test)
    
    print("\n--- Model Evaluation (P33 Preview) ---")
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
    # Precision is the most crucial safety metric (minimizes false alarms)
    print(f"Precision: {precision_score(Y_test, Y_pred):.4f}") 
    print(f"Recall: {recall_score(Y_test, Y_pred):.4f}")
    print(f"F1-Score: {f1_score(Y_test, Y_pred):.4f}")


    # Step 5: Save the model artifact
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel saved successfully to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_and_save_model()