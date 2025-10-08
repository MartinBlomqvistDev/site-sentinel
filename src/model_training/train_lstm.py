import numpy as np
import os
import joblib 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# --- CONFIGURATION ---
INPUT_NPZ_FILENAME = "lstm_sequence_data.npz"
MODEL_OUTPUT_FILENAME = "lstm_risk_predictor.h5" # Keras uses .h5 format

# Sequence parameters (must match the preparation script)
SEQUENCE_LENGTH = 15  # 15 frames of history
NUM_FEATURES = 5      # 5 features (TTC, speed_ms, etc.)
# ---------------------

def get_absolute_path(relative_path_from_root):
    """Helper function to construct absolute path from the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return os.path.join(project_root, relative_path_from_root)

def load_sequence_data():
    """Loads prepared sequences (X) and targets (Y) from NPZ file."""
    
    INPUT_FULL_PATH = get_absolute_path(os.path.join('data', 'logs', INPUT_NPZ_FILENAME))
    
    print(f"Loading sequence data from {INPUT_FULL_PATH}...")
    try:
        data = np.load(INPUT_FULL_PATH)
        X = data['X']
        Y = data['Y']
    except FileNotFoundError:
        print(f"FATAL ERROR: Sequence data not found. Ensure P34 preparation script ran successfully.")
        return None, None
        
    return X, Y

def build_lstm_model(input_shape):
    """Defines the sequential LSTM network architecture."""
    
    model = Sequential()
    
    # Layer 1: LSTM layer to process the sequences (return_sequences=False for classification)
    model.add(LSTM(units=50, activation='relu', input_shape=input_shape))
    
    # Dropout layer to prevent overfitting
    model.add(Dropout(0.2))
    
    # Layer 2: Classification layer (Dense output with sigmoid activation for binary classification)
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile the model
    # Using Adam optimizer and Binary Crossentropy for the binary risk prediction task
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_save_lstm():
    """Trains the LSTM model and saves the weights."""
    
    X, Y = load_sequence_data()
    if X is None:
        return

    print(f"Dataset shape: X={X.shape}, Y={Y.shape}")

    # Split data into training and testing sets (standard split for Deep Learning)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42, stratify=Y
    )
    
    # Determine input shape for the model: (sequence_length, num_features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Calculate class weight to handle imbalance (CRITICAL for risk prediction)
    # This weighs the risk class (Y=1) much higher than the safe class (Y=0)
    num_safe = np.sum(Y_train == 0)
    num_risk = np.sum(Y_train == 1)
    class_weight = {0: 1., 1: num_safe / num_risk} 
    
    print(f"Training LSTM for {len(X_train)} samples (Risk weight: 1.0/{num_safe/num_risk:.2f})")

    # Build and Train the Model
    model = build_lstm_model(input_shape)
    
    history = model.fit(
        X_train, Y_train,
        epochs=20, # Number of passes over the data (adjust based on time/results)
        batch_size=64,
        validation_data=(X_test, Y_test),
        class_weight=class_weight, # Apply the imbalance correction
        verbose=1
    )
    
    # Step 3: Evaluation and Saving
    
    # Predict probabilities and convert to binary classes (0 or 1)
    Y_pred_prob = model.predict(X_test)
    Y_pred = (Y_pred_prob > 0.5).astype(int)
    
    print("\n--- LSTM Model Evaluation ---")
    print(f"Test Precision: {precision_score(Y_test, Y_pred):.4f}")
    print(f"Test Recall: {recall_score(Y_test, Y_pred):.4f}")
    print(f"Test F1-Score: {f1_score(Y_test, Y_pred):.4f}")
    
    # Save the model
    MODEL_SAVE_PATH = get_absolute_path(os.path.join('models', MODEL_OUTPUT_FILENAME))
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    
    print(f"\nLSTM Model saved successfully to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_and_save_lstm()