# src/model_training/yolo_finetune.py

from ultralytics import YOLO
import os

# --- CONFIGURATION ---
DATASET_CONFIG = "../site_sentinel.yaml"
MODEL_SIZE = "yolov8m.pt"  # Medium model size balances speed and accuracy
EPOCHS = 30                # Number of training runs (Adjust based on time/results)
# ---------------------

def run_finetuning():
    """Loads a pre-trained YOLOv8 model and fine-tunes it on Site Sentinel data."""
    
    # Resolve the path to the YAML file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, DATASET_CONFIG)

    print(f"Loading base model: {MODEL_SIZE}")
    # Load a pre-trained model (Transfer Learning approach)
    model = YOLO(MODEL_SIZE)  

    print(f"Starting Fine-Tuning using config: {yaml_path}")
    # Start training process
    results = model.train(
        data=yaml_path, 
        epochs=EPOCHS, 
        imgsz=640, # Standard image size for detection
        name='site_sentinel_yolo' # Folder name where results are saved
    )
    
    print("\n--- YOLOv8 Fine-Tuning Complete ---")
    
if __name__ == "__main__":
    # NOTE: You MUST extract and organize your data before running this script!
    run_finetuning()