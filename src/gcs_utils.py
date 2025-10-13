# src/gcs_utils.py

from google.cloud import storage
import os

# --- CONFIGURATION ---
BUCKET_NAME = "site-sentinel-roadwork-data"

# CRITICAL FIX: Explicitly define the Project ID here
# This ID must match the ID used during gcloud init
GCP_PROJECT_ID = "site-sentinel-project" 
# -----------------------------------------------------------

def upload_file_to_gcs(local_file_path, destination_blob_name):
    """Uploads a file (model or artifact) to GCS, explicitly passing the project ID."""
    try:
        # FIX: Pass the project ID directly to the storage client (solves the authentication error)
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        
        blob.upload_from_filename(local_file_path)
        print(f"GCS SUCCESS: File {local_file_path} uploaded to gs://{BUCKET_NAME}/{destination_blob_name}")
        return True
    except Exception as e:
        print(f"GCS ERROR: Failed to upload {local_file_path}. Details: {e}")
        return False

def download_model_from_gcs(source_blob_name, local_file_path):
    """Downloads the trained model for local testing or initial deployment setup."""
    try:
        # FIX: Pass the project ID directly to the storage client
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        blob.download_to_filename(local_file_path)
        print(f"GCS SUCCESS: Model downloaded to {local_file_path}")
        return True
    except Exception as e:
        print(f"GCS ERROR: Failed to download {source_blob_name}. Details: {e}")
        return False