# src/cloud_verify.py

from google.cloud import storage

# --- REPLACE THIS WITH YOUR EXACT BUCKET NAME ---
BUCKET_NAME = "site-sentinel-roadwork-data" 
# -----------------------------------------------

def verify_local_access():
    """Checks if the local Python environment can read the bucket contents."""
    print("--- Testing Google Cloud Storage Access ---")
    
    try:
        # storage.Client() automatically uses the Application Default Credentials (ADC)
        storage_client = storage.Client()
        
        # We list objects in the 'raw/' subfolder where the large ZIP files were uploaded
        # If you uploaded them directly to the root, remove 'prefix="raw/"'
        blobs = storage_client.list_blobs(BUCKET_NAME, prefix="raw/")
        
        file_count = 0
        for blob in blobs:
            print(f"File found: {blob.name}")
            file_count += 1
        
        if file_count >= 4:
            print("\nSUCCESS! 🚀 Local Python is fully authenticated and data is accessible.")
        elif file_count > 0:
            print(f"\nSUCCESS! Found {file_count} files. Authentication works.")
        else:
            print("\nERROR: Bucket accessed, but no files found in the 'raw/' prefix. Check upload path.")

    except Exception as e:
        print(f"\nAUTHENTICATION FAILED. Error: {e}")
        print("ACTION: Ensure your bucket name is correct and run 'gcloud auth application-default login' again.")


if __name__ == "__main__":
    verify_local_access()