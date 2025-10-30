# gcs_utils.py
import os
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
import json
import io

# --- Configuration (from Streamlit Secrets or Environment Variables) ---
# In your Streamlit Cloud secrets, set:
# [gcs_connections.gcs]
# credentials_json = "..." (The full JSON key content as a string)
# bucket_name = "your-gcs-bucket-name"
# project_id = "your-gcp-project-id" (Optional but good practice)

try:
    # Attempt to load credentials from Streamlit secrets
    gcs_config = st.secrets.get("gcs_connections", {}).get("gcs", {})
    gcs_creds_json = gcs_config.get("credentials_json")
    bucket_name = gcs_config.get("bucket_name")
    project_id = gcs_config.get("project_id") # Get project_id

    # Fallback to environment variables
    if not gcs_creds_json:
        gcs_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON_STRING")
        print("GCS: Loaded credentials from env var")
    if not bucket_name:
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        print("GCS: Loaded bucket_name from env var")
    if not project_id:
        project_id = os.environ.get("GCP_PROJECT_ID") # Get project_id from env var
        print("GCS: Loaded project_id from env var")

    if not gcs_creds_json or not bucket_name:
        raise ValueError("GCS credentials_json or bucket_name not configured.")

    # Parse the JSON string credentials
    credentials_info = json.loads(gcs_creds_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    
    # Pass project_id to client if it exists (fixes some auth issues)
    if project_id:
        storage_client = storage.Client(credentials=credentials, project=project_id)
    else:
        storage_client = storage.Client(credentials=credentials)
        
    bucket = storage_client.bucket(bucket_name)
    print(f"✅ Successfully connected to GCS bucket: {bucket_name}")
    GCS_ENABLED = True

except (json.JSONDecodeError, ValueError, Exception) as e:
    print(f"⚠️ GCS Initialization Error: {e}")
    print("GCS functionality will be disabled. Check secrets/env vars.")
    storage_client = None
    bucket = None
    bucket_name = None # Ensure bucket_name is None if connection failed
    GCS_ENABLED = False


def check_gcs_connection():
    """Checks if GCS connection is established."""
    return GCS_ENABLED

def download_bytes_from_gcs(source_blob_name):
    """
    Downloads a file from GCS as bytes (useful for models, videos, numpy arrays).

    Args:
        source_blob_name (str): Path/name of the file in the GCS bucket.

    Returns:
        bytes or None: File content as bytes if successful, None otherwise.
    """
    if not GCS_ENABLED or not bucket:
        print("GCS is not enabled. Cannot download bytes.")
        return None
    try:
        blob = bucket.blob(source_blob_name)
        file_bytes = blob.download_as_bytes()
        print(f"✅ File gs://{bucket_name}/{source_blob_name} downloaded as bytes ({len(file_bytes)} bytes).")
        return file_bytes
    except Exception as e:
        print(f"❌ Error downloading bytes for {source_blob_name} from GCS: {e}")
        return None

def list_gcs_files(prefix=""):
    """Lists files in the GCS bucket, optionally filtering by prefix."""
    if not GCS_ENABLED or not storage_client or not bucket_name:
        print("GCS is not enabled. Cannot list files.")
        return []
    try:
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
        return [blob.name for blob in blobs]
    except Exception as e:
        print(f"❌ Error listing files in GCS bucket: {e}")
        return []

# Add other functions like upload_to_gcs if needed
