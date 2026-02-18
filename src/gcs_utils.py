"""
Google Cloud Storage client for Site Sentinel.

Credentials are read from Streamlit secrets (when deployed on Streamlit Cloud)
or from environment variables (for local development). The module degrades
gracefully if neither is configured — GCS_ENABLED will be False and all
functions return safe empty values.

Note: the demo video is served from a public GCS URL and does not require
authentication. This module is available for private-bucket access if needed
in future (e.g. downloading updated model weights at runtime).

Required secret structure (in .streamlit/secrets.toml or Streamlit Cloud):

    [gcs_connections.gcs]
    credentials_json = "<service account JSON as a string>"
    bucket_name      = "<your-bucket>"
    project_id       = "<your-project>"   # optional but recommended
"""

from __future__ import annotations

import json
import logging
import os
import traceback

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialisation — runs once at import time
# ---------------------------------------------------------------------------

GCS_ENABLED: bool = False
_storage_client: storage.Client | None = None
_bucket: storage.Bucket | None = None
_bucket_name: str | None = None

try:
    gcs_config = st.secrets.get("gcs_connections", {}).get("gcs", {})

    creds_json = gcs_config.get("credentials_json") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS_JSON_STRING"
    )
    _bucket_name = gcs_config.get("bucket_name") or os.environ.get("GCS_BUCKET_NAME")
    project_id = gcs_config.get("project_id") or os.environ.get("GCP_PROJECT_ID")

    if not creds_json or not _bucket_name:
        raise ValueError(
            "GCS not configured: credentials_json and bucket_name are both required. "
            "Add them to .streamlit/secrets.toml or as environment variables."
        )

    credentials_info = json.loads(creds_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_info)

    _storage_client = (
        storage.Client(credentials=credentials, project=project_id)
        if project_id
        else storage.Client(credentials=credentials)
    )
    _bucket = _storage_client.bucket(_bucket_name)
    GCS_ENABLED = True
    logger.info("GCS client initialised (bucket: %s)", _bucket_name)

except ValueError as exc:
    logger.info("GCS disabled: %s", exc)
except Exception:
    logger.warning("GCS initialisation failed:\n%s", traceback.format_exc())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return True if the GCS client initialised successfully."""
    return GCS_ENABLED


def download_bytes(blob_name: str) -> bytes | None:
    """
    Download a file from the configured GCS bucket and return its raw bytes.

    Args:
        blob_name: Path to the object within the bucket (e.g. "models/rf.pkl").

    Returns:
        File contents as bytes, or None if GCS is disabled or the blob does
        not exist.
    """
    if not GCS_ENABLED or _bucket is None:
        return None

    blob = _bucket.blob(blob_name)
    if not blob.exists():
        logger.warning("GCS blob not found: %s", blob_name)
        return None

    try:
        data = blob.download_as_bytes()
        logger.debug("Downloaded %d bytes from gs://%s/%s", len(data), _bucket_name, blob_name)
        return data
    except Exception:
        logger.error("Failed to download gs://%s/%s:\n%s", _bucket_name, blob_name, traceback.format_exc())
        return None


def list_blobs(prefix: str = "") -> list[str]:
    """
    List object names in the bucket, optionally filtered by a path prefix.

    Returns an empty list if GCS is disabled or the listing fails.
    """
    if not GCS_ENABLED or _storage_client is None or _bucket_name is None:
        return []

    try:
        return [b.name for b in _storage_client.list_blobs(_bucket_name, prefix=prefix)]
    except Exception:
        logger.error("Failed to list GCS blobs:\n%s", traceback.format_exc())
        return []
