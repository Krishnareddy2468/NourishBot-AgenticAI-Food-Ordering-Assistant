"""
Object storage abstraction for Dzukku vNext.

Supports: local filesystem, S3, GCS, Azure Blob.
Configure via STORAGE_PROVIDER env var.
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def _local_upload_dir() -> Path:
    """Return the local upload directory, creating it if needed."""
    d = settings.STORAGE_DIR / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def upload_image(file_data: bytes, filename: str, content_type: str = "image/jpeg") -> str:
    """
    Upload an image and return its URL.

    Args:
        file_data: Raw bytes of the image.
        filename: Original filename (used for extension).
        content_type: MIME type.

    Returns:
        URL string for the uploaded image.
    """
    ext = Path(filename).suffix or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"

    if settings.STORAGE_PROVIDER == "local":
        dest = _local_upload_dir() / unique_name
        dest.write_bytes(file_data)
        base = settings.STORAGE_BASE_URL or f"/storage/uploads"
        return f"{base}/{unique_name}"

    elif settings.STORAGE_PROVIDER == "s3":
        return await _upload_s3(file_data, unique_name, content_type)

    elif settings.STORAGE_PROVIDER == "gcs":
        return await _upload_gcs(file_data, unique_name, content_type)

    elif settings.STORAGE_PROVIDER == "azure":
        return await _upload_azure(file_data, unique_name, content_type)

    else:
        raise ValueError(f"Unknown STORAGE_PROVIDER: {settings.STORAGE_PROVIDER}")


async def delete_image(url: str) -> None:
    """Delete an image by its URL."""
    if settings.STORAGE_PROVIDER == "local":
        base = settings.STORAGE_BASE_URL or "/storage/uploads"
        if url.startswith(base):
            name = url.rsplit("/", 1)[-1]
            path = _local_upload_dir() / name
            if path.exists():
                path.unlink()
        return

    elif settings.STORAGE_PROVIDER == "s3":
        await _delete_s3(url)
    elif settings.STORAGE_PROVIDER == "gcs":
        await _delete_gcs(url)
    elif settings.STORAGE_PROVIDER == "azure":
        await _delete_azure(url)


# ── S3 implementation ────────────────────────────────────────────────────────

async def _upload_s3(file_data: bytes, key: str, content_type: str) -> str:
    try:
        import aioboto3
    except ImportError:
        raise ImportError("aioboto3 is required for S3 storage: pip install aioboto3")

    session = aioboto3.Session()
    async with session.client("s3") as s3:
        await s3.put_object(
            Bucket=settings.STORAGE_BUCKET,
            Key=key,
            Body=file_data,
            ContentType=content_type,
        )
    base = settings.STORAGE_BASE_URL or f"https://{settings.STORAGE_BUCKET}.s3.amazonaws.com"
    return f"{base}/{key}"


async def _delete_s3(url: str) -> None:
    try:
        import aioboto3
    except ImportError:
        return
    key = url.rsplit("/", 1)[-1]
    session = aioboto3.Session()
    async with session.client("s3") as s3:
        await s3.delete_object(Bucket=settings.STORAGE_BUCKET, Key=key)


# ── GCS implementation ───────────────────────────────────────────────────────

async def _upload_gcs(file_data: bytes, key: str, content_type: str) -> str:
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError:
        raise ImportError("google-cloud-storage is required for GCS storage")

    client = storage.Client()
    bucket = client.bucket(settings.STORAGE_BUCKET)
    blob = bucket.blob(key)
    blob.upload_from_string(file_data, content_type=content_type)
    return f"https://storage.googleapis.com/{settings.STORAGE_BUCKET}/{key}"


async def _delete_gcs(url: str) -> None:
    try:
        from google.cloud import storage
    except ImportError:
        return
    key = url.rsplit("/", 1)[-1]
    client = storage.Client()
    bucket = client.bucket(settings.STORAGE_BUCKET)
    bucket.blob(key).delete()


# ── Azure Blob implementation ────────────────────────────────────────────────

async def _upload_azure(file_data: bytes, key: str, content_type: str) -> str:
    try:
        from azure.storage.blob import ContainerClient
    except ImportError:
        raise ImportError("azure-storage-blob is required for Azure storage")

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    container = ContainerClient.from_connection_string(conn_str, container_name=settings.STORAGE_BUCKET)
    blob_client = container.get_blob_client(key)
    blob_client.upload_blob(file_data, overwrite=True, content_settings={"content_type": content_type})
    return f"{settings.STORAGE_BASE_URL}/{key}"


async def _delete_azure(url: str) -> None:
    try:
        from azure.storage.blob import ContainerClient
    except ImportError:
        return
    key = url.rsplit("/", 1)[-1]
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    container = ContainerClient.from_connection_string(conn_str, container_name=settings.STORAGE_BUCKET)
    container.get_blob_client(key).delete_blob()


import os  # noqa: E402 — needed for Azure conn str
