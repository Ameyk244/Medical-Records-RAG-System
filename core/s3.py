"""S3 wrapper — upload + bucket-init via aioboto3. Works against MinIO or AWS S3."""

from __future__ import annotations

import os

import aioboto3
from botocore.exceptions import ClientError


class S3Error(Exception):
    """Raised when an S3 operation fails. Wraps the underlying botocore error."""


_session: aioboto3.Session | None = None


def _get_session() -> aioboto3.Session:
    global _session
    if _session is None:
        _session = aioboto3.Session()
    return _session


def _client_kwargs() -> dict:
    access = os.environ.get("S3_ACCESS_KEY")
    secret = os.environ.get("S3_SECRET_KEY")
    if not access or not secret:
        raise S3Error("S3_ACCESS_KEY and S3_SECRET_KEY env vars are required")

    kwargs: dict = {
        "aws_access_key_id": access,
        "aws_secret_access_key": secret,
        "region_name": "us-east-1",  # MinIO ignores; AWS requires *something*
    }
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return kwargs


def _bucket() -> str:
    name = os.environ.get("S3_BUCKET")
    if not name:
        raise S3Error("S3_BUCKET env var is required")
    return name


async def upload_document(
    patient_id: str,
    document_id: str,
    filename: str,
    content: bytes,
) -> str:
    if not patient_id:
        raise ValueError("patient_id must be non-empty")
    if not document_id:
        raise ValueError("document_id must be non-empty")
    if not filename:
        raise ValueError("filename must be non-empty")
    if not content:
        raise ValueError("content must be non-empty")

    # Trust boundary: patient_id and document_id are app-controlled UUIDs;
    # filename comes from FastAPI UploadFile.filename (client-supplied — sanitize if
    # patient_id/document_id ever become user-controlled inputs).
    key = f"documents/{patient_id}/{document_id}/{filename}"

    session = _get_session()
    try:
        async with session.client("s3", **_client_kwargs()) as s3:
            await s3.put_object(Bucket=_bucket(), Key=key, Body=content)
    except ClientError as e:
        raise S3Error(f"failed to upload document to S3 (key={key})") from e

    return key


async def ensure_bucket_exists() -> None:
    session = _get_session()
    bucket = _bucket()
    async with session.client("s3", **_client_kwargs()) as s3:
        try:
            await s3.head_bucket(Bucket=bucket)
            return  # exists, nothing to do
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            # 404 means missing → create it. Any other error is a real problem.
            if code not in ("404", "NoSuchBucket", "NotFound"):
                raise S3Error(f"failed to check bucket {bucket!r}") from e

        try:
            await s3.create_bucket(Bucket=bucket)
        except ClientError as e:
            # If someone else just created it in a race, treat as success.
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                return
            raise S3Error(f"failed to create bucket {bucket!r}") from e
