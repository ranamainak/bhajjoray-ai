
import json
import os
from datetime import date
from typing import Any

import boto3
from botocore.config import Config


def _get_r2_client():
    """Create an authenticated Cloudflare R2 S3 client."""

    account_endpoint = os.getenv("R2_ENDPOINT")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

    missing = [
        name for name, value in {
            "R2_ENDPOINT": account_endpoint,
            "R2_ACCESS_KEY_ID": access_key,
            "R2_SECRET_ACCESS_KEY": secret_key,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            f"Missing required R2 environment variables: {', '.join(missing)}"
        )

    return boto3.client(
        service_name="s3",
        endpoint_url=account_endpoint.rstrip("/"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            connect_timeout=10,
            read_timeout=30,
            retries={
                "max_attempts": 2,
                "mode": "standard",
            },
        ),
    )


def upload_podcast_script(
    podcast_result: dict[str, Any],
    publication_date: str,
) -> dict[str, Any]:
    """
    Upload one successfully generated podcast script to Cloudflare R2.

    Hindi:
        podcast/daily/YYYY-MM-DD/script_hi.json

    English:
        podcast/daily/YYYY-MM-DD/script_en.json
    """

    bucket = os.getenv("R2_BUCKET_NAME", "bhajjoray")
    prefix = os.getenv("R2_PREFIX", "podcast/daily").strip("/")

    # Validate publication date.
    try:
        date.fromisoformat(publication_date)
    except ValueError as exc:
        raise ValueError(
            "publication_date must use YYYY-MM-DD format"
        ) from exc

    # Identify language and filename.
    language = str(podcast_result.get("language", "")).strip().lower()

    filenames = {
        "hindi": "script_hi.json",
        "english": "script_en.json",
    }

    if language not in filenames:
        raise ValueError(
            "Script language must be Hindi or English."
        )

    # Do not upload an empty or invalid script.
    script = podcast_result.get("script")

    if not isinstance(script, str) or not script.strip():
        raise ValueError("Podcast script is empty. Upload cancelled.")

    if not podcast_result.get("shloka_reference"):
        raise ValueError("Missing shloka reference. Upload cancelled.")

    # Build the R2 object path.
    object_key = (
        f"{prefix}/{publication_date}/{filenames[language]}"
    )

    # Preserve Unicode characters, especially Hindi Devanagari.
    json_content = json.dumps(
        podcast_result,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    # Upload to Cloudflare R2.
    client = _get_r2_client()

    response = client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=json_content,
        ContentType="application/json; charset=utf-8",
        CacheControl="no-cache",
    )

    # A successful PutObject response confirms the upload.
    return {
        "status": "success",
        "bucket": bucket,
        "object_key": object_key,
        "size_bytes": len(json_content),
        "etag": response.get("ETag"),
    }
