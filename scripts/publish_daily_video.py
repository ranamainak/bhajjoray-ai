"""Publish daily Hindi Sarv video from Cloudflare R2 to YouTube Shorts, Instagram Reels, and Facebook Page."""

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import boto3
import requests

TITLE = "भज्जोरे ज्ञान सीरीज़"
GRAPH_API_VERSION = "v21.0"

R2_ENDPOINT = os.getenv(
    "R2_ENDPOINT",
    "https://175497e00f741420116f78dbe6ca38c3.r2.cloudflarestorage.com",
)
R2_BUCKET = os.getenv("R2_BUCKET", "bhajjoray")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")


def ist_today() -> str:
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    return dt.datetime.now(ist).strftime("%Y-%m-%d")


def get_r2_client():
    if not R2_ACCESS_KEY or not R2_SECRET_KEY:
        raise RuntimeError("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set.")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )


def build_caption(shloka_ref: str) -> str:
    lines = [TITLE]
    if shloka_ref:
        lines.append(shloka_ref)
    lines.append("#Shorts #BhagavadGita #SpiritualWisdom #Bhajjoray")
    return "\n\n".join(lines)


def publish_youtube_shorts(video_path: Path, title: str, description: str) -> str:
    client_id = os.getenv("YOUTUBE_CLIENT_ID", "").strip()
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "").strip()
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN", "").strip()
    if not (client_id and client_secret and refresh_token):
        raise RuntimeError("Missing YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET / YOUTUBE_REFRESH_TOKEN")

    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if not token_resp.ok:
        raise RuntimeError(f"YouTube OAuth token error ({token_resp.status_code}): {token_resp.text}")
    access_token = token_resp.json()["access_token"]

    metadata = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": ["Shorts", "Bhagavad Gita", "Bhajjoray", "Sarv", "Hindi"],
            "categoryId": "22",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    file_size = video_path.stat().st_size
    init_resp = requests.post(
        "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(file_size),
            "X-Upload-Content-Type": "video/mp4",
        },
        json=metadata,
        timeout=30,
    )
    init_resp.raise_for_status()
    upload_url = init_resp.headers["Location"]

    with video_path.open("rb") as f:
        up_resp = requests.put(
            upload_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "video/mp4",
                "Content-Length": str(file_size),
            },
            data=f,
            timeout=600,
        )
    up_resp.raise_for_status()
    video_id = up_resp.json().get("id", "")
    return f"https://youtube.com/shorts/{video_id}"


def publish_instagram_reels(presigned_url: str, caption: str) -> str:
    ig_user_id = os.getenv("IG_USER_ID", "").strip()
    access_token = os.getenv("META_ACCESS_TOKEN", "").strip()
    if not (ig_user_id and access_token):
        raise RuntimeError("Missing IG_USER_ID / META_ACCESS_TOKEN")

    create_resp = requests.post(
        f"https://graph.facebook.com/{GRAPH_API_VERSION}/{ig_user_id}/media",
        data={
            "media_type": "REELS",
            "video_url": presigned_url,
            "caption": caption,
            "share_to_feed": "true",
            "access_token": access_token,
        },
        timeout=60,
    )
    create_resp.raise_for_status()
    creation_id = create_resp.json()["id"]

    # Poll until Instagram finishes downloading & encoding the Reel
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        status_resp = requests.get(
            f"https://graph.facebook.com/{GRAPH_API_VERSION}/{creation_id}",
            params={"fields": "status_code,status", "access_token": access_token},
            timeout=30,
        )
        status_resp.raise_for_status()
        info = status_resp.json()
        code = info.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Instagram processing failed: {info}")
        time.sleep(10)
    else:
        raise TimeoutError("Instagram Reel processing timed out after 300s")

    pub_resp = requests.post(
        f"https://graph.facebook.com/{GRAPH_API_VERSION}/{ig_user_id}/media_publish",
        data={"creation_id": creation_id, "access_token": access_token},
        timeout=60,
    )
    pub_resp.raise_for_status()
    return pub_resp.json().get("id", "")


def publish_facebook_post(presigned_url: str, title: str, description: str) -> str:
    fb_page_id = os.getenv("FB_PAGE_ID", "").strip()
    page_token = os.getenv("FB_PAGE_ACCESS_TOKEN", "").strip()
    meta_token = os.getenv("META_ACCESS_TOKEN", "").strip()
    if not fb_page_id or not (page_token or meta_token):
        raise RuntimeError("Missing FB_PAGE_ID or FB_PAGE_ACCESS_TOKEN / META_ACCESS_TOKEN")

    if not page_token:
        # Derive Page access token from user/system-user META_ACCESS_TOKEN
        pt_resp = requests.get(
            f"https://graph.facebook.com/{GRAPH_API_VERSION}/{fb_page_id}",
            params={"fields": "access_token", "access_token": meta_token},
            timeout=30,
        )
        pt_resp.raise_for_status()
        page_token = pt_resp.json().get("access_token") or meta_token

    resp = requests.post(
        f"https://graph-video.facebook.com/{GRAPH_API_VERSION}/{fb_page_id}/videos",
        data={
            "file_url": presigned_url,
            "title": title,
            "description": description,
            "access_token": page_token,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("id", "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish daily Hindi video to YT Shorts, IG Reels, FB.")
    parser.add_argument("--date", default=ist_today(), help="Date YYYY-MM-DD (default: IST today)")
    parser.add_argument("--platforms", default="youtube,instagram,facebook", help="Comma-separated platforms")
    parser.add_argument("--dry-run", action="store_true", help="Verify R2 assets & print plan without posting")
    args = parser.parse_args()

    s3 = get_r2_client()
    video_key = f"podcast/daily/{args.date}/video_hi.mp4"
    script_key = f"podcast/daily/{args.date}/script_hi.json"

    shloka_ref = ""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        try:
            script_local = tmp / "script_hi.json"
            s3.download_file(R2_BUCKET, script_key, str(script_local))
            shloka_ref = json.loads(script_local.read_text(encoding="utf-8")).get("shloka_reference", "")
        except Exception as exc:
            print(f"Note: could not read {script_key} ({exc}); continuing with base title.")

        video_local = tmp / "video_hi.mp4"
        print(f"Downloading s3://{R2_BUCKET}/{video_key}...")
        s3.download_file(R2_BUCKET, video_key, str(video_local))
        print(f"Downloaded {video_local.stat().st_size / (1024 * 1024):.2f} MB")

        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET, "Key": video_key},
            ExpiresIn=3600,
        )

        caption = build_caption(shloka_ref)
        platforms = [p.strip().lower() for p in args.platforms.split(",") if p.strip()]

        if args.dry_run:
            print(f"[DRY RUN] Title: {TITLE}\nCaption:\n{caption}\nPlatforms: {platforms}")
            return 0

        errors = []
        if "youtube" in platforms:
            try:
                yt_url = publish_youtube_shorts(video_local, TITLE, caption)
                print(f"[YouTube Shorts] Published: {yt_url}")
            except Exception as exc:
                print(f"[YouTube Shorts] ERROR: {exc}", file=sys.stderr)
                errors.append(("youtube", str(exc)))

        if "instagram" in platforms:
            try:
                ig_id = publish_instagram_reels(presigned_url, caption)
                print(f"[Instagram Reels] Published media ID: {ig_id}")
            except Exception as exc:
                print(f"[Instagram Reels] ERROR: {exc}", file=sys.stderr)
                errors.append(("instagram", str(exc)))

        if "facebook" in platforms:
            try:
                fb_id = publish_facebook_post(presigned_url, TITLE, caption)
                print(f"[Facebook Post] Published video ID: {fb_id}")
            except Exception as exc:
                print(f"[Facebook Post] ERROR: {exc}", file=sys.stderr)
                errors.append(("facebook", str(exc)))

        if errors:
            print(f"Completed with {len(errors)} platform error(s): {errors}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
