"""Trigger daily Hindi and English podcast script generation on Azure Container Apps."""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://bhajjoray-ai-app.kinddesert-f3b90dc1.eastus.azurecontainerapps.io"
VERSES_PATH = Path(__file__).resolve().parent.parent / "data" / "daily_verses.json"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
EPOCH_DATE = dt.date(2026, 11, 1)


def _request_json(method: str, url: str, payload: dict | None = None, timeout: int = 300) -> tuple[int, dict]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def wait_for_health(base_url: str, max_wait_seconds: int = 180) -> None:
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        try:
            status, data = _request_json("GET", f"{base_url}/health", timeout=30)
            if status == 200 and data.get("model_status") == "ready":
                print(f"Health OK: {data.get('app_version')} ({data.get('model_name')})")
                return
        except Exception as exc:
            print(f"Waiting for container health: {exc}")
        time.sleep(10)
    raise RuntimeError(f"Container at {base_url} did not become ready within {max_wait_seconds}s")


def select_verse(reference_override: str | None = None) -> dict:
    verses = json.loads(VERSES_PATH.read_text(encoding="utf-8"))
    if not verses:
        raise RuntimeError(f"No verses found in {VERSES_PATH}")

    if reference_override:
        target = reference_override.strip().lower()
        for item in verses:
            ref = item["shloka_reference"].lower()
            if ref == target or ref.endswith(f" {target}"):
                return item
        raise ValueError(f"Verse reference not found: {reference_override}")

    today_ist = dt.datetime.now(IST).date()
    day_index = (today_ist - EPOCH_DATE).days % len(verses)
    return verses[day_index]


def generate_one(base_url: str, verse: dict, language: str, max_attempts: int = 2) -> dict:
    payload = {**verse, "language": language}
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[{language}] Attempt {attempt}/{max_attempts} for {verse['shloka_reference']}...")
            status, resp = _request_json(
                "POST",
                f"{base_url}/generate-podcast",
                payload=payload,
                timeout=300,
            )
            r2 = resp.get("r2_upload") or {}
            print(
                f"[{language}] Success (HTTP {status}): "
                f"{resp.get('word_count')} words, R2 key={r2.get('object_key')}"
            )
            return resp
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {exc.code}: {body}"
            print(f"[{language}] Attempt {attempt} failed: {last_err}")
        except (URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
            print(f"[{language}] Attempt {attempt} network/timeout error: {last_err}")
        if attempt < max_attempts:
            time.sleep(5)
    raise RuntimeError(f"[{language}] Failed after {max_attempts} attempts: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("BHJ_API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--reference", default=os.getenv("SHLOKA_REFERENCE", "").strip() or None)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    verse = select_verse(args.reference)
    print(f"Selected verse for {dt.datetime.now(IST).date().isoformat()} IST: {verse['shloka_reference']}")

    wait_for_health(base_url)
    generate_one(base_url, verse, "Hindi")
    generate_one(base_url, verse, "English")
    return 0


if __name__ == "__main__":
    sys.exit(main())
