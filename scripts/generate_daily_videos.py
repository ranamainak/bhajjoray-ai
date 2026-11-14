"""Generate daily Hindi & English lip-synced Sarv videos and upload to Cloudflare R2.

Zero-cost pipeline:
1. Fetch daily script_hi.json & script_en.json from Cloudflare R2 (or local path).
2. Synthesize warm Indian male voiceovers via edge-tts (hi-IN-MadhurNeural, en-IN-PrabhatNeural).
3. Lip-sync Sarv's 15s seamless living idle loop using Wav2Lip-ONNX + YuNet face detection.
4. Mux with FFmpeg (H.264 + AAC, +faststart) and upload video_hi.mp4 / video_en.mp4 to R2.
"""

import argparse
import asyncio
import datetime
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import cv2
import edge_tts
import numpy as np
import onnxruntime as ort
from scipy import signal

FPS = 25
MEL_STEP_SIZE = 16
WAV2LIP_IMG_SIZE = 96

R2_ENDPOINT = os.getenv(
    "R2_ENDPOINT",
    "https://175497e00f741420116f78dbe6ca38c3.r2.cloudflarestorage.com",
)
R2_BUCKET = os.getenv("R2_BUCKET", "bhajjoray")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")

HF_WAV2LIP_URL = "https://huggingface.co/bluefoxcreation/Wav2lip-Onnx/resolve/main/wav2lip.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

VOICES = {
    "hi": ("hi-IN-MadhurNeural", "-5%"),
    "en": ("en-IN-PrabhatNeural", "-5%"),
}

PIPER_VOICES = {
    "hi": (
        "hi_IN-pratham-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json",
    ),
    "en": (
        "en_US-ryan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    ),
}


def ist_today() -> str:
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    return datetime.datetime.now(ist).strftime("%Y-%m-%d")


def get_r2_client():
    if not R2_ACCESS_KEY or not R2_SECRET_KEY:
        raise RuntimeError(
            "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables must be set."
        )
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )


def ensure_asset(local_path: Path, r2_key: Optional[str], fallback_url: Optional[str], s3=None) -> Path:
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if s3 is not None and r2_key:
        try:
            print(f"Downloading {r2_key} from R2 -> {local_path}")
            s3.download_file(R2_BUCKET, r2_key, str(local_path))
            return local_path
        except Exception as exc:
            print(f"R2 download failed for {r2_key}: {exc}")
    if fallback_url:
        print(f"Downloading {fallback_url} -> {local_path}")
        urllib.request.urlretrieve(fallback_url, str(local_path))
        return local_path
    raise FileNotFoundError(f"Asset not found locally or remotely: {local_path}")


async def _tts_async(text: str, voice: str, rate: str, out_mp3: str) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(out_mp3)


def _synthesize_piper(text: str, lang: str, out_mp3: str, models_dir: Path) -> None:
    import wave
    from piper.voice import PiperVoice

    fname, onnx_url, json_url = PIPER_VOICES[lang]
    piper_dir = models_dir / "piper"
    onnx_path = piper_dir / fname
    json_path = piper_dir / f"{fname}.json"
    ensure_asset(onnx_path, f"assets/models/piper/{fname}", onnx_url)
    ensure_asset(json_path, f"assets/models/piper/{fname}.json", json_url)

    voice = PiperVoice.load(str(onnx_path), config_path=str(json_path))
    raw_wav = out_mp3 + ".piper.wav"
    with wave.open(raw_wav, "wb") as wf:
        voice.synthesize_wav(text, wf)
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_wav, "-codec:a", "libmp3lame", "-b:a", "128k", out_mp3],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(raw_wav)


def synthesize_tts(text: str, lang: str, out_mp3: str, out_wav_16k: str, models_dir: Path) -> float:
    voice, rate = VOICES[lang]
    try:
        asyncio.run(_tts_async(text, voice, rate, out_mp3))
    except Exception as exc:
        print(f"[{lang}] edge-tts failed ({exc}); falling back to offline Piper neural TTS...")
        _synthesize_piper(text, lang, out_mp3, models_dir)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", out_mp3,
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            out_wav_16k,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Probe duration in seconds
    res = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            out_wav_16k,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(res.stdout.strip())


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_basis(sr: int = 16000, n_fft: int = 800, n_mels: int = 80, fmin: float = 55.0, fmax: float = 7600.0) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)
    mel_min = _hz_to_mel(np.array([fmin]))[0]
    mel_max = _hz_to_mel(np.array([fmax]))[0]
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mels)

    ramps = np.subtract.outer(hz_pts, fft_freqs)
    fdiff = np.diff(hz_pts)
    basis = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        basis[i] = np.maximum(0.0, np.minimum(lower, upper))
        # Slaney area normalization
        enorm = 2.0 / (hz_pts[i + 2] - hz_pts[i])
        basis[i] *= enorm
    return basis


def compute_wav2lip_mel(wav_path: str) -> np.ndarray:
    """Compute Wav2Lip-compatible (80, T) normalized mel spectrogram without librosa."""
    import wave

    with wave.open(wav_path, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        raw = wf.readframes(wf.getnframes())
    wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Pre-emphasis (0.97)
    wav = signal.lfilter([1.0, -0.97], [1.0], wav)

    n_fft = 800
    hop_length = 200
    win_length = 800
    # Reflect pad like librosa center=True
    pad = n_fft // 2
    wav_padded = np.pad(wav, (pad, pad), mode="reflect")
    window = np.hanning(win_length + 1)[:-1].astype(np.float32)

    n_frames = 1 + (len(wav_padded) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        wav_padded,
        shape=(n_frames, n_fft),
        strides=(wav_padded.strides[0] * hop_length, wav_padded.strides[0]),
    ).copy()
    frames *= window[None, :]
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)).T  # (401, T)

    mel_basis = _build_mel_basis()
    mel = np.dot(mel_basis, spec)
    min_level = np.exp(-100.0 / 20.0 * np.log(10.0))
    mel_db = 20.0 * np.log10(np.maximum(min_level, mel)) - 20.0

    # Symmetric normalization to [-4.0, 4.0]
    norm = np.clip((2.0 * 4.0) * ((mel_db - (-100.0)) / 100.0) - 4.0, -4.0, 4.0)
    return norm.astype(np.float32)


def build_mel_chunks(mel: np.ndarray, fps: int = FPS) -> List[np.ndarray]:
    mel_idx_multiplier = 80.0 / fps
    chunks: List[np.ndarray] = []
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > mel.shape[1]:
            chunks.append(mel[:, mel.shape[1] - MEL_STEP_SIZE :])
            break
        chunks.append(mel[:, start_idx : start_idx + MEL_STEP_SIZE])
        i += 1
    return chunks


def load_idle_frames_and_boxes(
    idle_video_path: Path, yunet_path: Path
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    cap = cv2.VideoCapture(str(idle_video_path))
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {idle_video_path}")

    h, w = frames[0].shape[:2]
    detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (w, h), 0.6, 0.3, 5000)

    raw_boxes: List[Tuple[int, int, int, int]] = []
    last_box: Optional[Tuple[int, int, int, int]] = None
    for frame in frames:
        _, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            x, y, bw, bh = faces[0][:4]
            # Pad slightly for Wav2Lip (include chin/beard cleanly)
            y1 = max(0, int(y - 0.05 * bh))
            y2 = min(h, int(y + 1.12 * bh))
            x1 = max(0, int(x - 0.08 * bw))
            x2 = min(w, int(x + 1.08 * bw))
            box = (x1, y1, x2, y2)
            last_box = box
        elif last_box is not None:
            box = last_box
        else:
            raise RuntimeError("No face detected in idle loop frame.")
        raw_boxes.append(box)

    # Smooth boxes slightly across circular 15s boundary
    n = len(raw_boxes)
    smoothed: List[Tuple[int, int, int, int]] = []
    for i in range(n):
        idxs = [(i + d) % n for d in (-2, -1, 0, 1, 2)]
        xs1 = int(round(sum(raw_boxes[j][0] for j in idxs) / 5))
        ys1 = int(round(sum(raw_boxes[j][1] for j in idxs) / 5))
        xs2 = int(round(sum(raw_boxes[j][2] for j in idxs) / 5))
        ys2 = int(round(sum(raw_boxes[j][3] for j in idxs) / 5))
        smoothed.append((xs1, ys1, xs2, ys2))

    return frames, smoothed


def build_mouth_feather_mask(box_w: int, box_h: int) -> np.ndarray:
    """Soft elliptical mask covering only the mouth & immediate beard/mustache for crisp blending."""
    mask = np.zeros((box_h, box_w), dtype=np.float32)
    cx = box_w * 0.50
    cy = box_h * 0.69
    rx = box_w * 0.28
    ry = box_h * 0.16
    yy, xx = np.mgrid[0:box_h, 0:box_w].astype(np.float32)
    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.clip(1.0 - dist, 0.0, 1.0) ** 1.6
    # Keep upper half strictly 0 so eyes/nose stay 100% HD original
    mask[: int(box_h * 0.52), :] = 0.0
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask[..., None]


def render_lipsync_video(
    idle_frames: List[np.ndarray],
    idle_boxes: List[Tuple[int, int, int, int]],
    wav_16k_path: str,
    audio_mp3_path: str,
    wav2lip_session: ort.InferenceSession,
    out_mp4_path: str,
    batch_size: int = 32,
) -> None:
    mel = compute_wav2lip_mel(wav_16k_path)
    mel_chunks = build_mel_chunks(mel, fps=FPS)
    total_frames = len(mel_chunks)
    n_idle = len(idle_frames)
    h, w = idle_frames[0].shape[:2]

    tmp_silent = out_mp4_path + ".silent.mp4"
    writer = cv2.VideoWriter(
        tmp_silent,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h),
    )

    input_vid_name = "vid"
    input_mel_name = "mel"
    output_name = wav2lip_session.get_outputs()[0].name

    mask_cache: Dict[Tuple[int, int], np.ndarray] = {}

    for start in range(0, total_frames, batch_size):
        end = min(start + batch_size, total_frames)
        cur_bs = end - start

        img_batch = np.zeros((cur_bs, 6, WAV2LIP_IMG_SIZE, WAV2LIP_IMG_SIZE), dtype=np.float32)
        mel_batch = np.zeros((cur_bs, 1, 80, MEL_STEP_SIZE), dtype=np.float32)
        coords_batch: List[Tuple[int, int, int, int]] = []
        base_frames: List[np.ndarray] = []

        for b_idx, f_idx in enumerate(range(start, end)):
            idle_idx = f_idx % n_idle
            frame = idle_frames[idle_idx].copy()
            x1, y1, x2, y2 = idle_boxes[idle_idx]
            face = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_96 = cv2.resize(face_rgb, (WAV2LIP_IMG_SIZE, WAV2LIP_IMG_SIZE)).astype(np.float32) / 255.0

            masked = face_96.copy()
            masked[WAV2LIP_IMG_SIZE // 2 :, :, :] = 0.0

            # Concatenate masked (3ch) + reference (3ch) -> NCHW (6, 96, 96)
            combined = np.concatenate([masked, face_96], axis=2).transpose(2, 0, 1)
            img_batch[b_idx] = combined
            mel_batch[b_idx, 0] = mel_chunks[f_idx]
            coords_batch.append((x1, y1, x2, y2))
            base_frames.append(frame)

        preds = wav2lip_session.run(
            [output_name],
            {input_mel_name: mel_batch, input_vid_name: img_batch},
        )[0]  # (B, 3, 96, 96)

        for b_idx in range(cur_bs):
            pred_rgb = np.clip(preds[b_idx].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
            pred = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
            x1, y1, x2, y2 = coords_batch[b_idx]
            bw, bh = x2 - x1, y2 - y1
            pred_resized = cv2.resize(pred, (bw, bh), interpolation=cv2.INTER_CUBIC)

            key = (bw, bh)
            if key not in mask_cache:
                mask_cache[key] = build_mouth_feather_mask(bw, bh)
            alpha = mask_cache[key]

            frame = base_frames[b_idx]
            orig_roi = frame[y1:y2, x1:x2].astype(np.float32)
            blended = orig_roi * (1.0 - alpha) + pred_resized.astype(np.float32) * alpha
            frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            writer.write(frame)

    writer.release()

    # Mux with AAC audio + web-ready H.264 +faststart
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_silent,
            "-i", audio_mp3_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart",
            out_mp4_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_silent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily Sarv lip-synced videos.")
    parser.add_argument("--date", default=ist_today(), help="Date YYYY-MM-DD (default: IST today)")
    parser.add_argument("--langs", default="hi,en", help="Comma-separated languages (hi,en)")
    parser.add_argument("--scripts-dir", default="", help="Optional local folder with script_hi.json / script_en.json")
    parser.add_argument("--out-dir", default="", help="Optional local output directory for generated videos")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading videos to R2")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    idle_path = repo_root / "assets" / "characters" / "sarv_idle_loop.mp4"
    yunet_path = repo_root / "models" / "face_detection_yunet_2023mar.onnx"
    wav2lip_path = repo_root / "models" / "wav2lip.onnx"

    s3 = None
    if not args.skip_upload or not args.scripts_dir or not idle_path.exists() or not wav2lip_path.exists():
        if R2_ACCESS_KEY and R2_SECRET_KEY:
            s3 = get_r2_client()

    ensure_asset(idle_path, "assets/characters/sarv_idle_loop.mp4", None, s3=s3)
    ensure_asset(yunet_path, None, YUNET_URL, s3=s3)
    ensure_asset(wav2lip_path, "assets/models/wav2lip.onnx", HF_WAV2LIP_URL, s3=s3)

    print(f"Loading idle frames & face boxes from {idle_path}...")
    idle_frames, idle_boxes = load_idle_frames_and_boxes(idle_path, yunet_path)
    print(f"Loaded {len(idle_frames)} idle frames.")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = os.cpu_count() or 4
    wav2lip_sess = ort.InferenceSession(str(wav2lip_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    out_root = Path(args.out_dir) if args.out_dir else repo_root / "generated_videos" / args.date
    out_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for lang in langs:
            script_name = f"script_{lang}.json"
            if args.scripts_dir:
                script_file = Path(args.scripts_dir) / script_name
                payload = json.loads(script_file.read_text(encoding="utf-8"))
            else:
                if s3 is None:
                    s3 = get_r2_client()
                r2_script_key = f"podcast/daily/{args.date}/{script_name}"
                local_script = Path(tmpdir) / script_name
                print(f"Fetching {r2_script_key} from R2...")
                s3.download_file(R2_BUCKET, r2_script_key, str(local_script))
                payload = json.loads(local_script.read_text(encoding="utf-8"))

            text = payload["script"]
            mp3_path = str(Path(tmpdir) / f"audio_{lang}.mp3")
            wav_path = str(Path(tmpdir) / f"audio_{lang}_16k.wav")
            out_mp4 = str(out_root / f"video_{lang}.mp4")

            print(f"[{lang}] Synthesizing TTS ({len(text.split())} words)...")
            dur = synthesize_tts(text, lang, mp3_path, wav_path, repo_root / "models")
            print(f"[{lang}] Audio duration: {dur:.1f}s -> rendering lip-sync...")

            render_lipsync_video(
                idle_frames=idle_frames,
                idle_boxes=idle_boxes,
                wav_16k_path=wav_path,
                audio_mp3_path=mp3_path,
                wav2lip_session=wav2lip_sess,
                out_mp4_path=out_mp4,
            )
            size_mb = os.path.getsize(out_mp4) / (1024 * 1024)
            print(f"[{lang}] Rendered {out_mp4} ({size_mb:.2f} MB)")

            if not args.skip_upload:
                if s3 is None:
                    s3 = get_r2_client()
                r2_video_key = f"podcast/daily/{args.date}/video_{lang}.mp4"
                print(f"[{lang}] Uploading -> s3://{R2_BUCKET}/{r2_video_key}")
                s3.upload_file(
                    out_mp4,
                    R2_BUCKET,
                    r2_video_key,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
                print(f"[{lang}] Uploaded successfully!")


if __name__ == "__main__":
    main()
