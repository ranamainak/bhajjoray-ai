"""Build Sarv's seamless 15-second living idle loop video from sarv_master.png."""

import os
import subprocess
import sys
import numpy as np
import cv2

FPS = 25
DURATION_SEC = 15.0
NUM_FRAMES = int(FPS * DURATION_SEC)  # 375 frames

# Exact eye centers from YuNet on sarv_master.png (1376x768)
RIGHT_EYE = (661.8, 257.3)
LEFT_EYE = (719.4, 258.0)

# 4 natural blink centers (seconds) within 15s loop
BLINK_TIMES = [2.1, 5.8, 9.7, 13.4]


def blink_amount(t: float) -> float:
    """Return 0.0 (open) to 1.0 (closed) with natural asymmetric blink curve."""
    val = 0.0
    for bt in BLINK_TIMES:
        dt = t - bt
        # Fast close (-0.08s..0s), slightly slower open (0s..+0.13s)
        if -0.08 <= dt <= 0.0:
            u = (dt + 0.08) / 0.08
            val = max(val, 0.5 - 0.5 * np.cos(np.pi * u))
        elif 0.0 < dt <= 0.13:
            u = 1.0 - (dt / 0.13)
            val = max(val, 0.5 - 0.5 * np.cos(np.pi * u))
    return float(val)


def apply_eye_blink(frame: np.ndarray, amount: float, skin_color: np.ndarray) -> np.ndarray:
    """Apply a realistic upper-eyelid blink around both eyes."""
    if amount < 0.02:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]

    for (cx, cy) in (RIGHT_EYE, LEFT_EYE):
        rx, ry = 18, 12
        x0 = max(0, int(cx - rx - 4))
        x1 = min(w, int(cx + rx + 4))
        y0 = max(0, int(cy - ry - 4))
        y1 = min(h, int(cy + ry + 4))

        patch = out[y0:y1, x0:x1].astype(np.float32)
        ph, pw = patch.shape[:2]
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        # Elliptical eye mask
        norm_x = (xx - cx) / rx
        norm_y = (yy - cy) / ry
        dist2 = norm_x ** 2 + norm_y ** 2
        mask = np.clip(1.0 - dist2, 0.0, 1.0) ** 1.3

        # Pull upper eyelid pixels downward over the eye opening
        eyelid_shift = amount * 6.5 * mask
        src_y = np.clip(yy - y0 - eyelid_shift, 0, ph - 1).astype(np.float32)
        src_x = (xx - x0).astype(np.float32)
        warped = cv2.remap(patch, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Subtle eyelid crease darkening near lower lid line when closing
        lid_line = np.exp(-((yy - (cy + 1.5 * amount)) / 2.2) ** 2) * mask * amount
        blended = warped * (1.0 - 0.35 * amount * mask[..., None]) + skin_color[None, None, :] * (0.35 * amount * mask[..., None])
        blended = blended * (1.0 - 0.28 * lid_line[..., None])

        alpha = np.clip(mask * (0.35 + 0.65 * amount), 0.0, 1.0)[..., None]
        out[y0:y1, x0:x1] = np.clip(patch * (1.0 - alpha) + blended * alpha, 0, 255).astype(np.uint8)

    return out


def build_idle_loop(master_path: str, out_path: str) -> None:
    img = cv2.imread(master_path)
    if img is None:
        raise FileNotFoundError(f"Could not read master image: {master_path}")

    h, w = img.shape[:2]

    # Sample upper-eyelid skin tone just above right eye
    eyelid_patch = img[int(RIGHT_EYE[1] - 11):int(RIGHT_EYE[1] - 7), int(RIGHT_EYE[0] - 6):int(RIGHT_EYE[0] + 6)]
    skin_color = eyelid_patch.mean(axis=(0, 1)).astype(np.float32)

    # Precompute spatial weights for seamless breathing + gentle sway
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / (w * 0.5)  # -1..1
    ny = yy / h                       # 0..1

    # Character region mask: centered horizontally around Sarv (x ~ 0.25..0.75), anchored at bottom (ny=1)
    char_horiz = np.exp(-((xx - w * 0.505) / (w * 0.22)) ** 4)
    # Torso & head rise on inhale, hips stay planted on grass
    breath_weight = char_horiz * np.clip(1.0 - (ny ** 2.2), 0.0, 1.0) * np.clip((ny - 0.12) / 0.25, 0.25, 1.0)
    # Subtle chest expansion outward around chest/shoulders (y ~ 0.45..0.75)
    chest_weight = char_horiz * np.exp(-((ny - 0.58) / 0.18) ** 2)
    # Head micro-sway (y ~ 0.18..0.52)
    head_weight = char_horiz * np.exp(-((ny - 0.34) / 0.20) ** 2)

    # Subtle sunlit mist shimmer mask on upper-right background (away from Sarv's face)
    sun_mask = (
        np.clip((xx - w * 0.62) / (w * 0.25), 0.0, 1.0)
        * np.clip(1.0 - ny / 0.75, 0.0, 1.0)
        * (1.0 - char_horiz)
    )[..., None]

    tmp_avi = out_path + ".tmp.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_avi, fourcc, FPS, (w, h))

    base_f = img.astype(np.float32)

    for i in range(NUM_FRAMES):
        t = i / FPS
        # Exact periodic cycles over 15s so frame 375 matches frame 0
        breath = np.sin(2.0 * np.pi * (t / 5.0))            # 3 breaths in 15s
        sway = np.sin(2.0 * np.pi * (t / 15.0))             # 1 gentle sway in 15s
        sun_pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * (t / 7.5))  # 2 soft light pulses in 15s

        # Map destination -> source coordinates
        dy = -2.2 * breath * breath_weight
        dx = (-1.6 * sway * head_weight) - (1.2 * breath * chest_weight * nx)

        map_x = np.clip(xx - dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(yy - dy, 0, h - 1).astype(np.float32)

        warped = cv2.remap(base_f, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Gentle early-morning golden shimmer on background mist
        warmth = np.array([0.98, 1.01, 1.03], dtype=np.float32)  # BGR slight golden warmth
        warped = warped * (1.0 + sun_mask * (0.035 * sun_pulse) * (warmth - 1.0 + 0.5))
        frame = np.clip(warped, 0, 255).astype(np.uint8)

        # Apply natural blink
        b_amt = blink_amount(t)
        if b_amt > 0.02:
            frame = apply_eye_blink(frame, b_amt, skin_color)

        writer.write(frame)

    writer.release()

    # Re-encode to web/Wav2Lip-friendly H.264 YUV420p MP4 via FFmpeg
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", tmp_avi,
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_avi)
    print(f"Built seamless 15s idle loop: {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master = os.path.join(root, "assets", "characters", "sarv_master.png")
    out = os.path.join(root, "assets", "characters", "sarv_idle_loop.mp4")
    build_idle_loop(master, out)
