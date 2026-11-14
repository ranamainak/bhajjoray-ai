"""Build Sarv's seamless 15-second living idle loop video from sarv_master.png.

Includes:
- Layered wind sway on background banyan leaves, hanging aerial roots, and grass
- Subtle breeze flutter on Sarv's loose temple hair & beard tips
- Natural breathing, gentle meditative body sway, and periodic calm hand movement
- Natural asymmetric eye blinks
"""

import os
import subprocess
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
        if -0.08 <= dt <= 0.0:
            u = (dt + 0.08) / 0.08
            val = max(val, 0.5 - 0.5 * np.cos(np.pi * u))
        elif 0.0 < dt <= 0.13:
            u = 1.0 - (dt / 0.13)
            val = max(val, 0.5 - 0.5 * np.cos(np.pi * u))
    return float(val)


def hand_gesture_curve(t: float) -> float:
    """Smooth periodic bell curve for gentle meditative hand/forearm movement."""
    val = 0.0
    for center, width in ((4.5, 1.8), (11.2, 1.9)):
        dt = abs(t - center)
        if dt < width:
            val = max(val, 0.5 * (1.0 + np.cos(np.pi * dt / width)))
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
        ph = patch.shape[0]
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        dist2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        mask = np.clip(1.0 - dist2, 0.0, 1.0) ** 1.3

        eyelid_shift = amount * 6.5 * mask
        src_y = np.clip(yy - y0 - eyelid_shift, 0, ph - 1).astype(np.float32)
        src_x = (xx - x0).astype(np.float32)
        warped = cv2.remap(patch, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        lid_line = np.exp(-((yy - (cy + 1.5 * amount)) / 2.2) ** 2) * mask * amount
        blended = warped * (1.0 - 0.35 * amount * mask[..., None]) + skin_color[None, None, :] * (0.35 * amount * mask[..., None])
        blended = blended * (1.0 - 0.28 * lid_line[..., None])

        alpha = np.clip(mask * (0.35 + 0.65 * amount), 0.0, 1.0)[..., None]
        out[y0:y1, x0:x1] = np.clip(patch * (1.0 - alpha) + blended * alpha, 0, 255).astype(np.uint8)

    return out


def build_sarv_masks(h: int, w: int):
    """Build smooth spatial masks for Sarv's head, hair, torso, hands, and background foliage."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / (w * 0.5)
    ny = yy / h

    # Head + upper shoulders silhouette
    head_mask = np.exp(-(((xx - w * 0.501) / (w * 0.075)) ** 4 + ((yy - h * 0.34) / (h * 0.16)) ** 4))
    # Torso + crossed legs silhouette
    torso_mask = np.exp(-(((xx - w * 0.505) / (w * 0.165)) ** 4 + ((yy - h * 0.72) / (h * 0.34)) ** 4))
    # Combined character body mask (protects Sarv from background wind waves)
    body_mask = np.clip(np.maximum(head_mask, torso_mask), 0.0, 1.0)
    body_mask = cv2.GaussianBlur(body_mask, (31, 31), 0)

    # Loose temple hair & beard tips (breeze flutter without disturbing eyes/mouth)
    left_temple = np.exp(-(((xx - 618.0) / 22.0) ** 2 + ((yy - 245.0) / 42.0) ** 2))
    right_temple = np.exp(-(((xx - 758.0) / 22.0) ** 2 + ((yy - 245.0) / 42.0) ** 2))
    beard_tip = np.exp(-(((xx - 688.0) / 38.0) ** 2 + ((yy - 368.0) / 24.0) ** 2))
    hair_mask = np.clip(left_temple + right_temple + 0.7 * beard_tip, 0.0, 1.0)

    # Breathing & gentle body sway weights
    breath_weight = body_mask * np.clip(1.0 - (ny ** 2.4), 0.0, 1.0) * np.clip((ny - 0.12) / 0.22, 0.25, 1.0)
    chest_weight = body_mask * np.exp(-((ny - 0.57) / 0.17) ** 2)
    head_sway_weight = body_mask * np.exp(-((ny - 0.34) / 0.20) ** 2)

    # Hands & forearms in lap (x ~ 530..820, y ~ 590..768)
    hands_mask = np.exp(-(((xx - 675.0) / 115.0) ** 4 + ((yy - 695.0) / 72.0) ** 2)) * body_mask

    # Background foliage / banyan roots / grass mask (strictly outside Sarv)
    bg_mask = np.clip(1.0 - body_mask, 0.0, 1.0)
    # Hanging roots & canopy sway stronger higher/mid-frame; grass ripples softly lower down
    canopy_weight = bg_mask * np.clip(1.0 - np.abs(ny - 0.32) / 0.45, 0.25, 1.0)
    grass_weight = bg_mask * np.clip((ny - 0.68) / 0.32, 0.0, 1.0)

    # Subtle golden morning light shimmer in upper-right mist
    sun_mask = (
        np.clip((xx - w * 0.62) / (w * 0.25), 0.0, 1.0)
        * np.clip(1.0 - ny / 0.75, 0.0, 1.0)
        * bg_mask
    )[..., None]

    return {
        "xx": xx,
        "yy": yy,
        "nx": nx,
        "hair_mask": hair_mask,
        "breath_weight": breath_weight,
        "chest_weight": chest_weight,
        "head_sway_weight": head_sway_weight,
        "hands_mask": hands_mask,
        "canopy_weight": canopy_weight,
        "grass_weight": grass_weight,
        "sun_mask": sun_mask,
    }


def build_idle_loop(master_path: str, out_path: str) -> None:
    img = cv2.imread(master_path)
    if img is None:
        raise FileNotFoundError(f"Could not read master image: {master_path}")

    h, w = img.shape[:2]
    eyelid_patch = img[int(RIGHT_EYE[1] - 11):int(RIGHT_EYE[1] - 7), int(RIGHT_EYE[0] - 6):int(RIGHT_EYE[0] + 6)]
    skin_color = eyelid_patch.mean(axis=(0, 1)).astype(np.float32)

    m = build_sarv_masks(h, w)
    xx, yy, nx = m["xx"], m["yy"], m["nx"]

    tmp_avi = out_path + ".tmp.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    base_f = img.astype(np.float32)

    for i in range(NUM_FRAMES):
        t = i / FPS
        # Exact integer cycles over 15.0s for a 100% seamless loop
        breath = np.sin(2.0 * np.pi * (t / 5.0))            # 3 breaths
        body_sway = np.sin(2.0 * np.pi * (t / 15.0))        # 1 gentle posture sway
        hand_lift = hand_gesture_curve(t)                   # 2 calm hand gestures
        sun_pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * (t / 7.5))

        # Background breeze: traveling multi-frequency wind waves for leaves & hanging banyan roots
        wind_x1 = np.sin(2.0 * np.pi * (t / 5.0) + yy * 0.012 + xx * 0.005)
        wind_x2 = np.sin(2.0 * np.pi * (t / 3.0) - yy * 0.018 + xx * 0.008)
        wind_y = np.cos(2.0 * np.pi * (t / 5.0) + xx * 0.009 + yy * 0.010)
        grass_wave = np.sin(2.0 * np.pi * (t / 3.75) + xx * 0.022)

        # Hair breeze ripple
        hair_wave = np.sin(2.0 * np.pi * (t / 3.0) + yy * 0.05)

        dx = (
            (-2.8 * body_sway * m["head_sway_weight"])
            - (1.6 * breath * m["chest_weight"] * nx)
            + (2.2 * hair_wave * m["hair_mask"])
            + ((3.6 * wind_x1 + 1.6 * wind_x2) * m["canopy_weight"])
            + (2.4 * grass_wave * m["grass_weight"])
            - (2.5 * hand_lift * m["hands_mask"] * nx)
        )
        dy = (
            (-2.8 * breath * m["breath_weight"])
            - (7.5 * hand_lift * m["hands_mask"])
            + (0.7 * hair_wave * m["hair_mask"])
            + (1.5 * wind_y * m["canopy_weight"])
            + (0.8 * grass_wave * m["grass_weight"])
        )

        map_x = np.clip(xx - dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(yy - dy, 0, h - 1).astype(np.float32)

        warped = cv2.remap(base_f, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        warmth = np.array([0.98, 1.01, 1.03], dtype=np.float32)
        warped = warped * (1.0 + m["sun_mask"] * (0.04 * sun_pulse) * (warmth - 1.0 + 0.5))
        frame = np.clip(warped, 0, 255).astype(np.uint8)

        b_amt = blink_amount(t)
        if b_amt > 0.02:
            frame = apply_eye_blink(frame, b_amt, skin_color)

        writer.write(frame)

    writer.release()

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
