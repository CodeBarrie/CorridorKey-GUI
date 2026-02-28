"""
CorridorKey GUI — Neural Network Green Screen Keyer
A Gradio web interface for Corridor Digital's CorridorKey AI keying engine.
Upload green screen footage + click to create masks → get VFX-quality mattes.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
import gc
import json
import threading
import tempfile
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import cv2
import gradio as gr

# ─── Constants ────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "CorridorKeyModule", "checkpoints", "CorridorKey.pth")
SAM_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "sam_vit_b.pth")
MATANYONE_MODEL_ID = "PeiqingYang/MatAnyone"
SUPPORTED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _make_output_dir(prefix):
    """Create a timestamped output subdirectory under OUTPUT_DIR."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(OUTPUT_DIR, f"{prefix}_{stamp}")
    os.makedirs(d, exist_ok=True)
    return d

# Point visualization colors (BGR converted to RGB for display)
COLOR_POSITIVE = (0, 200, 80)    # green dots for "include"
COLOR_NEGATIVE = (220, 50, 50)   # red dots for "exclude"
COLOR_MASK = (80, 120, 255)      # blue-ish mask overlay
MASK_ALPHA = 0.45
POINT_RADIUS = 8

# ─── Engine Singletons (lazy load, VRAM-managed) ─────────────────────────────

_engine = None
_engine_lock = threading.Lock()
_sam_predictor = None
_sam_lock = threading.Lock()
_matanyone = None
_matanyone_lock = threading.Lock()


def _unload_all_models():
    """Unload all GPU models to free VRAM."""
    global _engine, _sam_predictor, _matanyone
    import torch
    if _engine is not None:
        del _engine
        _engine = None
    if _sam_predictor is not None:
        del _sam_predictor
        _sam_predictor = None
    if _matanyone is not None:
        del _matanyone
        _matanyone = None
    torch.cuda.empty_cache()
    gc.collect()


def get_engine():
    """Lazy-load CorridorKey. Unloads SAM/MatAnyone first."""
    global _engine, _sam_predictor, _matanyone
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        # Free VRAM from other models
        if _sam_predictor is not None:
            with _sam_lock:
                del _sam_predictor
                _sam_predictor = None
        if _matanyone is not None:
            with _matanyone_lock:
                del _matanyone
                _matanyone = None
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        from CorridorKeyModule import CorridorKeyEngine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _engine = CorridorKeyEngine(
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
            img_size=2048,
        )
        return _engine


def get_sam_predictor():
    """Lazy-load SAM predictor. Lightweight (~1GB VRAM for vit_b)."""
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    with _sam_lock:
        if _sam_predictor is not None:
            return _sam_predictor
        from segment_anything import sam_model_registry, SamPredictor
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam.eval()
        _sam_predictor = SamPredictor(sam)
        return _sam_predictor


def load_matanyone():
    """Load MatAnyone. Unloads SAM and CorridorKey first."""
    global _matanyone, _engine, _sam_predictor
    with _matanyone_lock:
        if _matanyone is not None:
            return _matanyone
        # Free VRAM
        if _engine is not None:
            with _engine_lock:
                del _engine
                _engine = None
        if _sam_predictor is not None:
            with _sam_lock:
                del _sam_predictor
                _sam_predictor = None
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        from matanyone import InferenceCore
        _matanyone = InferenceCore(MATANYONE_MODEL_ID)
        return _matanyone


def unload_matanyone():
    """Unload MatAnyone and free VRAM."""
    global _matanyone
    with _matanyone_lock:
        if _matanyone is not None:
            del _matanyone
            _matanyone = None
            import torch
            torch.cuda.empty_cache()
            gc.collect()

# ─── GPU Status ──────────────────────────────────────────────────────────────

def check_gpu_status():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"**GPU:** {name} — {vram:.1f} GB VRAM"
        else:
            return "**GPU:** None detected — running on CPU (slow)"
    except Exception:
        return "**GPU:** Unknown"

# ─── Image Helpers ───────────────────────────────────────────────────────────

def to_float32(img):
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def to_uint8(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def load_mask(path):
    if path is None:
        return None
    ext = Path(path).suffix.lower()
    if ext == ".exr":
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
    return img


def save_exr(path, image):
    if image.ndim == 3 and image.shape[2] >= 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        bgr = image
    cv2.imwrite(path, bgr.astype(np.float32))


def save_png(path, image):
    img_u8 = to_uint8(image)
    if img_u8.ndim == 3 and img_u8.shape[2] >= 3:
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR) if img_u8.shape[2] == 3 else cv2.cvtColor(img_u8, cv2.COLOR_RGBA2BGRA)
    else:
        bgr = img_u8
    cv2.imwrite(path, bgr)


def frames_to_mp4(frame_dir, output_path, fps=24.0):
    """Compile a directory of image frames into an MP4 video."""
    files = sorted([f for f in os.listdir(frame_dir)
                    if f.lower().endswith(('.png', '.exr', '.jpg', '.jpeg', '.tif', '.tiff'))])
    if not files:
        return None
    first = cv2.imread(os.path.join(frame_dir, files[0]), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if first is None:
        return None
    h, w = first.shape[:2]
    # Try H264 first for browser playback, fall back to mp4v
    writer = None
    for codec in ['avc1', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        return None
    for fname in files:
        frame = cv2.imread(os.path.join(frame_dir, fname), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if frame is None:
            continue
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        writer.write(frame)
    writer.release()
    return output_path

# ─── SAM Interactive Masking ─────────────────────────────────────────────────

def paint_mask_on_image(image, mask, points, labels):
    """Paint mask overlay + point markers onto image for display."""
    painted = image.copy()
    if mask is not None:
        # Blue-ish semi-transparent overlay where mask is 1
        overlay = painted.copy()
        overlay[mask > 0.5] = (
            np.array(overlay[mask > 0.5], dtype=np.float32) * (1 - MASK_ALPHA)
            + np.array(COLOR_MASK, dtype=np.float32) * MASK_ALPHA
        ).astype(np.uint8)
        # Contour around mask
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        painted = overlay

    # Draw points
    if points is not None and len(points) > 0:
        for pt, lbl in zip(points, labels):
            color = COLOR_POSITIVE if lbl == 1 else COLOR_NEGATIVE
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(painted, (x, y), POINT_RADIUS, color, -1)
            cv2.circle(painted, (x, y), POINT_RADIUS, (255, 255, 255), 2)

    return painted


def extract_first_frame(video_file):
    """Extract the first frame from a video file for masking."""
    if video_file is None:
        return None, None, "Upload a video first."

    video_path = video_file if isinstance(video_file, str) else video_file.name if hasattr(video_file, "name") else str(video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "Failed to open video."

    ret, frame_bgr = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    if not ret or frame_bgr is None:
        return None, None, "Failed to read first frame."

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Embed in SAM
    predictor = get_sam_predictor()
    predictor.set_image(frame_rgb)

    status = f"Loaded: {total} frames @ {fps:.1f}fps, {frame_rgb.shape[1]}x{frame_rgb.shape[0]}"
    # Return frame for display, frame as state (original), status, total frames, fps
    return frame_rgb, frame_rgb, status, total, fps


def sam_click(original_frame, click_state_json, point_mode, evt: gr.SelectData):
    """Handle click on the frame image. Run SAM with accumulated points."""
    try:
        if original_frame is None:
            raise gr.Error("Load a video first.")

        # Parse accumulated state
        if click_state_json:
            click_state = json.loads(click_state_json)
            if "points" not in click_state:
                click_state = {"points": [], "labels": []}
        else:
            click_state = {"points": [], "labels": []}

        # Add new point
        x, y = int(evt.index[0]), int(evt.index[1])
        label = 1 if point_mode == "Positive (include)" else 0
        click_state["points"].append([x, y])
        click_state["labels"].append(label)

        points = np.array(click_state["points"])
        labels = np.array(click_state["labels"])

        # Re-embed image in SAM (needed after model might have been reloaded)
        predictor = get_sam_predictor()
        if not predictor.is_image_set:
            predictor.set_image(original_frame)

        # Run SAM prediction
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        # Pick best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        # Refine with logit feedback (two-pass like Wan2GP)
        masks2, scores2, logits2 = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=logits[best_idx:best_idx+1],
            multimask_output=False,
        )
        mask = masks2[0]

        # Paint visualization
        painted = paint_mask_on_image(original_frame, mask, points, labels)

        # Return raw mask separately so expansion can re-roll from it
        return painted, json.dumps(click_state), mask, mask

    except gr.Error:
        raise
    except Exception as e:
        print(f"[SAM Click ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise gr.Error(f"SAM click failed: {e}")


def clear_clicks(original_frame):
    """Reset all clicks and mask."""
    if original_frame is None:
        return None, "{}", None, None
    return original_frame.copy(), "{}", None, None


def add_sub_mask(current_mask, saved_masks_json):
    """Save current mask to the multi-mask list."""
    if current_mask is None:
        raise gr.Error("Create a mask first by clicking on the image.")

    saved = json.loads(saved_masks_json) if saved_masks_json else []
    # Serialize mask as base64 for state storage
    mask_u8 = (current_mask > 0.5).astype(np.uint8) * 255
    _, encoded = cv2.imencode(".png", mask_u8)
    import base64
    saved.append(base64.b64encode(encoded.tobytes()).decode("ascii"))

    count = len(saved)
    return json.dumps(saved), f"{count} mask(s) saved. Click more points for next mask, or Generate."


def combine_masks(saved_masks_json, current_mask):
    """Combine all saved masks + current into one binary mask."""
    import base64
    saved = json.loads(saved_masks_json) if saved_masks_json else []

    combined = None
    for b64 in saved:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if combined is None:
            combined = (m > 127).astype(np.uint8)
        else:
            combined = np.clip(combined + (m > 127).astype(np.uint8), 0, 1)

    if current_mask is not None:
        m = (current_mask > 0.5).astype(np.uint8)
        if combined is None:
            combined = m
        else:
            combined = np.clip(combined + m, 0, 1)

    return combined


def expand_mask(raw_mask, expansion_px):
    """Dilate a mask by N pixels using an elliptical kernel."""
    if raw_mask is None or expansion_px <= 0:
        return raw_mask
    kernel_size = int(expansion_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = (raw_mask > 0.5).astype(np.uint8)
    dilated = cv2.dilate(mask_u8, kernel)
    return dilated


def apply_expansion(raw_mask, original_frame, click_state_json, expansion_px):
    """Re-dilate raw SAM mask by expansion_px and repaint the preview."""
    if raw_mask is None or original_frame is None:
        return original_frame, raw_mask if expansion_px <= 0 else None

    expanded = expand_mask(raw_mask, expansion_px)

    # Repaint with points
    click_state = json.loads(click_state_json) if click_state_json else {}
    points = np.array(click_state.get("points", []))
    labels = np.array(click_state.get("labels", []))
    pts = points if len(points) > 0 else None
    lbls = labels if len(labels) > 0 else None

    painted = paint_mask_on_image(original_frame, expanded, pts, lbls)
    return painted, expanded


# ─── Multi-Keyframe SAM Masking ──────────────────────────────────────────────

def extract_frame_at(video_path, frame_num):
    """Seek to any frame, convert BGR→RGB, embed in SAM. Returns RGB numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret or frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor = get_sam_predictor()
    predictor.set_image(frame_rgb)
    return frame_rgb


def on_frame_slider_change(video_file, frame_num, keyframes_json):
    """Handle frame slider change. Extract frame, load saved keyframe mask if exists."""
    import base64

    if video_file is None:
        return None, None, "{}", None, "[]", "No video loaded."

    video_path = video_file if isinstance(video_file, str) else video_file.name if hasattr(video_file, "name") else str(video_file)
    frame_num = int(frame_num)

    frame_rgb = extract_frame_at(video_path, frame_num)
    if frame_rgb is None:
        return None, None, "{}", None, "[]", f"Failed to read frame {frame_num}."

    # Check if this frame has a saved keyframe
    keyframes = json.loads(keyframes_json) if keyframes_json else {}
    frame_key = str(frame_num)

    if frame_key in keyframes:
        # Load and display the saved mask
        kf = keyframes[frame_key]
        buf = np.frombuffer(base64.b64decode(kf["mask_b64"]), dtype=np.uint8)
        mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        mask_bool = (mask > 127).astype(np.uint8)
        painted = paint_mask_on_image(frame_rgb, mask_bool, None, None)
        # Restore sub-masks
        sub_masks = json.dumps(kf.get("sub_masks_b64", []))
        return painted, frame_rgb, "{}", mask_bool, sub_masks, f"Frame {frame_num} — keyframe mask loaded"
    else:
        return frame_rgb.copy(), frame_rgb, "{}", None, "[]", f"Frame {frame_num}"


def save_keyframe(frame_num, current_mask, saved_masks_json, keyframes_json):
    """Save current mask + sub-masks as a keyframe at the given frame number."""
    import base64

    combined = combine_masks(saved_masks_json, current_mask)
    if combined is None:
        raise gr.Error("Create a mask first by clicking on the image.")

    # Encode combined mask as base64 PNG
    mask_u8 = (combined > 0.5).astype(np.uint8) * 255
    _, encoded = cv2.imencode(".png", mask_u8)
    mask_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    # Save sub-masks for later editing
    saved = json.loads(saved_masks_json) if saved_masks_json else []
    sub_masks_b64 = list(saved)
    if current_mask is not None:
        curr_u8 = (current_mask > 0.5).astype(np.uint8) * 255
        _, curr_enc = cv2.imencode(".png", curr_u8)
        sub_masks_b64.append(base64.b64encode(curr_enc.tobytes()).decode("ascii"))

    keyframes = json.loads(keyframes_json) if keyframes_json else {}
    frame_key = str(int(frame_num))
    keyframes[frame_key] = {
        "mask_b64": mask_b64,
        "sub_masks_b64": sub_masks_b64,
    }

    sorted_keys = sorted(keyframes.keys(), key=int)
    info = f"{len(sorted_keys)} keyframe(s): " + ", ".join(f"Frame {k}" for k in sorted_keys)
    return json.dumps(keyframes), info


def delete_keyframe(frame_num, keyframes_json):
    """Remove a keyframe at the given frame number."""
    keyframes = json.loads(keyframes_json) if keyframes_json else {}
    frame_key = str(int(frame_num))
    if frame_key in keyframes:
        del keyframes[frame_key]

    if keyframes:
        sorted_keys = sorted(keyframes.keys(), key=int)
        info = f"{len(sorted_keys)} keyframe(s): " + ", ".join(f"Frame {k}" for k in sorted_keys)
    else:
        info = "No keyframes saved."

    return json.dumps(keyframes), info


def extract_video_segment(video_path, start_frame, end_frame, output_path, fps):
    """Write a sub-range of frames to a new MP4 for MatAnyone processing."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"Failed to open video for segment extraction.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    for codec in ['avc1', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)
        if writer.isOpened():
            break

    if writer is None or not writer.isOpened():
        cap.release()
        raise gr.Error("Failed to create video writer for segment.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()
    cap.release()
    return output_path


# ─── Batch / Sequence Processing ─────────────────────────────────────────────

def process_batch(
    video_file,
    mask_input,
    frame_start,
    frame_end,
    input_is_linear,
    despill_strength,
    auto_despeckle,
    despeckle_size,
    refiner_scale,
    progress=gr.Progress(track_tqdm=False),
):
    """Process a video sequence with pre-made alpha hints."""

    if video_file is None:
        raise gr.Error("Upload a video file first.")
    if mask_input is None:
        raise gr.Error("Upload an alpha hint mask or generate one in the Hint Generator tab.")

    video_path = video_file if isinstance(video_file, str) else video_file.name if hasattr(video_file, "name") else str(video_file)
    ext = Path(video_path).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXT:
        raise gr.Error(f"Unsupported video format: {ext}. Use MP4, MOV, AVI, or MKV.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Failed to open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    start = max(0, int(frame_start) if frame_start else 0)
    end = min(total_frames - 1, int(frame_end) if frame_end and frame_end > 0 else total_frames - 1)
    num_frames = end - start + 1

    if num_frames <= 0:
        cap.release()
        raise gr.Error(f"Invalid frame range: {start}-{end} (video has {total_frames} frames)")

    mask_path = mask_input if isinstance(mask_input, str) else mask_input.name if hasattr(mask_input, "name") else str(mask_input)

    # Detect if mask is a video (per-frame alpha) or single image
    mask_ext = Path(mask_path).suffix.lower()
    mask_is_video = mask_ext in SUPPORTED_VIDEO_EXT
    mask_base = None
    mask_cap = None

    if mask_is_video:
        mask_cap = cv2.VideoCapture(mask_path)
        if not mask_cap.isOpened():
            raise gr.Error("Failed to open alpha hint video.")
        mask_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    else:
        mask_base = load_mask(mask_path)
        if mask_base is None:
            raise gr.Error("Failed to load mask image.")

    tmp_dir = _make_output_dir("batch")
    alpha_dir = os.path.join(tmp_dir, "alpha")
    fg_dir = os.path.join(tmp_dir, "foreground")
    comp_dir = os.path.join(tmp_dir, "composite")
    processed_dir = os.path.join(tmp_dir, "processed_rgba")
    for d in [alpha_dir, fg_dir, comp_dir, processed_dir]:
        os.makedirs(d, exist_ok=True)

    engine = get_engine()
    gallery_previews = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for i in range(num_frames):
        frame_idx = start + i
        progress((i + 1) / num_frames, desc=f"Processing frame {frame_idx} ({i+1}/{num_frames})")

        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Get mask for this frame
        if mask_is_video:
            mret, mask_bgr = mask_cap.read()
            if mret and mask_bgr is not None:
                mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 if mask_bgr.ndim == 3 else mask_bgr.astype(np.float32) / 255.0
            else:
                mask = np.ones(frame_rgb.shape[:2], dtype=np.float32)
        else:
            mask = mask_base.copy()

        if mask.shape[:2] != frame_rgb.shape[:2]:
            mask = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        try:
            result = engine.process_frame(
                image=frame_rgb, mask_linear=mask, refiner_scale=refiner_scale,
                input_is_linear=input_is_linear, fg_is_straight=True,
                despill_strength=despill_strength, auto_despeckle=auto_despeckle,
                despeckle_size=int(despeckle_size),
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch; torch.cuda.empty_cache()
                raise gr.Error(f"GPU OOM at frame {frame_idx}. Try a shorter range.")
            raise gr.Error(f"Failed at frame {frame_idx}: {e}")

        frame_name = f"{frame_idx:06d}"
        alpha = result["alpha"]
        if alpha.ndim == 2:
            alpha = alpha[:, :, np.newaxis]
        save_exr(os.path.join(alpha_dir, f"{frame_name}.exr"), alpha)
        save_exr(os.path.join(fg_dir, f"{frame_name}.exr"), result["fg"])
        save_png(os.path.join(comp_dir, f"{frame_name}.png"), result["comp"])
        save_exr(os.path.join(processed_dir, f"{frame_name}.exr"), result["processed"])


        step = max(1, num_frames // 20)
        if i % step == 0 or i == num_frames - 1:
            gallery_previews.append((to_uint8(result["comp"]), f"Frame {frame_idx}"))

    cap.release()
    if mask_cap is not None:
        mask_cap.release()

    # Compile output MP4s
    comp_mp4 = os.path.join(tmp_dir, "composite.mp4")
    alpha_mp4 = os.path.join(tmp_dir, "alpha_matte.mp4")
    frames_to_mp4(comp_dir, comp_mp4, fps)
    frames_to_mp4(alpha_dir, alpha_mp4, fps)

    status = f"Done — processed {num_frames} frames ({start}–{end})"
    return status, gallery_previews, comp_mp4, alpha_mp4

# ─── MatAnyone Hint Generation ───────────────────────────────────────────────

def generate_hints_segmented(
    video_file,
    current_mask,
    saved_masks_json,
    keyframes_json,
    expansion_px,
    max_size,
    n_warmup,
    r_erode,
    r_dilate,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate per-frame alpha hints with multi-keyframe segmented processing."""
    import base64

    if video_file is None:
        raise gr.Error("Upload a video file first.")

    video_path = video_file if isinstance(video_file, str) else video_file.name if hasattr(video_file, "name") else str(video_file)

    # Parse keyframes
    keyframes = json.loads(keyframes_json) if keyframes_json else {}

    # If no keyframes, fall back to current mask at frame 0 (backward compatible)
    if not keyframes:
        combined = combine_masks(saved_masks_json, current_mask)
        if combined is None:
            raise gr.Error("Create a mask first by clicking on the frame.")
        mask_u8 = (combined > 0.5).astype(np.uint8) * 255
        _, encoded = cv2.imencode(".png", mask_u8)
        mask_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        keyframes = {"0": {"mask_b64": mask_b64}}

    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    if total_frames <= 0:
        raise gr.Error("Failed to read video frame count.")

    # Sort keyframes and build segments
    sorted_kf = sorted(keyframes.keys(), key=int)
    segments = []
    for i, kf_str in enumerate(sorted_kf):
        # First segment always starts at frame 0
        seg_start = 0 if i == 0 else int(kf_str)
        seg_end = int(sorted_kf[i + 1]) - 1 if i + 1 < len(sorted_kf) else total_frames - 1
        segments.append((seg_start, seg_end, keyframes[kf_str]))

    tmp_dir = _make_output_dir("hints")
    unified_pha_dir = os.path.join(tmp_dir, "pha_unified")
    os.makedirs(unified_pha_dir, exist_ok=True)

    try:
        processor = load_matanyone()

        for seg_idx, (seg_start, seg_end, kf_data) in enumerate(segments):
            progress_base = seg_idx / len(segments)
            progress_range = 1.0 / len(segments)

            progress(progress_base + 0.02, desc=f"Segment {seg_idx+1}/{len(segments)}: extracting frames {seg_start}–{seg_end}...")

            # Decode keyframe mask
            buf = np.frombuffer(base64.b64decode(kf_data["mask_b64"]), dtype=np.uint8)
            mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

            # Save mask to temp file
            seg_tmp = tempfile.mkdtemp(prefix=f"ck_seg{seg_idx}_")
            mask_file = os.path.join(seg_tmp, "mask.png")
            cv2.imwrite(mask_file, mask)

            # Extract video segment to MP4
            seg_video = os.path.join(seg_tmp, "segment.mp4")
            extract_video_segment(video_path, seg_start, seg_end, seg_video, fps)

            # Clear temporal state between segments (keeps model weights loaded)
            processor.clear_memory()

            progress(progress_base + progress_range * 0.15, desc=f"Segment {seg_idx+1}/{len(segments)}: running MatAnyone...")

            # Process segment
            seg_out = os.path.join(seg_tmp, "output")
            fgr_path, alpha_path = processor.process_video(
                input_path=seg_video,
                mask_path=mask_file,
                output_path=seg_out,
                n_warmup=int(n_warmup),
                r_erode=int(r_erode),
                r_dilate=int(r_dilate),
                max_size=int(max_size),
                save_image=True,
            )

            progress(progress_base + progress_range * 0.85, desc=f"Segment {seg_idx+1}/{len(segments)}: copying alpha frames...")

            # Copy alpha frames to unified directory with global frame numbering
            seg_video_name = Path(seg_video).stem
            pha_dir = os.path.join(seg_out, seg_video_name, "pha")
            if os.path.isdir(pha_dir):
                pha_files = sorted([f for f in os.listdir(pha_dir) if f.endswith(".png")])
                for fi, fname in enumerate(pha_files):
                    global_frame = seg_start + fi
                    src = os.path.join(pha_dir, fname)
                    dst = os.path.join(unified_pha_dir, f"{global_frame:06d}.png")
                    shutil.copy2(src, dst)

            progress(progress_base + progress_range, desc=f"Segment {seg_idx+1}/{len(segments)} done.")

    except RuntimeError as e:
        unload_matanyone()
        if "out of memory" in str(e).lower():
            raise gr.Error("GPU OOM during hint generation. Try max_size=720.")
        raise gr.Error(f"MatAnyone failed: {e}")
    finally:
        unload_matanyone()

    # Check output
    all_pha = sorted([f for f in os.listdir(unified_pha_dir) if f.endswith(".png")])
    if not all_pha:
        raise gr.Error("No alpha hint frames generated.")

    # Post-process: expand every output frame if expansion requested
    exp = int(expansion_px) if expansion_px else 0
    if exp > 0:
        kernel_size = exp * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        for fname in all_pha:
            fpath = os.path.join(unified_pha_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                dilated = cv2.dilate(img, kernel)
                cv2.imwrite(fpath, dilated)

    # Gallery previews
    gallery = []
    step_g = max(1, len(all_pha) // 20)
    for i, fname in enumerate(all_pha):
        if i % step_g == 0 or i == len(all_pha) - 1:
            img = cv2.imread(os.path.join(unified_pha_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                gallery.append((img, f"Frame {i}"))

    # Compile unified alpha frames to MP4
    alpha_mp4 = os.path.join(tmp_dir, "alpha_hints.mp4")
    frames_to_mp4(unified_pha_dir, alpha_mp4, fps)

    exp_note = f" (expanded {exp}px)" if exp > 0 else ""
    status = f"Done — generated {len(all_pha)} alpha hint frames across {len(segments)} segment(s){exp_note}"
    preview = gallery[0][0] if gallery else None

    return preview, status, gallery, alpha_mp4, alpha_mp4


def process_pipeline(
    video_file,
    current_mask,
    saved_masks_json,
    keyframes_json,
    expansion_px,
    ma_max_size, ma_warmup, ma_erode, ma_dilate,
    frame_start, frame_end,
    input_is_linear, despill_strength, auto_despeckle, despeckle_size, refiner_scale,
    progress=gr.Progress(track_tqdm=True),
):
    """Full pipeline: SAM mask -> MatAnyone hints -> CorridorKey keying."""
    import base64

    if video_file is None:
        raise gr.Error("Upload a video file first.")

    video_path = video_file if isinstance(video_file, str) else video_file.name if hasattr(video_file, "name") else str(video_file)
    tmp_dir = _make_output_dir("pipeline")

    # Parse keyframes for segmented approach
    keyframes = json.loads(keyframes_json) if keyframes_json else {}

    # If no keyframes, fall back to current mask at frame 0
    if not keyframes:
        combined = combine_masks(saved_masks_json, current_mask)
        if combined is None:
            raise gr.Error("Create a mask first by clicking on the frame.")
        mask_u8 = (combined > 0.5).astype(np.uint8) * 255
        _, encoded = cv2.imencode(".png", mask_u8)
        mask_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        keyframes = {"0": {"mask_b64": mask_b64}}

    # Get video info for segments
    cap_info = cv2.VideoCapture(video_path)
    vid_total = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap_info.get(cv2.CAP_PROP_FPS) or 24.0
    cap_info.release()

    # Build segments from keyframes
    sorted_kf = sorted(keyframes.keys(), key=int)
    segments = []
    for i, kf_str in enumerate(sorted_kf):
        seg_start = 0 if i == 0 else int(kf_str)
        seg_end = int(sorted_kf[i + 1]) - 1 if i + 1 < len(sorted_kf) else vid_total - 1
        segments.append((seg_start, seg_end, keyframes[kf_str]))

    unified_pha_dir = os.path.join(tmp_dir, "pha_unified")
    os.makedirs(unified_pha_dir, exist_ok=True)

    # ── Stage 1: MatAnyone (segmented) ──
    progress(0.0, desc="Stage 1/2: Loading MatAnyone...")
    try:
        processor = load_matanyone()
        for seg_idx, (seg_start, seg_end, kf_data) in enumerate(segments):
            seg_frac = (seg_idx + 1) / len(segments) * 0.35
            progress(0.02 + seg_frac * 0.5, desc=f"Stage 1/2: Segment {seg_idx+1}/{len(segments)} (frames {seg_start}–{seg_end})...")

            buf = np.frombuffer(base64.b64decode(kf_data["mask_b64"]), dtype=np.uint8)
            mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

            seg_tmp = tempfile.mkdtemp(prefix=f"ck_pipe_seg{seg_idx}_")
            mask_file = os.path.join(seg_tmp, "mask.png")
            cv2.imwrite(mask_file, mask)

            seg_video = os.path.join(seg_tmp, "segment.mp4")
            extract_video_segment(video_path, seg_start, seg_end, seg_video, vid_fps)

            processor.clear_memory()

            seg_out = os.path.join(seg_tmp, "output")
            processor.process_video(
                input_path=seg_video, mask_path=mask_file, output_path=seg_out,
                n_warmup=int(ma_warmup), r_erode=int(ma_erode),
                r_dilate=int(ma_dilate), max_size=int(ma_max_size), save_image=True,
            )

            seg_video_name = Path(seg_video).stem
            pha_src = os.path.join(seg_out, seg_video_name, "pha")
            if os.path.isdir(pha_src):
                for fi, fname in enumerate(sorted(f for f in os.listdir(pha_src) if f.endswith(".png"))):
                    shutil.copy2(os.path.join(pha_src, fname),
                                 os.path.join(unified_pha_dir, f"{seg_start + fi:06d}.png"))

    except RuntimeError as e:
        unload_matanyone()
        if "out of memory" in str(e).lower():
            raise gr.Error("GPU OOM during MatAnyone. Try max_size=720.")
        raise gr.Error(f"MatAnyone failed: {e}")
    finally:
        unload_matanyone()

    progress(0.4, desc="Stage 1/2: Hints generated. Loading CorridorKey...")

    pha_files = sorted([f for f in os.listdir(unified_pha_dir) if f.endswith(".png")])
    if not pha_files:
        raise gr.Error("No alpha hint frames generated.")

    # Post-process: expand every output frame if expansion requested
    exp = int(expansion_px) if expansion_px else 0
    if exp > 0:
        kernel_size = exp * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        for fname in pha_files:
            fpath = os.path.join(unified_pha_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                cv2.imwrite(fpath, cv2.dilate(img, kernel))

    # ── Stage 2: CorridorKey ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Failed to open video.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    start = max(0, int(frame_start) if frame_start else 0)
    end = min(total_frames - 1, int(frame_end) if frame_end and frame_end > 0 else total_frames - 1)
    num_frames = end - start + 1
    if num_frames <= 0:
        cap.release()
        raise gr.Error(f"Invalid frame range.")

    out_base = os.path.join(tmp_dir, "output")
    alpha_dir_out = os.path.join(out_base, "alpha")
    fg_dir = os.path.join(out_base, "foreground")
    comp_dir = os.path.join(out_base, "composite")
    processed_dir = os.path.join(out_base, "processed_rgba")
    for d in [alpha_dir_out, fg_dir, comp_dir, processed_dir]:
        os.makedirs(d, exist_ok=True)

    engine = get_engine()
    gallery_previews = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for i in range(num_frames):
        frame_idx = start + i
        frac = 0.4 + 0.55 * ((i + 1) / num_frames)
        progress(frac, desc=f"Stage 2/2: CorridorKey frame {frame_idx} ({i+1}/{num_frames})")

        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        hint_idx = frame_idx - start
        if hint_idx < len(pha_files):
            hint_path = os.path.join(unified_pha_dir, pha_files[hint_idx])
            mask = cv2.imread(hint_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32) / 255.0 if mask is not None else np.ones(frame_rgb.shape[:2], dtype=np.float32)
        else:
            mask = np.ones(frame_rgb.shape[:2], dtype=np.float32)

        if mask.shape[:2] != frame_rgb.shape[:2]:
            mask = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        try:
            result = engine.process_frame(
                image=frame_rgb, mask_linear=mask, refiner_scale=refiner_scale,
                input_is_linear=input_is_linear, fg_is_straight=True,
                despill_strength=despill_strength, auto_despeckle=auto_despeckle,
                despeckle_size=int(despeckle_size),
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch; torch.cuda.empty_cache()
                raise gr.Error(f"GPU OOM at frame {frame_idx}.")
            raise gr.Error(f"Failed at frame {frame_idx}: {e}")

        frame_name = f"{frame_idx:06d}"
        alpha = result["alpha"]
        if alpha.ndim == 2:
            alpha = alpha[:, :, np.newaxis]
        save_exr(os.path.join(alpha_dir_out, f"{frame_name}.exr"), alpha)
        save_exr(os.path.join(fg_dir, f"{frame_name}.exr"), result["fg"])
        save_png(os.path.join(comp_dir, f"{frame_name}.png"), result["comp"])
        save_exr(os.path.join(processed_dir, f"{frame_name}.exr"), result["processed"])


        step_g = max(1, num_frames // 20)
        if i % step_g == 0 or i == num_frames - 1:
            gallery_previews.append((to_uint8(result["comp"]), f"Frame {frame_idx}"))

    cap.release()

    # Compile output MP4s
    comp_mp4 = os.path.join(tmp_dir, "composite.mp4")
    alpha_mp4 = os.path.join(tmp_dir, "alpha_matte.mp4")
    frames_to_mp4(comp_dir, comp_mp4, fps)
    frames_to_mp4(alpha_dir_out, alpha_mp4, fps)

    progress(1.0, desc="Done!")
    status = f"Pipeline complete — {len(pha_files)} hints + {num_frames} frames keyed ({start}–{end})"
    return status, gallery_previews, comp_mp4, alpha_mp4

# ─── Theme + CSS ─────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#ecfdf5", c100="#d1fae5", c200="#a7f3d0", c300="#6ee7b7",
        c400="#34d399", c500="#10b981", c600="#059669", c700="#047857",
        c800="#065f46", c900="#064e3b", c950="#022c22",
    ),
    secondary_hue="gray",
    neutral_hue="gray",
    font=("Inter", "system-ui", "sans-serif"),
).set(
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    block_background_fill="#141414",
    block_background_fill_dark="#141414",
    block_border_width="1px",
    block_border_color="#1e1e1e",
    block_border_color_dark="#1e1e1e",
    block_label_background_fill="#1a1a1a",
    block_label_background_fill_dark="#1a1a1a",
    block_title_text_color="#e0e0e0",
    block_title_text_color_dark="#e0e0e0",
    body_text_color="#d0d0d0",
    body_text_color_dark="#d0d0d0",
    body_text_color_subdued="#888888",
    body_text_color_subdued_dark="#888888",
    button_primary_background_fill="#059669",
    button_primary_background_fill_dark="#059669",
    button_primary_background_fill_hover="#047857",
    button_primary_background_fill_hover_dark="#047857",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#1e1e1e",
    button_secondary_background_fill_dark="#1e1e1e",
    button_secondary_text_color="#d0d0d0",
    button_secondary_text_color_dark="#d0d0d0",
    input_background_fill="#1a1a1a",
    input_background_fill_dark="#1a1a1a",
    input_border_color="#2a2a2a",
    input_border_color_dark="#2a2a2a",
    input_placeholder_color="#666666",
    input_placeholder_color_dark="#666666",
    slider_color="#10b981",
    slider_color_dark="#10b981",
    checkbox_background_color="#1a1a1a",
    checkbox_background_color_dark="#1a1a1a",
    checkbox_border_color="#2a2a2a",
    checkbox_border_color_dark="#2a2a2a",
    checkbox_background_color_selected="#059669",
    checkbox_background_color_selected_dark="#059669",
    shadow_drop="none",
    shadow_drop_lg="none",
)

custom_css = """
:root, .dark { color-scheme: dark; }
body { background: #0a0a0a !important; }

.header-bar {
    background: linear-gradient(135deg, #064e3b 0%, #0a0a0a 60%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 8px;
}
.header-bar h1 { color: #ecfdf5 !important; font-size: 1.8rem !important; margin: 0 0 4px 0 !important; letter-spacing: -0.02em; }
.header-bar p { color: #6ee7b7 !important; margin: 0 !important; font-size: 0.9rem; }

.gpu-badge { background: #141414; border: 1px solid #1e1e1e; border-radius: 8px; padding: 8px 14px; font-size: 0.82rem; color: #a0a0a0; }
.gpu-badge strong { color: #10b981; }

.process-btn { font-size: 1.05rem !important; font-weight: 600 !important; letter-spacing: 0.02em; min-height: 48px !important; }
.process-btn:hover { box-shadow: 0 0 20px rgba(16, 185, 129, 0.3) !important; }

.tabs .tab-nav button { font-weight: 500 !important; font-size: 0.95rem !important; }
.tabs .tab-nav button.selected { border-color: #10b981 !important; color: #10b981 !important; }

.status-text { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #6ee7b7; }

.help-content { line-height: 1.7; color: #c0c0c0; }
.help-content h3 { color: #10b981 !important; margin-top: 1.2em; }
.help-content code { background: #1e1e1e; padding: 2px 6px; border-radius: 4px; color: #6ee7b7; }
.help-content table { width: 100%; border-collapse: collapse; }
.help-content th { text-align: left; color: #10b981; padding: 8px; border-bottom: 1px solid #2a2a2a; }
.help-content td { padding: 8px; border-bottom: 1px solid #1a1a1a; }

.mask-controls { gap: 8px !important; }
.mask-controls button { min-height: 40px !important; }

.step-label { color: #10b981 !important; font-weight: 600; font-size: 1.05rem; margin-bottom: 4px; }
"""

force_dark_js = """
() => {
    document.body.classList.add('dark');
    document.documentElement.style.colorScheme = 'dark';
}
"""

# ─── Gradio Layout ───────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore", message=".*parameters have been moved.*Gradio 6.0.*")

with gr.Blocks(
    theme=theme,
    css=custom_css,
    title="CorridorKey — AI Green Screen Keyer",
    js=force_dark_js,
) as app:

    # Header
    with gr.Row(elem_classes="header-bar"):
        with gr.Column(scale=4):
            gr.HTML("<h1>CorridorKey</h1><p>Neural Network Green Screen Keyer by Corridor Digital</p>")
        with gr.Column(scale=2):
            gpu_status = gr.Markdown(value=check_gpu_status(), elem_classes="gpu-badge")

    with gr.Tabs():

        # ═══════════════════════════════════════════════════════════════════
        # TAB: Hint Generator (Interactive SAM + MatAnyone)
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("Hint Generator", id="hints"):
            gr.Markdown(
                "**Step 1:** Upload video & click Load. "
                "**Step 2:** Click on the first frame to select your subject (green = include, red = exclude). "
                "**Step 3:** Generate per-frame alpha hints with MatAnyone.",
                elem_classes="status-text",
            )

            # Hidden state components
            original_frame_state = gr.State(value=None)     # original RGB numpy array
            current_mask_state = gr.State(value=None)        # current SAM mask (numpy bool)
            click_state = gr.State(value="{}")               # JSON: {points: [[x,y],...], labels: [1,0,...]}
            saved_masks_state = gr.State(value="[]")         # JSON: list of base64-encoded mask PNGs
            keyframes_state = gr.State(value="{}")           # JSON: {"0": {"mask_b64": "...", "sub_masks_b64": [...]}, ...}
            video_total_frames = gr.State(value=0)           # total frame count for slider max
            video_fps_state = gr.State(value=24.0)           # fps for segment MP4 writing
            raw_mask_state = gr.State(value=None)            # raw SAM mask before expansion (for re-rolling)

            with gr.Row():
                # ── Left: Upload + Mask Controls ──
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1: Upload Video", elem_classes="step-label")
                    hint_video = gr.File(
                        label="Video File",
                        file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
                    )
                    load_btn = gr.Button("Load Video", variant="secondary")
                    hint_video_status = gr.Textbox(label="Video Info", interactive=False, elem_classes="status-text")

                    gr.Markdown("### Frame Navigation", elem_classes="step-label")
                    frame_slider = gr.Slider(
                        label="Frame", minimum=0, maximum=100, step=1, value=0,
                        info="Scrub to any frame to create keyframe masks.",
                    )
                    frame_num_display = gr.Number(label="Frame #", value=0, precision=0, minimum=0)

                    gr.Markdown("### Step 2: Click to Mask", elem_classes="step-label")
                    point_mode = gr.Radio(
                        choices=["Positive (include)", "Negative (exclude)"],
                        value="Positive (include)",
                        label="Click Mode",
                        info="Green clicks select the subject. Red clicks remove areas.",
                    )
                    with gr.Row(elem_classes="mask-controls"):
                        clear_btn = gr.Button("Clear Clicks", variant="secondary", scale=1)
                        add_mask_btn = gr.Button("Add Mask", variant="secondary", scale=1)
                    expansion_slider = gr.Slider(
                        label="Mask Expansion (px)", minimum=0, maximum=80, step=1, value=0,
                        info="Dilate mask to catch hair, motion blur. Re-roll anytime — raw mask is preserved.",
                    )
                    mask_info = gr.Textbox(
                        label="Masks",
                        value="No masks saved yet. Click on the frame, then Add Mask.",
                        interactive=False,
                        elem_classes="status-text",
                    )

                    gr.Markdown("### Keyframes", elem_classes="step-label")
                    with gr.Row(elem_classes="mask-controls"):
                        save_kf_btn = gr.Button("Save Keyframe", variant="secondary", scale=1)
                        delete_kf_btn = gr.Button("Delete Keyframe", variant="secondary", scale=1)
                    keyframe_info = gr.Textbox(
                        label="Keyframes",
                        value="No keyframes. Save masks at different frames for multi-keyframe hints.",
                        interactive=False,
                        elem_classes="status-text",
                    )

                    gr.Markdown("### Step 3: Generate Hints", elem_classes="step-label")
                    with gr.Accordion("MatAnyone Settings", open=False):
                        hint_max_size = gr.Slider(
                            label="Processing Resolution", minimum=480, maximum=1440, step=10, value=1080,
                            info="720 = fast/low VRAM, 1080 = balanced. No benefit beyond 1080 for hints.",
                        )
                        hint_warmup = gr.Number(label="Warmup Frames", value=10, precision=0, minimum=0, maximum=50)
                        hint_erode = gr.Number(label="Mask Erosion (px)", value=10, precision=0, minimum=0, maximum=50)
                        hint_dilate = gr.Number(label="Mask Dilation (px)", value=10, precision=0, minimum=0, maximum=50)

                    generate_btn = gr.Button("Generate Hints", variant="primary", elem_classes="process-btn")

                # ── Right: Interactive Frame + Results ──
                with gr.Column(scale=2):
                    frame_display = gr.Image(
                        label="Click on the frame to select your subject",
                        type="numpy",
                        interactive=True,
                        height=450,
                    )
                    hint_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-text")
                    hint_gallery = gr.Gallery(
                        label="Alpha Hint Previews",
                        columns=5, rows=2, height=200, object_fit="contain",
                    )
                    hint_video_out = gr.Video(label="Alpha Hints Video", interactive=False)
                    hint_alpha_path = gr.Textbox(visible=False)  # hidden: path for batch tab

                    send_to_batch_btn = gr.Button(
                        "Send Alpha Video to Keyer Tab",
                        variant="secondary",
                        interactive=False,
                    )

            # ── Wire Hint Generator Events ──

            # Load video -> extract first frame + set slider max
            load_btn.click(
                fn=extract_first_frame,
                inputs=[hint_video],
                outputs=[frame_display, original_frame_state, hint_video_status, video_total_frames, video_fps_state],
            ).then(
                # Reset all state on new video
                fn=lambda: ("{}", "[]", None, None, "No masks saved yet.", "{}", "No keyframes.", 0, 0),
                outputs=[click_state, saved_masks_state, current_mask_state, raw_mask_state, mask_info,
                         keyframes_state, keyframe_info, frame_num_display, expansion_slider],
            ).then(
                # Update slider maximum to total frames - 1
                fn=lambda t: gr.update(maximum=max(0, int(t) - 1), value=0),
                inputs=[video_total_frames],
                outputs=[frame_slider],
            )

            # Frame slider -> navigate to frame, load keyframe mask if exists
            frame_slider.release(
                fn=on_frame_slider_change,
                inputs=[hint_video, frame_slider, keyframes_state],
                outputs=[frame_display, original_frame_state, click_state, current_mask_state,
                         saved_masks_state, hint_video_status],
            ).then(
                # Sync frame number display + reset raw mask and expansion
                fn=lambda s: (int(s), None, 0),
                inputs=[frame_slider],
                outputs=[frame_num_display, raw_mask_state, expansion_slider],
            )

            # Frame number input -> jump to frame
            frame_num_display.submit(
                fn=on_frame_slider_change,
                inputs=[hint_video, frame_num_display, keyframes_state],
                outputs=[frame_display, original_frame_state, click_state, current_mask_state,
                         saved_masks_state, hint_video_status],
            ).then(
                fn=lambda n: (int(n), None, 0),
                inputs=[frame_num_display],
                outputs=[frame_slider, raw_mask_state, expansion_slider],
            )

            # Click on frame -> SAM prediction (stores raw mask for expansion re-rolling)
            frame_display.select(
                fn=sam_click,
                inputs=[original_frame_state, click_state, point_mode],
                outputs=[frame_display, click_state, current_mask_state, raw_mask_state],
            ).then(
                # Auto-apply expansion if slider > 0
                fn=apply_expansion,
                inputs=[raw_mask_state, original_frame_state, click_state, expansion_slider],
                outputs=[frame_display, current_mask_state],
            )

            # Expansion slider -> re-roll from raw mask
            expansion_slider.release(
                fn=apply_expansion,
                inputs=[raw_mask_state, original_frame_state, click_state, expansion_slider],
                outputs=[frame_display, current_mask_state],
            )

            # Clear clicks
            clear_btn.click(
                fn=clear_clicks,
                inputs=[original_frame_state],
                outputs=[frame_display, click_state, current_mask_state, raw_mask_state],
            )

            # Add mask (sub-mask for current frame — saves expanded version)
            add_mask_btn.click(
                fn=add_sub_mask,
                inputs=[current_mask_state, saved_masks_state],
                outputs=[saved_masks_state, mask_info],
            ).then(
                # Reset clicks for next mask but keep saved masks
                fn=clear_clicks,
                inputs=[original_frame_state],
                outputs=[frame_display, click_state, current_mask_state, raw_mask_state],
            )

            # Save keyframe at current frame (saves expanded mask)
            save_kf_btn.click(
                fn=save_keyframe,
                inputs=[frame_num_display, current_mask_state, saved_masks_state, keyframes_state],
                outputs=[keyframes_state, keyframe_info],
            ).then(
                fn=clear_clicks,
                inputs=[original_frame_state],
                outputs=[frame_display, click_state, current_mask_state, raw_mask_state],
            )

            # Delete keyframe at current frame
            delete_kf_btn.click(
                fn=delete_keyframe,
                inputs=[frame_num_display, keyframes_state],
                outputs=[keyframes_state, keyframe_info],
            ).then(
                # Refresh display (reload frame without mask)
                fn=on_frame_slider_change,
                inputs=[hint_video, frame_num_display, keyframes_state],
                outputs=[frame_display, original_frame_state, click_state, current_mask_state,
                         saved_masks_state, hint_video_status],
            ).then(
                fn=lambda: (None, 0),
                outputs=[raw_mask_state, expansion_slider],
            )

            # Generate hints (segmented multi-keyframe)
            generate_btn.click(
                fn=generate_hints_segmented,
                inputs=[hint_video, current_mask_state, saved_masks_state, keyframes_state,
                        expansion_slider, hint_max_size, hint_warmup, hint_erode, hint_dilate],
                outputs=[frame_display, hint_status, hint_gallery, hint_video_out, hint_alpha_path],
            ).then(
                fn=lambda p: gr.update(interactive=bool(p)),
                inputs=[hint_alpha_path],
                outputs=[send_to_batch_btn],
            )

        # ═══════════════════════════════════════════════════════════════════
        # TAB: Batch / Sequence
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("Keyer", id="batch"):
            gr.Markdown(
                "Process a video sequence with an alpha hint mask. "
                "Use the **Hint Generator** to create one interactively, or upload your own.",
                elem_classes="status-text",
            )
            with gr.Row():
                with gr.Column(scale=1):
                    batch_video = gr.File(
                        label="Video File (MP4, MOV, AVI, MKV)",
                        file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
                    )
                    batch_mask = gr.File(
                        label="Alpha Hint Mask (single image or alpha video from Hint Generator)",
                        file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".bmp", ".mp4"],
                    )

                    with gr.Row():
                        batch_start = gr.Number(label="Start Frame", value=0, precision=0, minimum=0)
                        batch_end = gr.Number(label="End Frame (0 = all)", value=0, precision=0, minimum=0)

                    with gr.Accordion("CorridorKey Settings", open=False):
                        batch_linear = gr.Checkbox(label="Input is Linear", value=False,
                            info="Check for linear color (EXR). Leave unchecked for sRGB (PNG/MP4).")
                        batch_despill = gr.Slider(label="Despill Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                            info="Green spill removal. 0 = none, 1 = standard, 2 = aggressive.")
                        batch_despeckle = gr.Checkbox(label="Auto Despeckle", value=True,
                            info="Remove small noise artifacts from the matte.")
                        batch_despeckle_size = gr.Slider(label="Despeckle Size", minimum=50, maximum=2000, step=25, value=400,
                            info="Max artifact size in pixels to remove.")
                        batch_refiner = gr.Slider(label="Refiner Scale", minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                            info="Edge refinement intensity. 1.0 = default.")

                    batch_btn = gr.Button("Process Batch", variant="primary", elem_classes="process-btn")

                with gr.Column(scale=2):
                    batch_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-text")
                    batch_gallery = gr.Gallery(label="Composite Previews", columns=5, rows=2, height=240, object_fit="contain")
                    with gr.Row():
                        batch_comp_video = gr.Video(label="Composite", interactive=False)
                        batch_alpha_video = gr.Video(label="Alpha Matte", interactive=False)

            batch_btn.click(
                fn=process_batch,
                inputs=[batch_video, batch_mask, batch_start, batch_end,
                        batch_linear, batch_despill, batch_despeckle, batch_despeckle_size, batch_refiner],
                outputs=[batch_status, batch_gallery, batch_comp_video, batch_alpha_video],
            )

        # ═══════════════════════════════════════════════════════════════════
        # TAB: Full Pipeline
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("Full Pipeline", id="pipeline"):
            gr.Markdown(
                "**One-click end-to-end:** Uses your mask from the Hint Generator. "
                "MatAnyone generates per-frame hints, then CorridorKey produces final mattes. "
                "Models load sequentially to share VRAM.",
                elem_classes="status-text",
            )
            with gr.Row():
                with gr.Column(scale=1):
                    pipe_video = gr.File(label="Video File", file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"])

                    # Pipeline uses the same mask state from Hint Generator tab
                    pipe_info = gr.Markdown(
                        "This tab uses the mask you created in **Hint Generator**. "
                        "Go there first to click-create your mask, then come here to run the full pipeline."
                    )

                    with gr.Row():
                        pipe_start = gr.Number(label="Start Frame", value=0, precision=0, minimum=0)
                        pipe_end = gr.Number(label="End Frame (0 = all)", value=0, precision=0, minimum=0)

                    with gr.Accordion("MatAnyone Settings", open=False):
                        pipe_ma_size = gr.Slider(label="Hint Resolution", minimum=480, maximum=1440, step=10, value=1080)
                        pipe_ma_warmup = gr.Number(label="Warmup Frames", value=10, precision=0, minimum=0, maximum=50)
                        pipe_ma_erode = gr.Number(label="Mask Erosion (px)", value=10, precision=0, minimum=0, maximum=50)
                        pipe_ma_dilate = gr.Number(label="Mask Dilation (px)", value=10, precision=0, minimum=0, maximum=50)

                    with gr.Accordion("CorridorKey Settings", open=False):
                        pipe_linear = gr.Checkbox(label="Input is Linear", value=False)
                        pipe_despill = gr.Slider(label="Despill Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0)
                        pipe_despeckle = gr.Checkbox(label="Auto Despeckle", value=True)
                        pipe_despeckle_size = gr.Slider(label="Despeckle Size", minimum=50, maximum=2000, step=25, value=400)
                        pipe_refiner = gr.Slider(label="Refiner Scale", minimum=0.0, maximum=2.0, step=0.05, value=1.0)

                    pipe_btn = gr.Button("Run Full Pipeline", variant="primary", elem_classes="process-btn")

                with gr.Column(scale=2):
                    pipe_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-text")
                    pipe_gallery = gr.Gallery(label="Composite Previews", columns=5, rows=2, height=240, object_fit="contain")
                    with gr.Row():
                        pipe_comp_video = gr.Video(label="Composite", interactive=False)
                        pipe_alpha_video = gr.Video(label="Alpha Matte", interactive=False)

            pipe_btn.click(
                fn=process_pipeline,
                inputs=[pipe_video, current_mask_state, saved_masks_state, keyframes_state,
                        expansion_slider, pipe_ma_size, pipe_ma_warmup, pipe_ma_erode, pipe_ma_dilate,
                        pipe_start, pipe_end,
                        pipe_linear, pipe_despill, pipe_despeckle, pipe_despeckle_size, pipe_refiner],
                outputs=[pipe_status, pipe_gallery, pipe_comp_video, pipe_alpha_video],
            )

        # ═══════════════════════════════════════════════════════════════════
        # TAB: Help
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("Help", id="help"):
            gr.Markdown(
                """
## How It Works

CorridorKey is a neural network keyer that takes **two inputs**: RGB footage + a rough alpha hint mask.
The AI model (GreenFormer) produces a **high-quality alpha matte** with clean edges, proper hair detail, and minimal artifacts.

**MatAnyone** generates the alpha hints automatically from a mask, propagating it across all video frames.
**Multi-keyframe masking** lets you create SAM masks at different frames — each segment gets a clean anchor point for better tracking through complex motion.

---

## Workflow

### Hint Generator (Recommended Start)
1. Upload your green screen video and click **Load Video**
2. Click on the first frame to select your subject:
   - **Green dots (Positive)** = include this area
   - **Red dots (Negative)** = exclude this area
   - SAM updates the mask in real-time after each click
3. Click **Add Mask** to save and start a new sub-mask (for complex subjects)
4. Click **Save Keyframe** to lock this mask at the current frame
5. Use the **Frame slider** to scrub to other frames where the subject changes significantly
6. Create new SAM masks and save keyframes at those frames too
7. Click **Generate Hints** — MatAnyone processes each segment independently (~4-6 GB VRAM)
8. Download hints, or use **Send to Batch** / **Full Pipeline** to continue keying

> **Tip:** For simple shots, you can skip keyframes entirely — just click a mask on frame 0 and generate. Multi-keyframe is for shots where the subject changes shape dramatically (limbs disappearing, turns, etc.).

### Keyer
1. Upload video + a pre-made alpha hint mask (from Hint Generator or painted manually)
2. Set frame range, adjust CorridorKey settings
3. Click **Process Batch** — CorridorKey keys each frame (~22 GB VRAM)
4. Download ZIP with EXR sequences (alpha/, foreground/, composite/, processed_rgba/)

### Full Pipeline (One-Click)
1. Create your mask in the **Hint Generator** tab first
2. Upload the same video in Full Pipeline, set your settings
3. Click **Run Full Pipeline** — MatAnyone generates hints, then CorridorKey keys everything
4. Models load/unload sequentially to share VRAM on a single GPU

---

## Settings Reference

### CorridorKey Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Input is Linear** | Off | Enable for linear color space (EXR). Leave off for sRGB (PNG, MP4). |
| **Despill Strength** | 1.0 | Green spill removal. 0 = none, 1 = standard, 2 = aggressive. |
| **Auto Despeckle** | On | Removes small noise blobs from the alpha matte. |
| **Despeckle Size** | 400 | Max artifact size (px) to remove. |
| **Refiner Scale** | 1.0 | Edge refinement intensity. Experimental. |

### MatAnyone Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Processing Resolution** | 1080 | Resolution cap (shorter side). 720 = fast, 1080 = balanced. |
| **Warmup Frames** | 10 | Stabilization iterations on first frame. |
| **Mask Erosion** | 10px | Shrinks input mask to prevent background bleed. |
| **Mask Dilation** | 10px | Expands input mask for full subject coverage. |

---

## Tips

- **Click precisely** on the subject for best SAM results. A few well-placed clicks beat many random ones.
- **Negative clicks** are powerful — use them to exclude background areas that SAM incorrectly includes.
- **Add Mask** lets you build complex selections from multiple sub-masks (e.g., person + held object).
- **4K+ footage** is resized to 2048x2048 internally by CorridorKey, then upscaled back with Lanczos4.
- **VRAM budget:** SAM (~1 GB) + MatAnyone (~4-6 GB at 1080p) + CorridorKey (~22 GB) all load sequentially.
- **EXR outputs** are in linear color space with premultiplied alpha — ready for Nuke/AE/Fusion.
- **Multi-keyframe** is ideal for shots with big shape changes. Each keyframe gives MatAnyone a fresh anchor, preventing drift.
- **First-frame mask** can also be painted in Photoshop/GIMP or generated with [SAM2 online](https://huggingface.co/spaces/fffiloni/SAM2-Image-Predictor).
                """,
                elem_classes="help-content",
            )

    # ── Cross-tab wiring: Send alpha to Batch tab ──
    # When the user clicks "Send Alpha Video to Batch Tab", populate the batch mask input
    # (Gradio File components can't be set programmatically from a path easily,
    #  so we show the path and instruct the user to use it)

    send_to_batch_btn.click(
        fn=lambda p: p if p else None,
        inputs=[hint_alpha_path],
        outputs=[batch_mask],
    )

# ─── Launch ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
