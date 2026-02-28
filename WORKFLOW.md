# CorridorKey GUI — Workflow Guide

## Overview

This tool chains three AI models to key green screen video:

```
SAM (click-to-mask) → MatAnyone (propagate mask across frames) → CorridorKey (produce final key)
```

You only need to click on a few frames. The AI handles the rest.

---

## Quick Start: Full Pipeline (Easiest)

This is the simplest path from raw footage to keyed EXR output.

### Step 1: Create Your Mask

Open the **Hint Generator** tab.

1. Click **Upload Video** and select your green screen footage
2. Click **Load Video** — the first frame appears
3. Click on your subject in the frame:
   - Left panel defaults to **Positive** mode (green dots) — click areas to **include**
   - Switch to **Negative** mode (red dots) to **exclude** areas SAM grabs incorrectly
   - The blue mask overlay updates after each click
4. If your subject has separate parts (person + held prop), click **Add Mask** to start a new sub-mask, then click on the next part
5. Adjust the **Mask Expansion** slider if you want to grow the mask outward (helps capture hair/motion blur)

### Step 2: Run the Pipeline

Switch to the **Full Pipeline** tab.

1. Upload the same video
2. Leave defaults or tweak settings (see Settings section below)
3. Click **Run Full Pipeline**
4. Wait — MatAnyone generates hints first (~4-6 GB VRAM), then CorridorKey keys every frame (~22 GB VRAM)
5. Output appears in `output/` with timestamped folder

### Step 3: Use Your Output

Each frame produces four files:

| File | What It Is | Use For |
|------|-----------|---------|
| `*_processed.exr` | Premultiplied linear RGBA | **This is your main output.** Drop straight into Nuke/Fusion/AE. |
| `*_alpha.exr` | Linear alpha matte only | Separate matte pass for custom compositing |
| `*_fg.exr` | sRGB foreground (straight) | Color reference / manual recomp |
| `*_comp.png` | Preview composite on checkerboard | Quick visual check |

---

## Multi-Keyframe Workflow (Complex Shots)

Use this when your subject changes significantly during the shot — limbs disappearing behind motion blur, turns, major pose changes. Each keyframe gives MatAnyone a fresh reference point.

### When You Need This

- Subject's silhouette changes dramatically mid-shot
- Parts of the body disappear/reappear (arm behind back, leg kick, etc.)
- MatAnyone's single-mask propagation starts drifting or losing detail

### How To Do It

1. **Load your video** in the Hint Generator tab
2. **Create a mask on frame 0** (or wherever the shot starts) — click SAM points as usual
3. Click **Save Keyframe** — the info box shows "1 keyframe(s): Frame 0"
4. **Scrub the Frame Slider** to the point where the subject changes shape
5. **Create a new SAM mask** for that frame — click points on the new pose
6. Click **Save Keyframe** again — info updates to show both keyframes
7. Repeat for any other trouble spots
8. Click **Generate Hints** (or use Full Pipeline)

The system splits your video into segments at each keyframe boundary. Each segment runs MatAnyone independently with its own mask, then the results are stitched together.

**Example:** 500-frame video with keyframes at 0, 150, 300:
- Frames 0–149: uses mask from frame 0
- Frames 150–299: uses mask from frame 150
- Frames 300–499: uses mask from frame 300

### Tips

- You can navigate back to a saved keyframe and its mask will reload for editing
- **Delete Keyframe** removes the current frame's keyframe
- Re-saving a keyframe at the same frame overwrites the previous mask
- For simple shots, skip keyframes entirely — a single mask on frame 0 works great

---

## Batch / Keyer Workflow (Bring Your Own Masks)

If you already have alpha hint masks from Nuke, After Effects, or another tool:

1. Open the **Batch / Sequence** tab
2. Upload your green screen video
3. Upload your mask — either:
   - A **single image** (applied to all frames)
   - A **video file** (per-frame mask, must match frame count)
4. Set your frame range and CorridorKey settings
5. Click **Process Batch**
6. Output EXR sequences appear in `output/`

---

## Settings Reference

### CorridorKey Settings

| Setting | Default | What It Does |
|---------|---------|-------------|
| **Despill Strength** | 1.0 | Removes green light spilling onto skin/clothes. 0 = off, 1 = standard, 2 = aggressive. Start at 1.0. |
| **Auto Despeckle** | On | Cleans up small floating noise blobs in the alpha. |
| **Despeckle Size** | 400 | Pixel area threshold — blobs smaller than this get removed. |
| **Refiner Scale** | 1.0 | Edge detail refinement. 1.0 is default. Experimental — tweak if edges look soft or crunchy. |
| **Input is Linear** | Off | Only enable if your source is linear EXR. Leave off for MP4/PNG/ProRes. |

### MatAnyone Settings

| Setting | Default | What It Does |
|---------|---------|-------------|
| **Processing Resolution** | 1080 | Shorter side cap. 720 = faster/less VRAM. 1080 = better quality. |
| **Warmup Frames** | 10 | Iterations on the first frame to stabilize the mask. More = steadier start. |
| **Mask Erosion** | 10px | Shrinks the input mask edges. Prevents background bleed at the boundary. |
| **Mask Dilation** | 10px | Expands the input mask edges. Ensures full subject coverage. |

### Mask Expansion

| Setting | Default | What It Does |
|---------|---------|-------------|
| **Expansion (px)** | 0 | Grows the SAM mask outward by this many pixels. Applied to all output frames. Good for capturing hair wisps and motion blur that SAM's edge misses. |

---

## VRAM Budget

Models load one at a time. You only need enough VRAM for the largest single model:

| Stage | Model | VRAM | When |
|-------|-------|------|------|
| Masking | SAM ViT-B | ~1 GB | While clicking in Hint Generator |
| Hint Generation | MatAnyone | ~4-6 GB | During "Generate Hints" or pipeline stage 1 |
| Keying | CorridorKey | ~22 GB | During "Process Batch" or pipeline stage 2 |

SAM unloads before MatAnyone loads. MatAnyone unloads before CorridorKey loads.

**24 GB GPU (RTX 3090/4090/5090):** Full pipeline works at default settings.
**12 GB GPU:** May work at lower MatAnyone resolution (720) and reduced CorridorKey resolution — contributions welcome to improve this.

---

## Compositing Your Output

### Nuke
```
Read node → set to *_processed.exr sequence
Premult is already baked in — comp directly over your background with Merge (over)
```

### After Effects
```
Import *_processed.exr sequence
Interpret Footage → set to Straight alpha (AE will handle the premult)
Or import *_alpha.exr separately and use as a track matte
```

### DaVinci Resolve / Fusion
```
MediaIn → *_processed.exr sequence
Already premultiplied — Merge over your background plate
```

---

## Troubleshooting

**SAM isn't selecting my subject well**
- Use fewer, more precise clicks. One click on center mass often beats 10 scattered clicks.
- Use Negative clicks to carve out background areas SAM grabs incorrectly.
- Try Add Mask for disconnected parts (person + separate prop).

**MatAnyone output is drifting / losing the subject**
- Add more keyframes at the trouble spots. This is exactly what multi-keyframe is for.
- Increase Warmup Frames (try 20-30).
- Check that your SAM mask cleanly covers the subject at each keyframe.

**CorridorKey edges look soft**
- Try Refiner Scale > 1.0 (e.g., 1.5) for sharper edges.
- Check that MatAnyone hints aren't too eroded — reduce Mask Erosion.

**VRAM out of memory**
- Lower MatAnyone Processing Resolution to 720.
- Close other GPU-heavy apps (browsers with hardware acceleration, games, etc.).
- The models load sequentially — you should never need VRAM for more than one at a time.

**EXR files won't open**
- Make sure your comp software supports OpenEXR. Most do natively.
- In AE, you may need to set footage interpretation to 32-bit float.
