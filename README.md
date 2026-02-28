# CorridorKey GUI

A Gradio web interface for **CorridorKey** — the neural network green screen keyer by [Niko Pueringer](https://www.youtube.com/@NikoTech) / [Corridor Digital](https://www.corridordigital.com/). This GUI adds interactive masking via SAM + MatAnyone video matting to create a complete keying pipeline from raw green screen footage to production-ready EXR sequences.

## What This Does

**The problem:** CorridorKey needs a coarse alpha hint mask for each frame. Manually painting masks for every frame of a video is impractical.

**The solution:** This GUI chains three models together:
1. **SAM** (Segment Anything) — Click on your subject to create a mask on any keyframe
2. **MatAnyone** — Propagates that mask across the entire video with temporal consistency
3. **CorridorKey** — Uses those propagated masks as hints to produce broadcast-quality alpha mattes and despilled foreground

The result is a one-click pipeline from raw green screen video to keyed EXR sequences ready for compositing in Nuke, Fusion, or After Effects.

## Features

- **Multi-keyframe masking** — Set SAM masks at multiple points in the video. Each segment gets a clean anchor, so the solver recovers from occlusions and motion blur
- **Mask expansion** — Dilate masks to preserve fine detail like hair and motion blur
- **Interactive SAM** — Positive and negative click points, multi-mask support, real-time preview
- **Full Pipeline mode** — One button: SAM mask → MatAnyone propagation → CorridorKey keying → EXR output
- **Batch mode** — Bring your own pre-made masks from Nuke/AE and just run CorridorKey
- **Production color pipeline** — Linear EXR output with premultiplied alpha, proper sRGB/linear conversions throughout

## Requirements

- **GPU:** NVIDIA GPU with CUDA support
  - 24GB VRAM recommended (RTX 3090/4090) — runs all three models sequentially
  - 12GB VRAM may work at reduced resolution — contributions welcome to optimize this
- **Python:** 3.10 or 3.11
- **OS:** Windows 10/11 (tested), Linux (should work)

## Installation

### 1. Clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/CorridorKey-GUI.git
cd CorridorKey-GUI
```

### 2. Install dependencies
```bash
pip install -r requirements-corridorkey.txt
pip install -r requirements-gui.txt
```

### 3. Install MatAnyone
Clone [MatAnyone](https://github.com/pq-yang/MatAnyone) into this directory and install as editable:
```bash
git clone https://github.com/pq-yang/MatAnyone.git
pip install -e ./MatAnyone
```

> **Windows note:** You may need to remove `cchardet`, `PySide6`, and `pyqtdarktheme` from MatAnyone's `pyproject.toml` if they fail to build.

### 4. Download checkpoints

| Model | Size | Location |
|-------|------|----------|
| CorridorKey | ~400MB | `CorridorKeyModule/checkpoints/CorridorKey.pth` |
| SAM ViT-B | ~358MB | `checkpoints/sam_vit_b.pth` |
| MatAnyone | ~135MB | Auto-downloaded from HuggingFace on first run |

- **CorridorKey weights:** Obtain from [nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey) (not redistributed here)
- **SAM weights:** Download `sam_vit_b_01ec64.pth` from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) and rename to `sam_vit_b.pth`

### 5. Launch
```bash
python app.py
```
Opens at `http://localhost:7860`. A public share link is also generated.

## Usage

See **[WORKFLOW.md](WORKFLOW.md)** for the full step-by-step guide with settings reference, compositing tips, and troubleshooting.

### Quick Start (Full Pipeline)
1. Go to **Hint Generator** tab
2. Upload your green screen video
3. Click on your subject to create a SAM mask (green = include, red = exclude)
4. Switch to **Full Pipeline** tab and click Generate
5. Keyed EXR sequence appears in `output/`

### Multi-Keyframe Workflow
1. Load video in **Hint Generator**
2. Use the frame slider to navigate to key moments
3. Click SAM points and **Save Keyframe** at each
4. Generate — each segment processes independently with its own anchor mask

### Batch Mode
If you already have alpha hint masks (from Nuke, AE, etc.):
1. Go to **Batch / Sequence** tab
2. Upload video + mask (image or video)
3. Adjust CorridorKey settings and run

## Output

All output goes to `output/` with timestamped subdirectories:
- `*_alpha.exr` — Linear alpha matte
- `*_fg.exr` — sRGB foreground (straight/unpremultiplied)
- `*_processed.exr` — Linear premultiplied RGBA (ready for comp)
- `*_comp.png` — sRGB composite preview on checkerboard

## VRAM Management

Models load and unload sequentially to fit in VRAM:
1. **SAM** (~1GB) — loaded during masking
2. **MatAnyone** (~4-6GB) — loaded for hint generation, SAM unloaded first
3. **CorridorKey** (~22GB at 2048px) — loaded for keying, MatAnyone unloaded first

## Contributing

This project exists as a starting point for the community to build on. Key areas for contribution:

- **VRAM optimization** — Making this work on 12GB or even 8GB GPUs by using MatAnyone's lighter model variants or reducing CorridorKey's processing resolution
- **Performance** — Batch inference, TensorRT, or other acceleration
- **UI/UX** — Better keyframe management, timeline scrubbing, preview modes

## Credits & Licensing

**This project is non-commercial.** Both CorridorKey and MatAnyone carry non-commercial licenses. Any derivatives must remain free and open source.

### CorridorKey
Created by **Niko Pueringer** / **Corridor Digital**. Licensed under **CC BY-NC-SA 4.0**.
- Code: [github.com/nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey)
- Commercial use requires agreement with contact@corridordigital.com

### MatAnyone
By Peiqing Yang et al. (CVPR 2025). Licensed under **S-Lab License 1.0** (non-commercial use only).
- Code: [github.com/pq-yang/MatAnyone](https://github.com/pq-yang/MatAnyone)

### Segment Anything (SAM)
By Meta AI Research. Licensed under **Apache 2.0**.
- Code: [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

### This GUI
Licensed under **CC BY-NC-SA 4.0** (matching CorridorKey's share-alike requirement). See `LICENSE`.

---

Built by [FilmBarrie](https://github.com/FilmBarrie) with Claude Code.
