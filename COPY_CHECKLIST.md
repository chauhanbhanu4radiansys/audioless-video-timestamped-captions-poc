# Quick Copy Checklist

## Files to Copy AS-IS (No Changes)

| Source | Destination | Notes |
|-------|-------------|-------|
| `src/notify.py` | `src/notify.py` | SNS notifications - works as-is |
| `src/embedding/download.py` | `src/download.py` | Handles AWS S3 URLs and local files |

---

## Files to Copy and MODIFY

### 1. `src/embedding/processing.py` → `src/processing.py`

**KEEP:**
- ✅ `detect_scenes_with_adaptive_detector()` (lines 134-167)
- ✅ `get_video_duration()` (lines 170-201) - optional but useful
- ✅ `resize_frame_optimized()` (lines 238-245)

**REMOVE:**
- ❌ `extract_audio_from_video()` - not needed
- ❌ `find_segment_by_scene_end()` - not needed
- ❌ `find_segments_in_scene_range()` - not needed
- ❌ `split_video_into_chunks()` - not needed
- ❌ `prepare_scene_data()` - replace with simpler version

**ADD:**
- ➕ `extract_frames_from_scenes()` - NEW function (see guide)

**IMPORTS TO KEEP:**
```python
import os, sys, subprocess, cv2, numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from contextlib import redirect_stderr
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector
```

---

### 2. `src/embedding/models.py` → `src/models.py`

**KEEP:**
- ✅ `load_captioning_model()` (lines 231-253) - BLIP loading

**REMOVE:**
- ❌ `load_imagebind_model()` - not needed
- ❌ `load_whisper_model()` - not needed
- ❌ All ImageBind imports and logging code (lines 1-101)

**SIMPLIFIED IMPORTS:**
```python
import os, torch
from typing import Tuple, Optional
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None
```

---

### 3. `src/embedding/embeddings.py` → `src/embeddings.py`

**KEEP:**
- ✅ `generate_scene_descriptions()` (lines 92-111) - BLIP captioning

**REMOVE:**
- ❌ `get_batch_embeddings()` - not needed (no ImageBind)
- ❌ All ImageBind imports

**SIMPLIFIED IMPORTS:**
```python
from typing import List
from PIL import Image
```

---

### 4. `gpu/handler.py` → `gpu/handler.py`

**KEEP:**
- ✅ `download()` function (lines 193-198)
- ✅ `cleanup_temp_files()` (lines 200-207)
- ✅ `notify` import (line 19)

**MODIFY:**
- 🔧 `handle_embedding()` - Replace entire function (see guide for new version)
- 🔧 `handler()` - Simplify to only call `handle_embedding()`

**REMOVE:**
- ❌ `upload()` - not needed
- ❌ `handle_search()` - not needed
- ❌ `handle_analyse()` - not needed
- ❌ `download_with_range()` - optional, keep if needed
- ❌ All Pinecone code

---

### 5. `gpu/Dockerfile` → `gpu/Dockerfile`

**KEEP:**
- ✅ Base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- ✅ FFmpeg installation
- ✅ **BLIP model download** (lines 116-163) - CRITICAL!
- ✅ PyTorch installation

**REMOVE:**
- ❌ ImageBind installation (lines 63-66, 100-114)
- ❌ ImageBind model download
- ❌ Pinecone-related code

**SIMPLIFY:**
- 🔧 Reduce dependencies to only what's needed
- 🔧 Remove ImageBind cache directories

---

### 6. `gpu/requirements.txt` → `gpu/requirements.txt`

**KEEP:**
- ✅ torch, torchvision, torchaudio
- ✅ transformers, huggingface_hub (for BLIP)
- ✅ opencv-python, scenedetect (for video processing)
- ✅ pillow (for images)
- ✅ requests (for downloads)
- ✅ boto3 (for SNS)
- ✅ python-dotenv

**REMOVE:**
- ❌ ImageBind deps: timm, ftfy, regex, einops, iopath
- ❌ pinecone
- ❌ pydub, torchcodec
- ❌ openai-whisper

---

## Files to CREATE NEW

### 1. `main.py` (Root directory)

**Purpose:** Simplified pipeline that:
1. Gets FPS from video using cv2
2. Detects scenes with adaptive detector
3. Extracts frames from scenes
4. Loads BLIP model
5. Generates descriptions
6. Returns list of dicts

**See full code in RESTRUCTURE_GUIDE.md**

---

## Final Directory Structure

```
your-project/
├── src/
│   ├── __init__.py
│   ├── notify.py              # ✅ Copy as-is
│   ├── download.py            # ✅ Copy as-is
│   ├── processing.py          # 🔧 Copy + modify
│   ├── models.py              # 🔧 Copy + modify
│   └── embeddings.py          # 🔧 Copy + modify
├── gpu/
│   ├── handler.py             # 🔧 Copy + modify
│   ├── Dockerfile             # 🔧 Copy + modify
│   └── requirements.txt       # 🔧 Copy + modify
└── main.py                    # ✨ Create new
```

---

## Key Functions Flow

```
handle_embedding() [gpu/handler.py]
    ↓
download_from_url() [src/download.py]  # Handles AWS/local
    ↓
process_video_frames() [main.py]
    ↓
detect_scenes_with_adaptive_detector() [src/processing.py]
    ↓
extract_frames_from_scenes() [src/processing.py]
    ↓
load_captioning_model() [src/models.py]  # BLIP
    ↓
generate_scene_descriptions() [src/embeddings.py]  # BLIP
    ↓
Return list of dicts with: id, description, start_time, end_time, start_frame, end_frame
```

---

## Testing Commands

```bash
# Local test
python main.py /path/to/video.mp4

# Docker build
docker build -t video-processor -f gpu/Dockerfile .

# Docker run
docker run --rm --gpus all \
  -e VIDEO_URL="https://s3.amazonaws.com/bucket/video.mp4" \
  video-processor
```

---

## Critical Points

1. **BLIP Model Download**: Must be in Dockerfile (lines 116-163 from original) - pre-downloads during build
2. **AWS Fetching**: `download_from_url()` automatically handles both S3 URLs and local paths
3. **Notify Code**: Copy `src/notify.py` exactly as-is - no changes needed
4. **FPS**: Get from `cv2.VideoCapture.get(cv2.CAP_PROP_FPS)` - used to calculate frame numbers
5. **Output Format**: Exactly matches your specification with id, description, start_time, end_time, start_frame, end_frame
