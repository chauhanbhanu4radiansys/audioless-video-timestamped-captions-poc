# Detailed Restructuring Guide for Video Frame Extraction with BLIP

## Overview
This guide explains how to copy and restructure code from `gpu/` and `src/` directories to create a simplified pipeline that:
1. Reads a video (from local file or AWS S3)
2. Extracts key frames using adaptive detector
3. Uses BLIP to generate descriptions for each extracted frame
4. Returns a list of dictionaries with frame metadata

## Directory Structure

Create the following structure:
```
your-project/
├── src/
│   ├── __init__.py
│   ├── notify.py              # Copy as-is from src/notify.py
│   ├── download.py            # Copy as-is from src/embedding/download.py
│   ├── processing.py          # Modified copy from src/embedding/processing.py
│   ├── models.py              # Modified copy from src/embedding/models.py
│   └── embeddings.py          # Modified copy from src/embedding/embeddings.py
├── gpu/
│   ├── handler.py             # Modified copy from gpu/handler.py
│   ├── Dockerfile             # Modified copy from gpu/Dockerfile
│   └── requirements.txt       # Copy as-is from gpu/requirements.txt
└── main.py                    # NEW: Simplified main processing function
```

---

## Step-by-Step Copy Instructions

### 1. Copy `src/notify.py` (AS-IS)
**Source:** `src/notify.py`  
**Destination:** `src/notify.py`  
**Action:** Copy entire file without modifications

**Why:** This handles SNS notifications for job completion/failure. It's already well-structured and doesn't need changes.

---

### 2. Copy `src/embedding/download.py` (AS-IS)
**Source:** `src/embedding/download.py`  
**Destination:** `src/download.py`  
**Action:** Copy entire file without modifications

**Why:** This handles downloading videos from both:
- Pre-signed S3 URLs (AWS bucket)
- Local file paths (`file://` or absolute paths)

The `download_from_url()` function automatically detects if the input is a URL or local path and handles both cases.

---

### 3. Copy `src/embedding/processing.py` (MODIFIED)
**Source:** `src/embedding/processing.py`  
**Destination:** `src/processing.py`  
**Action:** Copy but keep ONLY these functions:

**Keep these functions:**
- `detect_scenes_with_adaptive_detector()` - Lines 134-167
- `get_video_duration()` - Lines 170-201 (optional, but useful)
- `resize_frame_optimized()` - Lines 238-245

**Remove these functions (not needed):**
- `extract_audio_from_video()` - Not needed for your use case
- `find_segment_by_scene_end()` - Not needed (no transcript matching)
- `find_segments_in_scene_range()` - Not needed (no transcript matching)
- `split_video_into_chunks()` - Not needed (simplified pipeline)
- `prepare_scene_data()` - Will be replaced with simpler version

**Add this NEW function to extract frames from scenes:**

```python
def extract_frames_from_scenes(video_path: str, scene_list: List, fps: float) -> List[Dict]:
    """
    Extract middle frame from each detected scene.
    
    Args:
        video_path: Path to video file
        scene_list: List of scenes from detect_scenes_with_adaptive_detector
        fps: Video FPS (from cv2.VideoCapture)
    
    Returns:
        List of dicts with frame data:
        {
            'start_time': float,
            'end_time': float,
            'start_frame': int,
            'end_frame': int,
            'middle_frame': int,
            'frame_image': PIL.Image
        }
    """
    import cv2
    from PIL import Image
    
    frames_data = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return []
    
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        middle_frame = start_frame + (end_frame - start_frame) // 2
        
        # Extract middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame_np = cap.read()
        
        if ret:
            pil_image = resize_frame_optimized(frame_np)
            frames_data.append({
                'start_time': start_time,
                'end_time': end_time,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'middle_frame': middle_frame,
                'frame_image': pil_image
            })
    
    cap.release()
    return frames_data
```

**Required imports (keep only these):**
```python
import os
import sys
import subprocess
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from contextlib import redirect_stderr
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector
```

---

### 4. Copy `src/embedding/models.py` (MODIFIED)
**Source:** `src/embedding/models.py`  
**Destination:** `src/models.py`  
**Action:** Copy but keep ONLY BLIP loading function

**Keep this function:**
- `load_captioning_model()` - Lines 231-253

**Remove these functions (not needed):**
- `load_imagebind_model()` - Not needed (no embeddings)
- `load_whisper_model()` - Not needed (no audio transcription)

**Remove all ImageBind-related code:**
- Remove ImageBind imports
- Remove all logging suppression code (lines 1-101) - keep minimal warnings if needed
- Keep only transformers import for BLIP

**Simplified imports:**
```python
import os
import torch
from typing import Tuple, Optional

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    print("Warning: Transformers not available. Install with: pip install transformers")
```

---

### 5. Copy `src/embedding/embeddings.py` (MODIFIED)
**Source:** `src/embedding/embeddings.py`  
**Destination:** `src/embeddings.py`  
**Action:** Copy but keep ONLY BLIP description generation

**Keep this function:**
- `generate_scene_descriptions()` - Lines 92-111

**Remove these functions (not needed):**
- `get_batch_embeddings()` - Not needed (no ImageBind embeddings)

**Remove ImageBind imports:**
- Remove all ImageBind-related imports
- Keep only PIL Image import

**Simplified imports:**
```python
from typing import List
from PIL import Image
```

---

### 6. Copy `gpu/handler.py` (MODIFIED)
**Source:** `gpu/handler.py`  
**Destination:** `gpu/handler.py`  
**Action:** Copy but simplify to only handle video processing

**Keep these functions:**
- `download()` - Lines 193-198 (simple download function)
- `download_with_range()` - Lines 38-48 (if you need range downloads)
- `cleanup_temp_files()` - Lines 200-207
- `notify()` import - Line 19

**Remove these functions (not needed):**
- `upload()` - Not needed (no uploads)
- `handle_search()` - Not needed (no search functionality)
- `handle_analyse()` - Not needed (no analysis functionality)
- All Pinecone-related code

**Modify `handle_embedding()` to call your simplified pipeline:**

Replace the entire `handle_embedding()` function with:

```python
def handle_embedding(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle video processing requests.
    
    Args:
        input_data: Dictionary with video processing parameters
            - videoPathURL: S3 URL or local file path
            - video_name: Name of the video (optional)
            - video_id: Unique ID (optional, auto-generated)
            - tenant: Tenant ID (for notifications)
            - id: Job ID (for notifications)
            - snsTopicArn: SNS topic ARN (for notifications)
        
    Returns:
        JSON response with processing results
    """
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.download import download_from_url
    from src.notify import notify
    from main import process_video_frames  # Your new main function
    
    videoPathURL = input_data.get('videoPathURL')
    video_name = input_data.get('video_name', 'unknown_video')
    video_id = input_data.get('video_id')
    job_id = input_data.get('id')
    tenant = input_data.get('tenant')
    sns_topic_arn = input_data.get('snsTopicArn')
    
    # Validate required parameters
    if not videoPathURL:
        raise ValueError("Missing required parameter: videoPathURL")
    
    print("=" * 80)
    print("Starting Video Frame Extraction")
    print("=" * 80)
    print(f"Video URL/Path: {videoPathURL[:100] if len(videoPathURL) > 100 else videoPathURL}...")
    print(f"Video Name: {video_name}")
    print(f"Video ID: {video_id or 'auto-generated'}")
    print("=" * 80)
    
    # Download video if it's a URL (download_from_url handles local paths too)
    local_video_path = None
    try:
        local_video_path = download_from_url(videoPathURL)
        if not local_video_path or not os.path.exists(local_video_path):
            raise Exception(f"Failed to download video from: {videoPathURL[:100]}...")
        print(f"Video ready at: {local_video_path}")
    except Exception as e:
        error_msg = f"Download failed: {str(e)}"
        print(f"❌ {error_msg}")
        if tenant and job_id and sns_topic_arn:
            notify(tenant, job_id, False, sns_topic_arn, {'error': error_msg})
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': error_msg
            })
        }
    
    # Process video
    try:
        results = process_video_frames(local_video_path)
        
        print("=" * 80)
        print("Video Frame Extraction Completed Successfully")
        print("=" * 80)
        print(f"Extracted {len(results)} frames")
        
        # Send success notification
        if tenant and job_id and sns_topic_arn:
            notify(tenant, job_id, True, sns_topic_arn, {
                'video_name': video_name,
                'video_id': video_id,
                'frames_extracted': len(results)
            })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'message': 'Video frames extracted successfully',
                'video_name': video_name,
                'video_id': video_id,
                'frames_count': len(results),
                'results': results
            }, indent=2)
        }
        
    except Exception as e:
        error_message = str(e)
        print(f"❌ Error processing video: {error_message}")
        import traceback
        traceback.print_exc()
        
        # Send failure notification
        if tenant and job_id and sns_topic_arn:
            notify(tenant, job_id, False, sns_topic_arn, {'error': error_message})
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': error_message
            })
        }
    finally:
        # Cleanup downloaded file
        if local_video_path and os.path.exists(local_video_path):
            try:
                cleanup_temp_files(local_video_path)
            except Exception:
                pass
```

**Keep the handler wrapper:**
```python
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler"""
    try:
        input_data = job.get('input', {})
        return handle_embedding(input_data)
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': str(e)
            })
        }

# For RunPod serverless
if runpod is not None:
    runpod.serverless.start({"handler": handler})
```

---

### 7. Create NEW `main.py` (Simplified Pipeline)
**Location:** Root directory  
**Action:** Create new file with simplified processing logic

```python
"""
Main video processing pipeline: Extract frames and generate descriptions
"""
import os
import sys
import uuid
import cv2
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.processing import detect_scenes_with_adaptive_detector, extract_frames_from_scenes
from src.models import load_captioning_model
from src.embeddings import generate_scene_descriptions


def process_video_frames(video_path: str) -> List[Dict[str, Any]]:
    """
    Main processing function:
    1. Read video
    2. Extract key frames using adaptive detector
    3. Use BLIP to get description of each extracted frame
    4. Return list of dicts with frame metadata
    
    Args:
        video_path: Path to video file (local or downloaded from S3)
    
    Returns:
        List of dictionaries:
        [
            {
                "id": "uuid",
                "description": "SUV,",
                "start_time": 10.44,
                "end_time": 10.92,
                "start_frame": 12,
                "end_frame": 34
            },
            ...
        ]
    """
    # Step 1: Get video FPS using ffmpeg/cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    print(f"Video FPS: {fps}")
    
    # Step 2: Detect scenes using adaptive detector
    print("Detecting scenes with adaptive detector...")
    scene_list = detect_scenes_with_adaptive_detector(video_path)
    
    if not scene_list:
        print("No scenes detected in video")
        return []
    
    print(f"Detected {len(scene_list)} scenes")
    
    # Step 3: Extract frames from scenes
    print("Extracting frames from scenes...")
    frames_data = extract_frames_from_scenes(video_path, scene_list, fps)
    
    if not frames_data:
        print("No frames extracted")
        return []
    
    print(f"Extracted {len(frames_data)} frames")
    
    # Step 4: Load BLIP model
    print("Loading BLIP model...")
    caption_processor, caption_model = load_captioning_model()
    
    if not caption_processor or not caption_model:
        raise Exception("Failed to load BLIP model")
    
    device = "cuda:0" if next(caption_model.parameters()).device.type == "cuda" else "cpu"
    print(f"BLIP model loaded on {device}")
    
    # Step 5: Generate descriptions for all frames (batch processing)
    print("Generating descriptions with BLIP...")
    pil_images = [frame_data['frame_image'] for frame_data in frames_data]
    descriptions = generate_scene_descriptions(pil_images, caption_processor, caption_model, device)
    
    # Step 6: Build result list
    results = []
    for i, frame_data in enumerate(frames_data):
        result = {
            "id": str(uuid.uuid4()),
            "description": descriptions[i] if i < len(descriptions) else "",
            "start_time": frame_data['start_time'],
            "end_time": frame_data['end_time'],
            "start_frame": frame_data['start_frame'],
            "end_frame": frame_data['end_frame']
        }
        results.append(result)
    
    print(f"Generated {len(results)} frame descriptions")
    return results


if __name__ == "__main__":
    # For local testing
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    results = process_video_frames(video_path)
    
    import json
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(json.dumps(results, indent=2))
```

---

### 8. Copy `gpu/Dockerfile` (MODIFIED)
**Source:** `gpu/Dockerfile`  
**Destination:** `gpu/Dockerfile`  
**Action:** Copy but simplify - remove ImageBind, keep BLIP

**Key Changes:**

1. **Remove ImageBind installation** (lines 63-66, 100-114)
2. **Keep BLIP model download** (lines 116-163) - This is what you need!
3. **Remove Pinecone-related code** (not needed)
4. **Simplify requirements** - Only install what's needed for BLIP and video processing

**Simplified Dockerfile structure:**

```dockerfile
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install Python and system dependencies
RUN python3.11 -m pip install --upgrade pip && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy requirements and install Python dependencies
COPY ./gpu/requirements.txt /requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /requirements.txt

# Create model cache directories
RUN mkdir -p /model_cache/huggingface && \
    mkdir -p /model_cache/transformers

# Set cache environment variables
ENV HF_HOME=/model_cache/huggingface
ENV TRANSFORMERS_CACHE=/model_cache/transformers
ENV HF_HUB_CACHE=/model_cache/huggingface

# Pre-download BLIP model during build (CRITICAL - this is what you need!)
RUN echo "Pre-downloading BLIP captioning model..." && \
    python3.11 << 'PYEOF'
import os
os.environ['HF_HOME'] = '/model_cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/model_cache/transformers'
os.environ['HF_HUB_CACHE'] = '/model_cache/huggingface'

from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import snapshot_download

print('Downloading BLIP model files...')
snapshot_download('Salesforce/blip-image-captioning-large',
                  cache_dir='/model_cache/huggingface',
                  local_files_only=False)

print('Loading BLIP processor...')
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large',
                                          cache_dir='/model_cache/huggingface')

print('Loading BLIP model...')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large',
                                                      cache_dir='/model_cache/huggingface')

print('✓ BLIP model downloaded and cached')
PYEOF

# Copy application files
COPY ./src /src
COPY ./gpu/handler.py /handler.py
COPY ./main.py /main.py

# Set environment variables
ENV PYTHONPATH=/
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python3.11", "-u", "/handler.py"]
```

---

### 9. Copy `gpu/requirements.txt` (MODIFIED)
**Source:** `gpu/requirements.txt`  
**Destination:** `gpu/requirements.txt`  
**Action:** Copy but remove ImageBind dependencies

**Keep these:**
- PyTorch (torch, torchvision, torchaudio)
- Transformers (for BLIP)
- opencv-python (for video processing)
- scenedetect (for adaptive detector)
- pillow (for image processing)
- requests (for downloading)
- boto3 (for AWS SNS notifications)
- python-dotenv (for environment variables)

**Remove these (not needed):**
- ImageBind dependencies (timm, ftfy, regex, einops, iopath)
- pinecone (no vector database)
- pydub (no audio processing)
- torchcodec (no audio processing)
- openai-whisper (no transcription)

**Simplified requirements.txt:**
```txt
# PyTorch
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Transformers for BLIP
transformers>=4.35.2
huggingface_hub>=0.19.4

# Video processing
opencv-python>=4.8.0
scenedetect[opencv]>=0.6.2

# Image processing
pillow>=9.0.0

# HTTP requests
requests>=2.31.0

# AWS SDK
boto3>=1.28.0

# Utilities
python-dotenv>=1.0.0
```

---

## Summary of Changes

### What to Keep (Copy As-Is):
1. ✅ `src/notify.py` - SNS notifications
2. ✅ `src/embedding/download.py` → `src/download.py` - AWS/local file handling

### What to Modify:
1. 🔧 `src/embedding/processing.py` → `src/processing.py`
   - Keep: `detect_scenes_with_adaptive_detector()`, `get_video_duration()`, `resize_frame_optimized()`
   - Add: `extract_frames_from_scenes()` (new function)
   - Remove: Audio/transcript related functions

2. 🔧 `src/embedding/models.py` → `src/models.py`
   - Keep: `load_captioning_model()` (BLIP)
   - Remove: ImageBind and Whisper loading

3. 🔧 `src/embedding/embeddings.py` → `src/embeddings.py`
   - Keep: `generate_scene_descriptions()` (BLIP)
   - Remove: `get_batch_embeddings()` (ImageBind)

4. 🔧 `gpu/handler.py`
   - Keep: `download()`, `cleanup_temp_files()`, `notify()` import
   - Simplify: `handle_embedding()` to call new `main.py`
   - Remove: Search, analysis, Pinecone code

5. 🔧 `gpu/Dockerfile`
   - Keep: BLIP model download (lines 116-163)
   - Remove: ImageBind download
   - Simplify: Dependencies

### What to Create New:
1. ✨ `main.py` - Simplified pipeline that:
   - Gets FPS from video
   - Detects scenes
   - Extracts frames
   - Generates BLIP descriptions
   - Returns list of dicts

---

## Testing the Restructured Code

### Local Testing:
```bash
# Test with local video
python main.py /path/to/video.mp4

# Test with S3 URL
python main.py https://s3.amazonaws.com/bucket/video.mp4
```

### Docker Testing:
```bash
# Build image
docker build -t video-processor -f gpu/Dockerfile .

# Run container
docker run --rm --gpus all \
  -e VIDEO_URL="https://s3.amazonaws.com/bucket/video.mp4" \
  video-processor
```

---

## Key Points

1. **AWS Fetching**: The `download_from_url()` function in `src/download.py` automatically handles both:
   - Pre-signed S3 URLs (`https://...`)
   - Local file paths (`/path/to/file` or `file:///path/to/file`)

2. **Notify Code**: `src/notify.py` sends SNS notifications - copy as-is, no changes needed

3. **BLIP Model Download**: The Dockerfile pre-downloads BLIP model during build (lines 116-163 in original), so it's available immediately at runtime

4. **Frame Extraction**: Uses `detect_scenes_with_adaptive_detector()` to find scene boundaries, then extracts middle frame from each scene

5. **FPS Calculation**: Uses `cv2.VideoCapture` to get FPS, which is used to calculate frame numbers from timestamps

6. **Output Format**: Returns exactly the format you specified:
   ```python
   {
       "id": uuid,
       "description": "SUV,",
       "start_time": 10.44,
       "end_time": 10.92,
       "start_frame": 12,
       "end_frame": 34
   }
   ```

---

## Next Steps

1. Create the directory structure
2. Copy files as specified above
3. Make the modifications listed
4. Test locally with a sample video
5. Build Docker image and test
6. Deploy to your environment
