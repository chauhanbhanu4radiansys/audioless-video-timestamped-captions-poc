# Detailed Instructions: Copying local.py and Creating Test Files

## Overview
This guide assumes you have already completed the main restructuring steps:
- ✅ Created `main.py`
- ✅ Modified `gpu/handler.py`
- ✅ Copied and modified all `src/` files
- ✅ Updated `gpu/Dockerfile` and `requirements.txt`

**Now we focus ONLY on:**
1. Copying and modifying `gpu/local.py` for testing
2. Creating test scripts for Docker build and testing
3. Ensuring everything adheres to the original aim

---

## Part 1: Copy and Modify `gpu/local.py`

### Original Aim Reminder
- Read video (local or AWS)
- Extract key frames using adaptive detector
- Use BLIP to get description of each extracted frame
- Return list of dicts: `{id, description, start_time, end_time, start_frame, end_frame}`

### Step-by-Step Instructions

#### 1.1: Read the Original File
**Source:** `gpu/local.py` from original repo

**Identify what to keep:**
- Lines 1-20: Environment setup (keep)
- Lines 22-49: GPU verification (keep)
- Line 51: Import handler (keep)
- Lines 53-318: Task handling (modify - keep only embedding)

#### 1.2: Create Modified Version

**File:** `gpu/local.py`

**Copy this exact code:**

```python
import os
import sys
from dotenv import load_dotenv

# Set runtime environment
os.environ['RUNTIME_ENVIRONMENT'] = 'local'

# Get project root directory (parent of gpu directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from project root
env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()  # Fallback to current directory

# Add parent directory to Python path
sys.path.insert(0, PROJECT_ROOT)

# GPU Verification at startup
try:
    import torch
    print("\n" + "="*60)
    print("GPU VERIFICATION")
    print("="*60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"✅ GPU Device: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.2f} GB")
        print(f"✅ PyTorch CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CRITICAL: CUDA NOT AVAILABLE - Running on CPU!")
        print("   This will be 10-100x slower than GPU")
        print("   Expected processing time: 10-30+ minutes")
        print("")
        print("   To enable GPU:")
        print("   1. Check Docker GPU access:")
        print("      docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi")
        print("   2. Install NVIDIA Container Toolkit if needed")
        print("   3. Restart Docker: sudo systemctl restart docker")
    print("="*60 + "\n")
except ImportError:
    print("⚠️  Warning: PyTorch not available, cannot verify GPU")
except Exception as e:
    print(f"⚠️  Warning: Could not verify GPU: {e}")

from handler import handler

if __name__ == "__main__":
    print("=" * 60)
    print("VIDEO FRAME EXTRACTION - LOCAL TESTING")
    print("=" * 60)
    print()
    
    # Get video URL/path from environment variables or command line
    video_url = os.getenv('VIDEO_URL', '')
    video_name = os.getenv('VIDEO_NAME', 'test_video')
    video_id = os.getenv('VIDEO_ID', None)
    job_id = os.getenv('JOB_ID', 'test-job-123')
    tenant = os.getenv('TENANT', 'test-tenant')
    sns_topic_arn = os.getenv('SNS_TOPIC_ARN', '')
    
    # If URL not in environment, check command line arguments
    if not video_url and len(sys.argv) > 1:
        video_url = sys.argv[1]
    if len(sys.argv) > 2:
        video_name = sys.argv[2]
    if len(sys.argv) > 3:
        video_id = sys.argv[3]
    
    # If still no URL, use placeholder with instructions
    if not video_url:
        print("⚠️  No video URL/path provided!")
        print()
        print("Usage options:")
        print("  1. Set environment variables:")
        print("     export VIDEO_URL='https://s3.amazonaws.com/bucket/video.mp4'")
        print("     export VIDEO_URL='/path/to/local/video.mp4'  # or local file path")
        print("     export VIDEO_NAME='my_video'")
        print("     python local.py")
        print()
        print("  2. Pass as command line arguments:")
        print("     python local.py <video_url_or_path> [video_name] [video_id]")
        print("     Examples:")
        print("       python local.py https://s3.amazonaws.com/bucket/video.mp4")
        print("       python local.py /path/to/video.mp4 my_video")
        print("       python local.py file:///path/to/video.mp4 my_video video-123")
        print()
        print("  3. Update the placeholder below and run:")
        print()
        
        # ========== PLACEHOLDER: Update with your video URL or local path ==========
        video_url = video_url or "PLACEHOLDER_VIDEO_URL_OR_PATH"
    
    # Create job payload - MATCHES handler.py input format
    # NOTE: No transcriptPathURL - not part of original aim
    video_job = {
        "input": {
            "task_type": "embedding",
            "videoPathURL": video_url,
            "video_name": video_name,
            "video_id": video_id,
            "id": job_id,
            "tenant": tenant,
            "snsTopicArn": sns_topic_arn if sns_topic_arn else None
        }
    }

    print("Job Configuration:")
    print(f"  Task Type: embedding")
    print(f"  Video URL/Path: {video_url[:80]}..." if len(video_url) > 80 else f"  Video URL/Path: {video_url}")
    print(f"  Video Name: {video_name}")
    print(f"  Video ID: {video_id or 'auto-generated'}")
    print(f"  Job ID: {job_id}")
    print(f"  Tenant: {tenant}")
    if sns_topic_arn:
        print(f"  SNS Topic: {sns_topic_arn}")
    print()
    print("=" * 60)
    print("Starting Processing...")
    print("=" * 60)
    print()
    
    try:
        result = handler(video_job)
        print()
        print("=" * 60)
        print("Result:")
        print("=" * 60)
        
        # Pretty print JSON result
        import json
        if isinstance(result, dict) and 'body' in result:
            body = json.loads(result['body'])
            print(json.dumps(body, indent=2))
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print()
        print("=" * 60)
        print("Error occurred:")
        print("=" * 60)
        print(f"❌ {e}")
        import traceback
        try:
            traceback.print_exc()
        except (OSError, ValueError) as exc:
            try:
                print(f"Traceback (stderr unavailable): {exc}", file=sys.stdout)
            except:
                pass
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("USAGE")
    print("=" * 60)
    print()
    print("The handler() function processes videos and extracts frames with BLIP descriptions.")
    print()
    print("Required fields in input:")
    print("  - videoPathURL: S3 pre-signed URL or local file path")
    print("  - video_name: Name of the video")
    print()
    print("Optional fields in input:")
    print("  - video_id: Unique identifier (auto-generated if None)")
    print("  - id: Job ID for notifications")
    print("  - tenant: Tenant ID for notifications")
    print("  - snsTopicArn: SNS topic ARN for notifications")
    print()
    print("=" * 60)
```

**Key Points:**
- ✅ Only handles video processing (no search/analyse)
- ✅ No transcript URL required
- ✅ Calls `handler()` with correct payload format
- ✅ Matches simplified `handler.py` input structure

---

## Part 2: Update Dockerfile

### Step 2.1: Add local.py to Dockerfile

**File:** `gpu/Dockerfile`

**Find this section (near the end):**
```dockerfile
# Copy application files
COPY ../src /src
COPY ./gpu/handler.py /handler.py
```

**Replace with:**
```dockerfile
# Copy application files
COPY ./src /src
COPY ./gpu/handler.py /handler.py
COPY ./gpu/local.py /local.py
COPY ./main.py /main.py
```

**Keep CMD as:**
```dockerfile
CMD ["python3.11", "-u", "/handler.py"]
```

**Note:** This is for production. When testing, override CMD to use `local.py`.

---

## Part 3: Create Test Scripts

### Option A: Create `test_docker.sh` (Bash)

**File:** `test_docker.sh` (root directory)

**Copy this exact code:**

```bash
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="video-frame-extractor"
CONTAINER_NAME="video-test-container"
TEST_VIDEO_PATH="${1:-}"  # From argument

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Docker Build and Test Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if docker info | grep -q "nvidia"; then
    echo -e "${GREEN}✅ NVIDIA Docker runtime detected${NC}"
    GPU_FLAG="--gpus all"
else
    echo -e "${YELLOW}⚠️  NVIDIA Docker runtime not detected. Will run on CPU (slow).${NC}"
    GPU_FLAG=""
fi

# Step 1: Build Docker image
echo ""
echo -e "${GREEN}Step 1: Building Docker image...${NC}"
echo "----------------------------------------"

if docker build -t ${IMAGE_NAME} -f gpu/Dockerfile .; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Step 2: Verify files exist in image
echo ""
echo -e "${GREEN}Step 2: Verifying files in image...${NC}"
echo "----------------------------------------"

FILES_TO_CHECK=("/handler.py" "/local.py" "/main.py")
ALL_EXIST=true

for file_path in "${FILES_TO_CHECK[@]}"; do
    if docker run --rm ${IMAGE_NAME} test -f ${file_path} > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${file_path} exists${NC}"
    else
        echo -e "${RED}❌ ${file_path} not found${NC}"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo -e "${RED}❌ Some required files are missing${NC}"
    exit 1
fi

# Step 3: Run test with local.py (if video provided)
if [ -n "$TEST_VIDEO_PATH" ]; then
    echo ""
    echo -e "${GREEN}Step 3: Running test with local.py...${NC}"
    echo "----------------------------------------"
    
    # Determine if video is URL or local file
    if [[ "$TEST_VIDEO_PATH" =~ ^https?:// ]]; then
        # It's a URL - no need to mount
        echo "Using video URL: $TEST_VIDEO_PATH"
        docker run --rm ${GPU_FLAG} \
            -e VIDEO_URL="$TEST_VIDEO_PATH" \
            -e VIDEO_NAME="test_video" \
            --name ${CONTAINER_NAME} \
            ${IMAGE_NAME} \
            python3.11 -u /local.py
    else
        # It's a local file - need to mount it
        if [ ! -f "$TEST_VIDEO_PATH" ]; then
            echo -e "${RED}❌ Video file not found: $TEST_VIDEO_PATH${NC}"
            exit 1
        fi
        
        VIDEO_DIR=$(dirname "$(realpath "$TEST_VIDEO_PATH")")
        VIDEO_FILE=$(basename "$TEST_VIDEO_PATH")
        echo "Mounting local video: $TEST_VIDEO_PATH"
        
        docker run --rm ${GPU_FLAG} \
            -v "${VIDEO_DIR}:/videos" \
            -e VIDEO_URL="/videos/${VIDEO_FILE}" \
            -e VIDEO_NAME="test_video" \
            --name ${CONTAINER_NAME} \
            ${IMAGE_NAME} \
            python3.11 -u /local.py
    fi
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Test completed successfully${NC}"
    else
        echo ""
        echo -e "${RED}❌ Test failed with exit code: $TEST_EXIT_CODE${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${YELLOW}⚠️  No video provided. Skipping video test.${NC}"
    echo "To test with video: ./test_docker.sh /path/to/video.mp4"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ All checks passed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  - Docker image: ${IMAGE_NAME}"
echo "  - Production handler: handler.py (used by RunPod/serverless)"
echo "  - Test handler: local.py (used for local testing)"
echo ""
echo "To test manually:"
echo "  docker run --rm ${GPU_FLAG} ${IMAGE_NAME} python3.11 -u /local.py <video_url>"
```

**Make executable:**
```bash
chmod +x test_docker.sh
```

---

### Option B: Create `test_docker.py` (Python)

**File:** `test_docker.py` (root directory)

**Copy this exact code:**

```python
#!/usr/bin/env python3
"""
Docker build and test script for video frame extraction.
Tests using local.py while production uses handler.py.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_step(step_num, message):
    print(f"\n{Colors.GREEN}{'='*60}{Colors.NC}")
    print(f"{Colors.GREEN}Step {step_num}: {message}{Colors.NC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}\n")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.NC}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.NC}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")

def check_docker():
    """Check if Docker is running"""
    try:
        result = subprocess.run(
            ['docker', 'info'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_nvidia_runtime():
    """Check if NVIDIA Docker runtime is available"""
    try:
        result = subprocess.run(
            ['docker', 'info'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return 'nvidia' in result.stdout.lower()
    except:
        return False

def build_docker_image(image_name, dockerfile_path):
    """Build Docker image"""
    print(f"Building Docker image: {image_name}")
    print(f"Using Dockerfile: {dockerfile_path}")
    
    cmd = ['docker', 'build', '-t', image_name, '-f', dockerfile_path, '.']
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode == 0

def verify_files_in_image(image_name):
    """Verify that handler.py and local.py exist in the image"""
    files_to_check = ['/handler.py', '/local.py', '/main.py']
    all_exist = True
    
    for file_path in files_to_check:
        result = subprocess.run(
            ['docker', 'run', '--rm', image_name, 'test', '-f', file_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_success(f"{file_path} exists")
        else:
            print_error(f"{file_path} not found")
            all_exist = False
    
    return all_exist

def run_local_test(image_name, video_path, gpu_flag):
    """Run local.py test inside Docker container"""
    print(f"Running test with video: {video_path}")
    
    is_url = video_path.startswith('http://') or video_path.startswith('https://')
    
    cmd = ['docker', 'run', '--rm']
    
    if gpu_flag:
        cmd.extend(['--gpus', 'all'])
    
    if is_url:
        cmd.extend([
            '-e', f'VIDEO_URL={video_path}',
            '-e', 'VIDEO_NAME=test_video',
            image_name,
            'python3.11', '-u', '/local.py'
        ])
    else:
        if not os.path.exists(video_path):
            print_error(f"Video file not found: {video_path}")
            return False
        
        video_dir = str(Path(video_path).parent.resolve())
        video_file = Path(video_path).name
        
        cmd.extend([
            '-v', f'{video_dir}:/videos',
            '-e', f'VIDEO_URL=/videos/{video_file}',
            '-e', 'VIDEO_NAME=test_video',
            image_name,
            'python3.11', '-u', '/local.py'
        ])
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode == 0

def main():
    parser = argparse.ArgumentParser(
        description='Build Docker image and test with local.py'
    )
    parser.add_argument(
        'video_path',
        nargs='?',
        default=None,
        help='Path to test video (local file or S3 URL)'
    )
    parser.add_argument(
        '--image-name',
        default='video-frame-extractor',
        help='Docker image name (default: video-frame-extractor)'
    )
    parser.add_argument(
        '--dockerfile',
        default='gpu/Dockerfile',
        help='Path to Dockerfile (default: gpu/Dockerfile)'
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip Docker build (use existing image)'
    )
    
    args = parser.parse_args()
    
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}Docker Build and Test Script{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
    
    # Step 1: Check Docker
    print_step(1, "Checking Docker")
    if not check_docker():
        print_error("Docker is not running. Please start Docker first.")
        sys.exit(1)
    print_success("Docker is running")
    
    has_gpu = check_nvidia_runtime()
    if has_gpu:
        print_success("NVIDIA Docker runtime detected")
        gpu_flag = True
    else:
        print_warning("NVIDIA Docker runtime not detected. Will run on CPU (slow).")
        gpu_flag = False
    
    # Step 2: Build Docker image
    if not args.skip_build:
        print_step(2, "Building Docker Image")
        if not build_docker_image(args.image_name, args.dockerfile):
            print_error("Docker build failed")
            sys.exit(1)
        print_success("Docker image built successfully")
    else:
        print_step(2, "Skipping Build (using existing image)")
    
    # Step 3: Verify files in image
    print_step(3, "Verifying Files in Image")
    if not verify_files_in_image(args.image_name):
        print_error("Required files missing in image")
        sys.exit(1)
    print_success("All required files present")
    
    # Step 4: Run test
    if args.video_path:
        print_step(4, "Running Test with local.py")
        if run_local_test(args.image_name, args.video_path, gpu_flag):
            print_success("Test completed successfully")
        else:
            print_error("Test failed")
            sys.exit(1)
    else:
        print_step(4, "Skipping Test (no video provided)")
        print_warning("No video path provided.")
        print("\nTo test manually:")
        print(f"  docker run --rm {'--gpus all' if gpu_flag else ''} {args.image_name} python3.11 -u /local.py <video_url>")
    
    # Summary
    print()
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
    print(f"{Colors.GREEN}✅ All checks passed!{Colors.NC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
    print()
    print("Summary:")
    print(f"  - Docker image: {args.image_name}")
    print(f"  - Production handler: handler.py (used by RunPod/serverless)")
    print(f"  - Test handler: local.py (used for local testing)")

if __name__ == '__main__':
    main()
```

**Make executable:**
```bash
chmod +x test_docker.py
```

---

## Part 4: Verification Checklist

### Verify local.py
- [ ] File exists at `gpu/local.py`
- [ ] Only handles video processing (no search/analyse)
- [ ] No transcript URL required
- [ ] Calls `handler()` with correct payload
- [ ] Payload matches `handler.py` input format

### Verify Dockerfile
- [ ] Copies `handler.py` to `/handler.py`
- [ ] Copies `local.py` to `/local.py`
- [ ] Copies `main.py` to `/main.py`
- [ ] CMD uses `handler.py` (production)

### Verify Test Scripts
- [ ] `test_docker.sh` exists and is executable
- [ ] `test_docker.py` exists and is executable (optional)
- [ ] Both scripts build Docker image
- [ ] Both scripts verify files exist
- [ ] Both scripts can test with video

### Verify Output Format
- [ ] Output matches original aim:
  ```python
  {
      "id": "uuid",
      "description": "SUV,",
      "start_time": 10.44,
      "end_time": 10.92,
      "start_frame": 12,
      "end_frame": 34
  }
  ```
- [ ] No extra fields added
- [ ] All required fields present

---

## Usage Examples

### Build and Test
```bash
# Using bash script
./test_docker.sh /path/to/video.mp4

# Using Python script
python3 test_docker.py /path/to/video.mp4

# With S3 URL
./test_docker.sh https://s3.amazonaws.com/bucket/video.mp4
```

### Manual Testing
```bash
# Build image
docker build -t video-extractor -f gpu/Dockerfile .

# Test with local.py
docker run --rm --gpus all \
  -v /path/to/videos:/videos \
  -e VIDEO_URL="/videos/test.mp4" \
  video-extractor \
  python3.11 -u /local.py

# Production mode (uses handler.py automatically)
docker run --rm --gpus all \
  -e VIDEO_URL="https://s3.amazonaws.com/bucket/video.mp4" \
  video-extractor
```

---

## Important Reminders

✅ **Stick to Original Aim:**
- Video input only (no transcript)
- Frame extraction (adaptive detector)
- BLIP descriptions
- Output in specified format

❌ **Do NOT Add:**
- Search functionality
- Analysis functionality
- Transcript processing
- Any features beyond the original aim

✅ **Two Entry Points:**
- `handler.py` → Production
- `local.py` → Testing

Both should produce the same output format and functionality.
