# Local Testing Checklist - Copying local.py and Creating Test Files

## Prerequisites
✅ **ASSUMED COMPLETED:**
- `main.py` created and working
- `gpu/handler.py` copied and modified
- `src/` directory files copied and modified:
  - `src/notify.py` (as-is)
  - `src/download.py` (as-is)
  - `src/processing.py` (modified)
  - `src/models.py` (modified - BLIP only)
  - `src/embeddings.py` (modified - BLIP only)
- `gpu/Dockerfile` updated (BLIP model download)
- `gpu/requirements.txt` simplified

---

## Step 1: Copy and Modify `gpu/local.py`

### Source File
`gpu/local.py` (from original repo)

### What to Keep (Strictly for Video Processing Only)
- ✅ GPU verification code (lines 22-49)
- ✅ Environment variable loading (lines 1-20)
- ✅ Basic error handling structure
- ✅ Import of `handler` function

### What to Remove (Not Part of Original Aim)
- ❌ Search task handling (lines 70-152)
- ❌ Analyse task handling (lines 154-224)
- ❌ All references to transcript URLs
- ❌ All references to Pinecone/search/analysis

### What to Modify
- 🔧 Simplify to ONLY handle video processing (embedding task)
- 🔧 Remove `transcriptPathURL` requirement
- 🔧 Update job payload to match simplified `handler.py`
- 🔧 Update usage instructions

### Key Requirements
1. **Input:** Only video URL/path (no transcript)
2. **Output:** List of dicts with: id, description, start_time, end_time, start_frame, end_frame
3. **No extra features:** No search, no analysis, no embeddings upload

---

## Step 2: Update `gpu/Dockerfile` to Include local.py

### What to Add
```dockerfile
# Copy both handler.py (production) and local.py (testing)
COPY ./gpu/handler.py /handler.py
COPY ./gpu/local.py /local.py
COPY ./main.py /main.py
```

### CMD Configuration
```dockerfile
# Production CMD: Use handler.py (for RunPod/serverless)
CMD ["python3.11", "-u", "/handler.py"]
```

**Note:** When testing, override CMD to use `local.py`:
```bash
docker run ... image_name python3.11 -u /local.py
```

---

## Step 3: Create Test Scripts

### Option A: Bash Script (`test_docker.sh`)
**Purpose:** Build Docker image and test with `local.py`

**Requirements:**
- Build Docker image
- Check GPU availability
- Test with video (local file or S3 URL)
- Verify both `handler.py` and `local.py` exist
- **No extra features** - just build and test

### Option B: Python Script (`test_docker.py`)
**Purpose:** Same as bash script, but with better error handling

**Requirements:**
- Same as bash script
- Command-line argument support
- Real-time output streaming
- **No extra features** - just build and test

---

## Step 4: Verify Everything Works

### Checklist
- [ ] `local.py` only handles video processing (no search/analyse)
- [ ] `local.py` doesn't require transcript URL
- [ ] `local.py` calls `handler()` with correct payload format
- [ ] Dockerfile copies both `handler.py` and `local.py`
- [ ] Test script builds image successfully
- [ ] Test script runs `local.py` successfully
- [ ] Output matches original aim format:
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

---

## What NOT to Include

❌ **Do NOT add:**
- Search functionality
- Analysis functionality
- Transcript processing
- Pinecone uploads
- Any features beyond video → frames → BLIP descriptions

✅ **ONLY include:**
- Video input (local file or S3 URL)
- Frame extraction (adaptive detector)
- BLIP description generation
- Output in specified format

---

## Testing Workflow

1. **Build image:**
   ```bash
   docker build -t video-extractor -f gpu/Dockerfile .
   ```

2. **Test with local.py:**
   ```bash
   docker run --rm --gpus all \
     -e VIDEO_URL="/videos/test.mp4" \
     video-extractor \
     python3.11 -u /local.py
   ```

3. **Verify output format:**
   - Check JSON output matches specification
   - Verify all required fields present
   - Confirm no extra fields added

---

## Success Criteria

✅ **local.py works correctly if:**
- Accepts video URL/path as input
- Processes video through `handler()`
- Returns list of dicts in correct format
- No errors during execution

✅ **Test script works correctly if:**
- Builds Docker image successfully
- Runs `local.py` without errors
- Verifies both handlers exist in image
- Provides clear success/failure messages
