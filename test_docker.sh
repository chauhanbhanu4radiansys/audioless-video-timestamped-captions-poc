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
