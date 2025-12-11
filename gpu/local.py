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

