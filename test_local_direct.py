#!/usr/bin/env python3
"""
Local test script to run video processing directly without Docker.
This script runs the pipeline on your local machine.
"""
import os
import sys
import json
import time
from pathlib import Path

# Set runtime environment
os.environ['RUNTIME_ENVIRONMENT'] = 'local'

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Optionally load environment variables (only needed for AWS SNS notifications)
try:
    from dotenv import load_dotenv
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()  # Fallback to current directory
except ImportError:
    # dotenv is optional for local testing without notifications
    pass

# Add project root to Python path
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
        print("ℹ️  Running on CPU")
        print("   This will be slower than GPU but will work")
        print("   CPU optimizations are enabled for better performance")
    print("="*60 + "\n")
except ImportError:
    print("⚠️  Warning: PyTorch not available, cannot verify GPU")
except Exception as e:
    print(f"⚠️  Warning: Could not verify GPU: {e}")

# Import main processing function
from main import process_video_frames


def main():
    """Main test function"""
    print("=" * 60)
    print("VIDEO FRAME EXTRACTION - LOCAL DIRECT TEST")
    print("=" * 60)
    print()
    
    # Get video path from command line arguments
    if len(sys.argv) < 2:
        print("⚠️  No video path provided!")
        print()
        print("Usage:")
        print("  python test_local_direct.py <video_path>")
        print()
        print("Examples:")
        print("  python test_local_direct.py /path/to/video.mp4")
        print("  python test_local_direct.py ~/Downloads/video.mp4")
        print("  python test_local_direct.py https://s3.amazonaws.com/bucket/video.mp4")
        print()
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Expand user path and resolve
    video_path = str(Path(video_path).expanduser().resolve())
    
    # Check if file exists (for local files)
    if not video_path.startswith('http://') and not video_path.startswith('https://'):
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            print()
            print("Please check:")
            print("  1. The file path is correct")
            print("  2. The file exists and is readable")
            sys.exit(1)
    
    print("Configuration:")
    print(f"  Video Path: {video_path}")
    print(f"  Project Root: {PROJECT_ROOT}")
    print()
    print("=" * 60)
    print("Starting Processing...")
    print("=" * 60)
    print()
    
    try:
        overall_start = time.time()
        # Process video
        results = process_video_frames(video_path)
        total_time = time.time() - overall_start
        
        print()
        print("=" * 60)
        print("Processing Completed Successfully!")
        print("=" * 60)
        print()
        print(f"Total frames processed: {len(results)}")
        print()
        
        # Save to file
        output_file = os.path.join(PROJECT_ROOT, "results.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print()
            print(f"✅ Results saved to: {output_file}")
        except Exception as save_error:
            print(f"⚠️  Could not save results to file: {save_error}")

        print(f"⏱️  Total processing time: {total_time:.2f} seconds")
        print(f"📁 Results file: {output_file}")
        print()
        print("=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⚠️  Processing interrupted by user")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Error occurred:")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        try:
            traceback.print_exc()
        except (OSError, ValueError) as exc:
            try:
                print(f"Traceback (stderr unavailable): {exc}", file=sys.stdout)
            except:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
