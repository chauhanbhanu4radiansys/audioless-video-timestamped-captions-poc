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
