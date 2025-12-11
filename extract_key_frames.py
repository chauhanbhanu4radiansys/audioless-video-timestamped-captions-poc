#!/usr/bin/env python3
"""
Extract key frames from video using adaptive scene detector.
Saves extracted frames as images in the 'images' directory.
"""
import os
import sys
import cv2
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.processing import detect_scenes_with_adaptive_detector, extract_frames_from_scenes


def save_frames_to_images(video_path: str, output_dir: str = "images") -> List[str]:
    """
    Extract key frames from video and save them as images.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save images (default: "images")
    
    Returns:
        List of saved image file paths
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("KEY FRAME EXTRACTION")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output directory: {output_path.absolute()}")
    print()
    
    # Get video FPS
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    print(f"Video FPS: {fps}")
    
    # Detect scenes using adaptive detector
    print("Detecting scenes with adaptive detector...")
    scene_list = detect_scenes_with_adaptive_detector(video_path)
    
    if not scene_list:
        print("No scenes detected in video")
        return []
    
    print(f"Detected {len(scene_list)} scenes")
    
    # Extract frames from scenes
    print("Extracting frames from scenes...")
    frames_data = extract_frames_from_scenes(video_path, scene_list, fps)
    
    if not frames_data:
        print("No frames extracted")
        return []
    
    print(f"Extracted {len(frames_data)} frames")
    print()
    
    # Get video filename without extension for naming
    video_name = Path(video_path).stem
    
    # Save frames as images
    saved_files = []
    print("Saving frames as images...")
    
    for i, frame_data in enumerate(frames_data):
        # Create filename with scene info
        start_time = frame_data['start_time']
        end_time = frame_data['end_time']
        frame_num = frame_data['middle_frame']
        
        # Format: video_name_scene_001_time_10.44-10.92_frame_123.jpg
        filename = f"{video_name}_scene_{i+1:04d}_time_{start_time:.2f}-{end_time:.2f}_frame_{frame_num}.jpg"
        filepath = output_path / filename
        
        # Save image
        frame_data['frame_image'].save(filepath, quality=95)
        saved_files.append(str(filepath))
        
        if (i + 1) % 10 == 0 or (i + 1) == len(frames_data):
            print(f"  Saved {i + 1}/{len(frames_data)} frames...")
    
    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total frames saved: {len(saved_files)}")
    print(f"Output directory: {output_path.absolute()}")
    print()
    
    return saved_files


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python extract_key_frames.py <video_path> [output_dir]")
        print()
        print("Examples:")
        print("  python extract_key_frames.py ~/Downloads/video.mp4")
        print("  python extract_key_frames.py ~/Downloads/video.mp4 my_frames")
        print()
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "images"
    
    # Expand user path
    video_path = str(Path(video_path).expanduser().resolve())
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        saved_files = save_frames_to_images(video_path, output_dir)
        
        if saved_files:
            print("✅ Successfully extracted and saved key frames!")
            print(f"\nFirst few files:")
            for f in saved_files[:5]:
                print(f"  - {f}")
            if len(saved_files) > 5:
                print(f"  ... and {len(saved_files) - 5} more")
        else:
            print("⚠️  No frames were extracted")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
