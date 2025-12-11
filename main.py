"""
Main video processing pipeline: Extract frames and generate descriptions
"""
import os
import sys
import uuid
import cv2
from typing import List, Dict, Any

# CPU optimization: Set environment variables before importing torch
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    # Optimize CPU performance
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() or 4)
    os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count() or 4)

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
    # Step 1: Get video FPS using cv2
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
    
    # Use optimized batch_size=16 and num_workers=6 (from blip_faster.py)
    batch_size = 16
    num_workers = 6
    descriptions = generate_scene_descriptions(
        pil_images, 
        caption_processor, 
        caption_model, 
        device,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
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

