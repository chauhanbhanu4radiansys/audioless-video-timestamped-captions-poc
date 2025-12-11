"""
Video processing functions: scene detection, frame extraction
"""
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


def detect_scenes_with_adaptive_detector(video_path: str) -> List:
    """
    Detect scenes using AdaptiveDetector.
    Returns all detected scenes without any minimum duration filtering.
    """
    video_stream = None
    with open(os.devnull, 'w') as devnull:
        with redirect_stderr(devnull):
            try:
                video_stream = open_video(video_path)
                scene_manager = SceneManager()
                custom_weights = ContentDetector.Components(
                    delta_hue=1.0,
                    delta_sat=1.0,
                    delta_lum=5.0,
                    delta_edges=0.0
                )
                scene_manager.add_detector(AdaptiveDetector(
                    adaptive_threshold=1.0,
                    weights=custom_weights
                ))
                scene_manager.detect_scenes(video_stream, show_progress=False)
                scene_list = scene_manager.get_scene_list()
                return scene_list
            except Exception as e:
                print(f"Scene detection failed for {os.path.basename(video_path)}: {e}")
                return []
            finally:
                if video_stream:
                    try:
                        if hasattr(video_stream, 'close'):
                            video_stream.close()
                    except Exception:
                        pass


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using multiple methods."""
    if not os.path.exists(video_path):
        return 0.0
    
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            if duration > 0:
                return duration
    except (FileNotFoundError, Exception):
        pass
    
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
                if duration > 0:
                    return duration
    except Exception:
        pass
    
    return 0.0


def resize_frame_optimized(frame_np: np.ndarray, max_size: int = 512) -> Image.Image:
    """Resize frame to reduce processing time while maintaining aspect ratio."""
    pil_image = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    return pil_image


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

