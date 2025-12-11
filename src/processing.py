"""
Video processing functions: scene detection, frame extraction
Optimized for fast frame extraction using ffmpeg + parallelism
"""
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector


def detect_scenes_with_adaptive_detector(video_path: str) -> List:
    """Detect scenes using AdaptiveDetector."""
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
                    delta_edges=0.0,
                )
                scene_manager.add_detector(
                    AdaptiveDetector(adaptive_threshold=1.0, weights=custom_weights)
                )
                scene_manager.detect_scenes(video_stream, show_progress=False)
                return scene_manager.get_scene_list()
            except Exception as err:  # pragma: no cover - diagnostics only
                print(f"Scene detection failed for {os.path.basename(video_path)}: {err}")
                return []
            finally:
                if video_stream and hasattr(video_stream, "close"):
                    try:
                        video_stream.close()
                    except Exception:
                        pass


def get_video_duration(video_path: str) -> float:
    """Best-effort duration lookup using ffprobe/cv2."""
    if not os.path.exists(video_path):
        return 0.0

    # Prefer ffprobe for accuracy/speed
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return max(float(result.stdout.strip()), 0.0)
    except (FileNotFoundError, ValueError):
        pass

    # Fallback to cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (frame_count / fps) if fps > 0 else 0.0


def resize_frame_optimized(frame_np: np.ndarray, max_size: int = 512) -> Image.Image:
    """Resize frame to reduce processing time while maintaining aspect ratio."""
    pil_image = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    return pil_image


def _ffmpeg_available() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=2)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _extract_frame_ffmpeg(video_path: str, timestamp: float) -> np.ndarray | None:
    """Extract frame at timestamp using ffmpeg (fast seek)."""
    try:
        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=5)
        if result.returncode != 0:
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width <= 0 or height <= 0:
            return None

        frame_bytes = result.stdout
        expected = width * height * 3
        if len(frame_bytes) != expected:
            return None
        frame_rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _extract_frame_cv2(video_path: str, frame_number: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_frames_from_scenes(video_path: str, scene_list: List, fps: float) -> List[Dict]:
    """Extract middle frame for each scene using fast I/O pipeline."""
    if not scene_list:
        return []

    ffmpeg_fast = _ffmpeg_available()
    print(f"Frame extraction method: {'ffmpeg (fast)' if ffmpeg_fast else 'cv2 (fallback)'}")

    tasks = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        middle_frame = start_frame + (end_frame - start_frame) // 2
        middle_ts = start_time + (end_time - start_time) / 2
        tasks.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "middle_frame": middle_frame,
                "middle_ts": middle_ts,
            }
        )

    frames_data: List[Dict] = []
    start = time.time()
    num_workers = min(8, len(tasks), os.cpu_count() or 4)

    def _extract(task: Dict) -> Dict | None:
        frame_np = None
        if ffmpeg_fast:
            frame_np = _extract_frame_ffmpeg(video_path, task["middle_ts"])
        if frame_np is None:
            frame_np = _extract_frame_cv2(video_path, task["middle_frame"])
        if frame_np is None:
            return None
        pil_img = resize_frame_optimized(frame_np)
        return {
            "start_time": task["start_time"],
            "end_time": task["end_time"],
            "start_frame": task["start_frame"],
            "end_frame": task["end_frame"],
            "middle_frame": task["middle_frame"],
            "frame_image": pil_img,
        }

    if num_workers > 1 and len(tasks) > 1:
        print(f"Extracting {len(tasks)} frames in parallel with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(_extract, task): idx for idx, task in enumerate(tasks)}
            completed = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                if result:
                    frames_data.append(result)
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"  Extracted {completed}/{len(tasks)} frames...")
    else:
        print(f"Extracting {len(tasks)} frames sequentially...")
        for idx, task in enumerate(tasks):
            result = _extract(task)
            if result:
                frames_data.append(result)
            if (idx + 1) % 10 == 0 or idx + 1 == len(tasks):
                print(f"  Extracted {idx + 1}/{len(tasks)} frames...")

    elapsed = time.time() - start
    if frames_data:
        avg = elapsed / len(frames_data)
        print(
            f"✓ Frame extraction completed: {len(frames_data)} frames in {elapsed:.2f}s "
            f"({avg:.3f}s/frame)"
        )
    frames_data.sort(key=lambda item: item["start_time"])
    return frames_data
