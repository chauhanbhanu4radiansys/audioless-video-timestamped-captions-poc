import os
import sys
import json
from typing import Dict, Any

# Optional runpod import (only needed for RunPod serverless deployment)
try:
    import runpod
except ImportError:
    runpod = None

# Add parent directory to Python path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notify import notify
from src.download import download_from_url
from main import process_video_frames
from dotenv import load_dotenv

load_dotenv()


def download(inpath, outpath):
    """Simple download function for compatibility."""
    import requests
    r = requests.get(inpath, stream=True, timeout=300)
    r.raise_for_status()
    with open(outpath, 'wb') as f:
        f.write(r.content)
    return True


def cleanup_temp_files(*file_paths):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✓ Cleaned up temporary file: {file_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not clean up {file_path}: {e}")


def handle_embedding(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle video processing requests.
    
    Args:
        input_data: Dictionary with video processing parameters
            - videoPathURL: S3 URL or local file path
            - video_name: Name of the video (optional)
            - video_id: Unique ID (optional, auto-generated)
            - tenant: Tenant ID (for notifications)
            - id: Job ID (for notifications)
            - snsTopicArn: SNS topic ARN (for notifications)
        
    Returns:
        JSON response with processing results
    """
    videoPathURL = input_data.get('videoPathURL')
    video_name = input_data.get('video_name', 'unknown_video')
    video_id = input_data.get('video_id')
    job_id = input_data.get('id')
    tenant = input_data.get('tenant')
    sns_topic_arn = input_data.get('snsTopicArn')
    
    # Validate required parameters
    if not videoPathURL:
        raise ValueError("Missing required parameter: videoPathURL")
    
    print("=" * 80)
    print("Starting Video Frame Extraction")
    print("=" * 80)
    print(f"Video URL/Path: {videoPathURL[:100] if len(videoPathURL) > 100 else videoPathURL}...")
    print(f"Video Name: {video_name}")
    print(f"Video ID: {video_id or 'auto-generated'}")
    print("=" * 80)
    
    # Download video if it's a URL (download_from_url handles local paths too)
    local_video_path = None
    try:
        local_video_path = download_from_url(videoPathURL)
        if not local_video_path or not os.path.exists(local_video_path):
            raise Exception(f"Failed to download video from: {videoPathURL[:100]}...")
        print(f"Video ready at: {local_video_path}")
    except Exception as e:
        error_msg = f"Download failed: {str(e)}"
        print(f"❌ {error_msg}")
        if tenant and job_id and sns_topic_arn:
            notify(tenant, job_id, False, sns_topic_arn, {'error': error_msg})
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': error_msg
            })
        }
    
    # Process video
    try:
        results = process_video_frames(local_video_path)
        
        print("=" * 80)
        print("Video Frame Extraction Completed Successfully")
        print("=" * 80)
        print(f"Extracted {len(results)} frames")
        
        # Send success notification
        if tenant and job_id and sns_topic_arn:
            notify(tenant, job_id, True, sns_topic_arn, {
                'video_name': video_name,
                'video_id': video_id,
                'frames_extracted': len(results)
            })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'message': 'Video frames extracted successfully',
                'video_name': video_name,
                'video_id': video_id,
                'frames_count': len(results),
                'results': results
            }, indent=2)
        }
        
    except Exception as e:
        error_message = str(e)
        print(f"❌ Error processing video: {error_message}")
        import traceback
        traceback.print_exc()
        
        # Send failure notification
        if tenant and job_id and sns_topic_arn:
            notify(tenant, job_id, False, sns_topic_arn, {'error': error_message})
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': error_message
            })
        }
    finally:
        # Cleanup downloaded file
        if local_video_path and os.path.exists(local_video_path):
            try:
                cleanup_temp_files(local_video_path)
            except Exception:
                pass


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler"""
    try:
        input_data = job.get('input', {})
        return handle_embedding(input_data)
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': str(e)
            })
        }


# For RunPod serverless
if runpod is not None:
    runpod.serverless.start({"handler": handler})
