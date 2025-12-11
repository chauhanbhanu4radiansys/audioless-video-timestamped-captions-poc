"""
Download utilities for pre-signed S3 URLs
"""
import os
import tempfile
import requests
from typing import Optional, Callable


def download_from_url(url: str, output_path: Optional[str] = None, chunk_size: int = 8 * 1024 * 1024, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Downloads a file from a pre-signed S3 URL or returns local file path if already local.
    
    Args:
        url: Pre-signed S3 URL or local file path (file:// or /path/to/file)
        output_path: Optional output path. If None, creates a temporary file (for URLs only).
        chunk_size: Chunk size for streaming download (default: 8MB for better performance)
        progress_callback: Optional callback function(status_message) for progress updates
        
    Returns:
        Path to downloaded file or local file path, or None if download failed
    """
    if progress_callback is None:
        progress_callback = print
    
    # Check if it's a local file path (file:// protocol or absolute path)
    local_path = None
    if url.startswith('file://'):
        local_path = url[7:]  # Remove 'file://' prefix
    elif url.startswith('/') and os.path.exists(url):
        local_path = url
    
    # If it's a local file, return it directly
    if local_path:
        if os.path.exists(local_path):
            progress_callback(f"Using local file: {local_path}")
            return local_path
        else:
            raise Exception(f"Local file path does not exist: {local_path}")
    
    session = None
    try:
        if output_path is None:
            # Create temporary file
            file_ext = os.path.splitext(url.split('?')[0])[1] or '.tmp'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            output_path = temp_file.name
            temp_file.close()
        
        progress_callback("Downloading...")
        
        # Use session for connection pooling and better performance
        session = requests.Session()
        session.headers.update({
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        # Download with streaming
        response = session.get(url, stream=True, timeout=300, allow_redirects=True)
        
        # Check status code
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.reason}")
        
        # Download without progress callbacks in the loop for maximum speed
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        # Verify file was downloaded
        if not os.path.exists(output_path):
            raise Exception("Downloaded file does not exist")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise Exception("Downloaded file is empty")
        
        progress_callback(f"Download complete: {file_size / (1024*1024):.2f} MB")
        return output_path
        
    except requests.exceptions.Timeout:
        error_msg = "Download timeout (300s exceeded)"
        print(f"Error: {error_msg}")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error: {str(e)}"
        print(f"Error: {error_msg}")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return None
    except Exception as e:
        error_msg = f"Error downloading from URL: {type(e).__name__}: {str(e)}"
        print(f"Error: {error_msg}")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return None
    finally:
        if session:
            session.close()
