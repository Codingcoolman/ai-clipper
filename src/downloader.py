import os
from pathlib import Path
import yt_dlp
import logging
import werkzeug.utils
from typing import Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_uploaded_video(video_file, output_dir: str = "output") -> str:
    """
    Save an uploaded video file and return its local file path.
    
    Args:
        video_file: The uploaded file object from request.files
        output_dir (str): Directory to save the video
        
    Returns:
        str: Path to the saved video file
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Securely save the file with its original name
        filename = werkzeug.utils.secure_filename(video_file.filename)
        file_path = str(output_path / filename)
        
        # If a file with this name already exists, add a number to make it unique
        base, ext = os.path.splitext(file_path)
        counter = 1
        while os.path.exists(file_path):
            file_path = f"{base}_{counter}{ext}"
            counter += 1
        
        video_file.save(file_path)
        logger.info(f"Video uploaded successfully to: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded video: {str(e)}")
        raise

def download_youtube_video(url: str, output_dir: str = "output", progress_callback: Optional[Callable[[float, str], None]] = None) -> str:
    """
    Download a YouTube video and return its local file path.
    
    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to save the video
        progress_callback (Optional[Callable[[float, str], None]]): Callback for progress updates
        
    Returns:
        str: Path to the downloaded video file
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def progress_hook(d):
            if d['status'] == 'downloading':
                # Calculate download progress
                if 'total_bytes' in d:
                    progress = (d['downloaded_bytes'] / d['total_bytes']) * 100
                elif 'total_bytes_estimate' in d:
                    progress = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
                else:
                    progress = 0
                
                # Get download speed and ETA
                if 'speed' in d and d['speed'] is not None:
                    speed = d['speed'] / 1024 / 1024  # Convert to MB/s
                    speed_str = f"{speed:.1f} MB/s"
                else:
                    speed_str = "-- MB/s"
                
                if 'eta' in d and d['eta'] is not None:
                    eta_str = f"{d['eta']}s"
                else:
                    eta_str = "--"
                
                status = f"Downloading: {speed_str} (ETA: {eta_str})"
                
                if progress_callback:
                    progress_callback(progress, status)
                    
            elif d['status'] == 'finished':
                if progress_callback:
                    progress_callback(100, "Download complete, processing video...")
        
        # Configure yt-dlp options for best quality
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',  # Best video+audio or fallback to best combined
            'outtmpl': str(output_path / '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'noplaylist': True,
            'progress_hooks': [progress_hook],
            'verbose': True,  # Add verbose output for debugging
        }
        
        # Download the video
        logger.info(f"Starting download from {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First extract info to get the title
            if progress_callback:
                progress_callback(0, "Extracting video information...")
                
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise ValueError("Could not extract video information")
                
            video_title = info.get('title', 'video')
            safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            video_path = str(output_path / f"{safe_title}.mp4")
            
            # Now download with the known title
            logger.info(f"Extracting video: {safe_title}")
            ydl.download([url])
            
            if not os.path.exists(video_path):
                # If the exact path doesn't exist, try to find any .mp4 file with a similar name
                potential_files = list(output_path.glob(f"{safe_title}*.mp4"))
                if potential_files:
                    video_path = str(potential_files[0])
                else:
                    raise FileNotFoundError(f"Could not find downloaded video in {output_path}")
            
            logger.info(f"Video successfully downloaded to: {video_path}")
            return video_path
            
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise 