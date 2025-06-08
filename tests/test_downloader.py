import pytest
from pathlib import Path
from src.downloader import download_youtube_video

def test_download_youtube_video():
    """Test downloading a short YouTube video."""
    # Use a short Creative Commons video for testing
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - First YouTube video
    output_dir = "output/test"
    
    # Download video
    video_path = download_youtube_video(test_url, output_dir)
    
    # Verify file exists and has size > 0
    assert Path(video_path).exists()
    assert Path(video_path).stat().st_size > 0
    
    # Clean up
    Path(video_path).unlink() 