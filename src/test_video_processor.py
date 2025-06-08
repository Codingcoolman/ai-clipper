import os
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_processor import VideoProcessor, ClipRequest

def test_single_clip():
    # Test video path
    video_path = os.path.join('output', 'If you can spare me 5 minutes, you\'ll get 5 years of your life back.-1VPqXCMcPy8.mp4')
    
    print(f"Testing video path: {video_path}")
    print(f"File exists: {os.path.exists(video_path)}")
    
    # Create output directory for test clips
    output_dir = os.path.join('output', 'test_clips')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video processor
    processor = VideoProcessor(video_path)
    
    # Test just one clip - the "I still remember when I was a kid" section
    clip_request = ClipRequest(
        start_time=3.560,
        end_time=16.940,
        output_path=os.path.join(output_dir, "test_clip_1.mp4"),
        quality='high'
    )
    
    # Extract the clip
    success = processor.extract_clip(clip_request)
    
    if success:
        print(f"Clip extracted successfully!")
        size = os.path.getsize(clip_request.output_path) / (1024 * 1024)  # Convert to MB
        print(f"Clip size: {size:.2f}MB")
    else:
        print("Failed to extract clip")

if __name__ == '__main__':
    test_single_clip() 