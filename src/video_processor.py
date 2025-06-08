import os
import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.caption_generator import CaptionGenerator
from src.caption_burner import CaptionBurner, CaptionStyle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClipRequest:
    """Data class to hold clip information"""
    start_time: float
    end_time: float
    output_path: str
    quality: str = 'high'  # 'high', 'medium', 'low'
    transcript: Optional[List[dict]] = None  # Add transcript field
    caption_style: Optional[CaptionStyle] = None  # Add caption styling

class VideoProcessor:
    def __init__(self, source_video_path: str):
        """Initialize the video processor with source video path."""
        if not os.path.exists(source_video_path):
            raise FileNotFoundError(f"Video file not found: {source_video_path}")
        
        self.source_video_path = source_video_path
        self.caption_generator = CaptionGenerator()
        self.caption_burner = CaptionBurner()
        logger.info(f"Initialized VideoProcessor for {os.path.basename(source_video_path)}")
    
    def extract_clip(self, clip_request: ClipRequest) -> bool:
        """Extract a single clip from the source video using optimized FFmpeg."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(clip_request.output_path), exist_ok=True)
            
            # Clean up any existing files
            temp_output = clip_request.output_path + ".temp.mp4"
            if os.path.exists(temp_output):
                os.remove(temp_output)
            if os.path.exists(clip_request.output_path):
                os.remove(clip_request.output_path)
            
            # Extract clip using FFmpeg with optimized settings
            logger.info(f"Extracting clip from {clip_request.start_time:.2f}s to {clip_request.end_time:.2f}s")
            duration = clip_request.end_time - clip_request.start_time
            
            # Optimized FFmpeg command for fast extraction
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-ss', str(clip_request.start_time),  # Seek position
                '-i', self.source_video_path,  # Input
                '-t', str(duration),  # Duration
                '-c:v', 'h264',  # Use hardware acceleration if available
                '-preset', 'veryfast',  # Fast encoding
                '-tune', 'zerolatency',  # Minimize latency
                '-movflags', '+faststart',  # Web optimization
                '-avoid_negative_ts', 'make_zero',
                temp_output
            ]
            
            # Run FFmpeg command
            subprocess.run(cmd, check=True, capture_output=True)
            
            final_output = clip_request.output_path
            
            # Generate captions if transcript is available
            if clip_request.transcript:
                ass_path = self.caption_generator.get_subtitle_path(temp_output)
                if os.path.exists(ass_path):
                    os.remove(ass_path)
                    
                self.caption_generator.generate_ass(
                    clip_request.transcript,
                    clip_request.start_time,
                    clip_request.end_time,
                    ass_path,
                    style=clip_request.caption_style
                )
                logger.info(f"Generated ASS captions at {ass_path}")
                
                # Burn captions if style is provided
                if clip_request.caption_style:
                    logger.info("Burning captions into video...")
                    
                    # Convert paths to absolute
                    temp_output_abs = os.path.abspath(temp_output)
                    ass_path_abs = os.path.abspath(ass_path)
                    final_output_abs = os.path.abspath(final_output)
                    
                    success = self.caption_burner.burn_captions(
                        temp_output_abs,
                        ass_path_abs,
                        final_output_abs,
                        clip_request.caption_style
                    )
                    
                    if success:
                        logger.info("Successfully burned captions into video")
                        os.remove(temp_output)
                        os.remove(ass_path)
                    else:
                        logger.error("Failed to burn captions into video")
                        os.rename(temp_output, final_output)
                else:
                    os.rename(temp_output, final_output)
            else:
                os.rename(temp_output, final_output)
            
            return os.path.exists(final_output)
                
        except Exception as e:
            logger.error(f"Error extracting clip: {str(e)}")
            return False 

    def extract_clips(self, clips: List[Tuple[float, float]], output_dir: str, quality: str = 'high', 
                     transcript: Optional[List[dict]] = None, caption_style: Optional[CaptionStyle] = None) -> List[str]:
        """
        Extract multiple clips from the source video in parallel.
        
        Args:
            clips: List of (start_time, end_time) tuples
            output_dir: Directory to save clips
            quality: Quality setting ('high', 'medium', 'low')
            transcript: Optional transcript data for caption generation
            caption_style: Optional styling for burned captions
            
        Returns:
            List[str]: List of paths to successfully created clips
        """
        successful_clips = []
        future_to_clip = {}
        
        # Process clips in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, (start, end) in enumerate(clips, 1):
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(self.source_video_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_clip_{i}.mp4")
                
                # Create clip request
                clip_request = ClipRequest(
                    start_time=start,
                    end_time=end,
                    output_path=output_path,
                    quality=quality,
                    transcript=transcript,
                    caption_style=caption_style
                )
                
                # Submit clip extraction task
                future = executor.submit(self.extract_clip, clip_request)
                future_to_clip[future] = output_path
            
            # Collect results as they complete
            for future in as_completed(future_to_clip):
                output_path = future_to_clip[future]
                try:
                    if future.result():
                        successful_clips.append(output_path)
                except Exception as e:
                    logger.error(f"Error processing clip {output_path}: {str(e)}")
                
        return successful_clips 