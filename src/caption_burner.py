import os
import subprocess
from dataclasses import dataclass
from typing import Optional, Literal
import sys

@dataclass
class CaptionStyle:
    # Font settings
    font_family: str = 'Impact'  # Keeping Impact as default
    font_size: int = 72  # Size relative to original video resolution
    font_color: str = '&H00FFFFFF'  # White in ASS hex format (AABBGGRR)
    font_bold: bool = True
    font_italic: bool = False
    font_underline: bool = False
    
    # Border and background
    border_color: str = '&H00000000'  # Black border
    border_size: int = 4  # Thicker border for readability
    background_color: str = '&H00000000'  # No background
    shadow_size: int = 0  # No shadow
    
    # Position and alignment
    position: Literal['top', 'middle', 'bottom'] = 'middle'
    alignment: Literal['left', 'center', 'right'] = 'center'
    margin_vertical: int = 20  # Increased vertical margin for better spacing
    margin_horizontal: int = 100  # Increased horizontal margins
    
    # Text scaling
    scale_x: int = 100  # Normal horizontal scaling
    scale_y: int = 100  # Normal vertical scaling
    spacing: int = 0  # Letter spacing
    angle: int = 0  # Text rotation angle
    
    # Fade effects (in milliseconds)
    fade_in_ms: int = 200  # Slightly longer fade in
    fade_out_ms: int = 150  # Slightly longer fade out
    min_gap_ms: int = 50  # Minimum gap between captions
    
    def get_ass_style(self) -> str:
        """Convert style settings to ASS style format."""
        # Convert position and alignment to ASS values
        pos_align = {
            ('top', 'left'): 7,
            ('top', 'center'): 8,
            ('top', 'right'): 9,
            ('middle', 'left'): 4,
            ('middle', 'center'): 5,
            ('middle', 'right'): 6,
            ('bottom', 'left'): 1,
            ('bottom', 'center'): 2,
            ('bottom', 'right'): 3
        }
        alignment = pos_align.get((self.position, self.alignment), 2)  # Default to bottom-center
        
        # Convert boolean values to 0/1
        bold = 1 if self.font_bold else 0
        italic = 1 if self.font_italic else 0
        underline = 1 if self.font_underline else 0
        
        # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,
        # Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle,
        # Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        return (
            f"Style: Default,{self.font_family},{self.font_size},"
            f"{self.font_color},&H00FFFFFF,{self.border_color},{self.background_color},"
            f"{bold},{italic},{underline},0,"  # StrikeOut always 0
            f"{self.scale_x},{self.scale_y},{self.spacing},{self.angle},"
            f"1,{self.border_size},{self.shadow_size},"  # BorderStyle=1 (outline)
            f"{alignment},{self.margin_horizontal},{self.margin_horizontal},{self.margin_vertical},1"
        )

class CaptionBurner:
    def __init__(self):
        self.default_style = CaptionStyle()
        # Default Windows font paths
        self.system_fonts = {
            'Arial': 'C:/Windows/Fonts/arial.ttf',
            'Calibri': 'C:/Windows/Fonts/calibri.ttf',
            'TimesNewRoman': 'C:/Windows/Fonts/times.ttf',
            'Impact': 'C:/Windows/Fonts/impact.ttf',
            'ComicSansMS': 'C:/Windows/Fonts/comic.ttf',
            'Verdana': 'C:/Windows/Fonts/verdana.ttf'
        }
        
    def _get_font_path(self, font_name: str) -> str:
        """Get the full path to the font file."""
        if font_name in self.system_fonts and os.path.exists(self.system_fonts[font_name]):
            return self.system_fonts[font_name]
        # Fallback to Arial if specified font not found
        if os.path.exists(self.system_fonts['Arial']):
            return self.system_fonts['Arial']
        raise FileNotFoundError(f"Could not find font file for {font_name}")

    def _format_path_for_ffmpeg(self, path: str) -> str:
        """Format a path for ffmpeg compatibility."""
        # Replace backslashes with forward slashes
        path = path.replace('\\', '/')
        # Escape colons
        path = path.replace(':', r'\:')
        return path

    def _parse_srt(self, srt_path: str) -> list:
        """Parse SRT file into a list of caption entries."""
        captions = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Split into caption blocks
        blocks = content.split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:  # Valid caption block should have at least 3 lines
                try:
                    # Parse timecode line
                    timecode = lines[1].strip()
                    start_time, end_time = timecode.split(' --> ')
                    
                    # Join all text lines
                    text = ' '.join(lines[2:])
                    
                    # Convert timecodes to seconds
                    start_seconds = self._timestamp_to_seconds(start_time)
                    end_seconds = self._timestamp_to_seconds(end_time)
                    
                    captions.append({
                        'start': start_seconds,
                        'end': end_seconds,
                        'text': text
                    })
                    
                    print(f"Parsed caption: {start_seconds}-{end_seconds}: {text}")
                    
                except Exception as e:
                    print(f"Error parsing caption block: {str(e)}")
                    continue
        
        return captions

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp to seconds."""
        # Format: 00:00:00,000
        timestamp = timestamp.strip()
        hours, minutes, remainder = timestamp.split(':')
        seconds, milliseconds = remainder.split(',')
        total_seconds = (int(hours) * 3600 + 
                        int(minutes) * 60 + 
                        int(seconds) + 
                        int(milliseconds) / 1000)
        return total_seconds

    def burn_captions(self, input_video: str, ass_path: str, output_path: str, style: Optional[CaptionStyle] = None) -> bool:
        """
        Burn ASS subtitles into video using optimized FFmpeg settings.
        Process:
        1. Add captions to original video
        2. Crop to vertical format
        Uses hardware acceleration when available and optimized encoding settings.
        """
        try:
            if not os.path.exists(ass_path):
                print(f"ASS subtitle file not found: {ass_path}")
                return False
            
            # Get video dimensions
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                input_video
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error getting video dimensions: {result.stderr}")
                return False
                
            video_size = result.stdout.strip()
            print(f"Video dimensions: {video_size}")
            
            # Format ASS path for FFmpeg
            ass_path = ass_path.replace('\\', '/').replace(':', '\\:')
            
            # Single-pass command that combines cropping, scaling, and caption burning
            cmd = [
                'ffmpeg',
                '-y',
                '-i', input_video,
                # Ensure width and height are even numbers with scale filter
                '-vf', f"crop=in_h*9/16:in_h,scale=1080:1920:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2,ass='{ass_path}'",
                '-c:v', 'h264',  # Use h264 for hardware acceleration
                '-preset', 'veryfast',  # Fast encoding
                '-tune', 'zerolatency',  # Minimize latency
                '-c:a', 'aac',  # AAC audio codec
                '-b:a', '128k',  # Audio bitrate
                '-movflags', '+faststart',  # Web optimization
                output_path
            ]
            
            print("\nProcessing video with HD captions:")
            print(' '.join(cmd))
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            return os.path.exists(output_path)
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            return False
        except Exception as e:
            print(f"Error burning captions: {str(e)}")
            return False 