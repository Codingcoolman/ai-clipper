import os
from datetime import timedelta
from .caption_burner import CaptionStyle

class CaptionGenerator:
    def __init__(self):
        self.output_dir = "output/clips"

    def format_time(self, seconds):
        """Convert seconds to ASS timestamp format (H:MM:SS.cc)"""
        td = timedelta(seconds=float(seconds))
        hours = int(td.seconds // 3600)
        minutes = int((td.seconds % 3600) // 60)
        seconds = int(td.seconds % 60)
        centiseconds = int(td.microseconds / 10000)  # Convert to centiseconds
        
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    def generate_ass(self, transcript, clip_start, clip_end, output_path, style: CaptionStyle = None):
        """
        Generate ASS subtitle file from transcript segments within clip timeframe.
        """
        print(f"\nGenerating ASS file for clip {clip_start:.2f}s to {clip_end:.2f}s")
        print(f"Found {len(transcript)} transcript segments")
        
        if style is None:
            style = CaptionStyle()
        
        # ASS header with styles
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style.get_ass_style()}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Group words into natural lines
        lines = self.group_words_into_lines(transcript, clip_start, clip_end)
        
        # Generate dialogue events with fades
        events = []
        prev_end = None
        
        for start, end, text in lines:
            # Ensure minimum gap between captions
            if prev_end is not None:
                min_start = prev_end + (style.min_gap_ms / 1000)  # Convert ms to seconds
                if start < min_start:
                    start = min_start
            
            # Convert timestamps relative to clip
            start_time = self.format_time(max(0, start - clip_start))
            end_time = self.format_time(min(clip_end - clip_start, end - clip_start))
            
            # Escape ASS special characters: \, {, }
            text = text.replace('\\', r'\\').replace('{', r'\{').replace('}', r'\}')
            
            # Add fade effect
            fade_effect = f"{{\\fad({style.fade_in_ms},{style.fade_out_ms})}}"
            
            # Create dialogue line with fade
            event = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{fade_effect}{text}"
            events.append(event)
            print(f"Caption: {start_time}-{end_time}: {text} (with fade)")
            
            prev_end = end
        
        # Write ASS file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + '\n'.join(events) + '\n')
        
        print(f"Generated ASS file with {len(events)} captions at {output_path}")
        
        # Validate the generated file
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"\nFirst 500 characters of generated ASS:\n{content[:500]}")
        
        return output_path

    def get_subtitle_path(self, clip_path):
        """Generate subtitle file path from clip path"""
        base_path = os.path.splitext(clip_path)[0]
        return f"{base_path}.ass"  # Using .ass extension now

    def group_words_into_lines(self, words, clip_start, clip_end, max_gap=0.3, max_words=3):
        """
        Group words into natural caption lines based on timing gaps and word limits.
        
        Args:
            words: List of word segments with start/end times and text
            clip_start: Start time of the clip in seconds
            clip_end: End time of the clip in seconds
            max_gap: Maximum gap in seconds between words to keep them in same line
            max_words: Maximum number of words per caption line (default: 3)
            
        Returns:
            List of tuples (start_time, end_time, text)
        """
        # Filter words within clip timeframe
        clip_words = [
            word for word in words 
            if word['end'] >= clip_start and word['start'] <= clip_end
        ]
        
        lines = []
        current_line_words = []
        line_start = None
        line_end = None
        
        for word in sorted(clip_words, key=lambda x: x['start']):
            if not current_line_words:
                # First word in line
                current_line_words = [word['word'].strip().upper()]  # Convert to uppercase
                line_start = word['start']
                line_end = word['end']
            else:
                # Check if this word should start a new line
                gap = word['start'] - line_end
                word_count = len(current_line_words)
                
                if gap > max_gap or word_count >= max_words:
                    # Add current line to results
                    text = ' '.join(current_line_words)
                    lines.append((line_start, line_end, text))
                    # Start new line
                    current_line_words = [word['word'].strip().upper()]  # Convert to uppercase
                    line_start = word['start']
                else:
                    # Add to current line
                    current_line_words.append(word['word'].strip().upper())  # Convert to uppercase
                line_end = word['end']
        
        # Add final line if exists
        if current_line_words:
            text = ' '.join(current_line_words)
            lines.append((line_start, line_end, text))
        
        return lines 