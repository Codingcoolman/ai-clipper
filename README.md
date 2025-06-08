# AI Clipper

An AI-powered video clipping platform that automatically extracts, processes, and enhances the most engaging segments from videos.

## Project Status

### ✅ Step 1: YouTube Video Download
- Implemented using yt-dlp
- Successfully downloads videos to output directory
- Returns reliable file paths
- Tested and working

### ✅ Step 2: Word-Level Transcription with WhisperX
- Implemented using Whisper
- Generates accurate transcriptions with timestamps
- Returns structured transcript data
- Tested and working

### ✅ Step 3: Select Noteworthy Clips from Transcript
- Implemented clip selection logic
- Identifies engaging segments
- Returns start/end timestamp pairs
- Tested and working

### ✅ Step 4: Video Clipping
- Implemented using moviepy
- Successfully extracts clips from source video
- Saves to output/clips directory
- Tested and working

### 🔄 Step 5: Generate Captions for Clips (Next Step)
Implementation Plan:
1. Create `src/caption_generator.py`:
   ```python
   class CaptionGenerator:
       def __init__(self):
           self.output_dir = "output/clips"
   
       def generate_srt(self, transcript, clip_start, clip_end, output_path):
           # Convert transcript to SRT format
           # Adjust timestamps relative to clip start
           pass
   
       def format_time(self, seconds):
           # Convert seconds to SRT timestamp format
           # Format: 00:00:00,000
           pass
   ```

2. Required Functions:
   - Parse Whisper timestamps
   - Filter transcript segments for clip
   - Adjust timestamps relative to clip start
   - Generate SRT format
   - Handle multi-line captions
   - UTF-8 encoding support

3. Integration Points:
   - Update VideoProcessor to call CaptionGenerator
   - Store .srt files alongside clips
   - Match subtitle files with video files

4. Testing Plan:
   - Test with sample transcript
   - Verify timestamp alignment
   - Check UTF-8 handling
   - Validate SRT format

### ⏳ Step 6: Burn Captions Into Clips
- TODO: Implement caption burning using ffmpeg
- TODO: Create hardcoded caption embedding
- TODO: Test visual output

## Recent Optimizations (March 2024)

### 1. Transcription Improvements
- Migrated from vanilla Whisper to faster-whisper for improved speed
- Added parallel processing and hardware acceleration
- Set default confidence value of 1.0 for segment compatibility

### 2. Video Processing Enhancements
- Replaced moviepy with direct FFmpeg commands
- Implemented parallel clip processing using ThreadPoolExecutor
- Added hardware acceleration for video encoding
- Optimized FFmpeg settings (veryfast preset, zerolatency tune)

### 3. Caption Quality Improvements
- Enhanced caption rendering quality with proper resolution handling
- Optimized caption burning process:
  - Captions rendered at full resolution before cropping
  - Guaranteed even dimensions for x264 encoding
  - Improved font settings and readability
- Added hardware acceleration support for caption burning
- Streamlined two-step process (crop then caption)

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd ai-clipper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Required System Dependencies:
- FFmpeg (latest version)
- Python 3.8+
- CUDA-compatible GPU (optional, for hardware acceleration)

## Configuration

The system uses the following optimized settings:

- Video Processing:
  - Hardware-accelerated encoding (when available)
  - Optimized FFmpeg presets for speed/quality balance
  - Parallel processing for multiple clips

- Caption Styling:
  - Font: Impact (default)
  - Font Size: 72 (relative to video resolution)
  - Bold text with black outline for readability
  - Centered positioning with adequate margins

## Usage

1. Start the application:
```bash
python run.py
```

2. The server will start on `http://localhost:5000`

## Performance Notes

- Processing Time: ~2-3 minutes for 5-minute videos (with optimizations)
- Hardware Acceleration: Automatically used when available
- Memory Usage: Optimized for parallel processing

## Dependencies

Key dependencies and their versions:
- faster-whisper >= 0.10.0
- ffmpeg-python >= 0.2.0
- flask >= 3.0.0
- torch >= 2.2.0
- numpy >= 1.24.0

## Known Limitations

- Video dimensions must be even numbers for x264 encoding
- Hardware acceleration requires compatible GPU
- Some features may require specific FFmpeg version

## Future Improvements

- Further optimization of parallel processing
- Enhanced error handling and recovery
- Additional caption styling options
- Support for more video formats

## Troubleshooting

If you encounter issues:

1. Ensure FFmpeg is properly installed and accessible
2. Check GPU compatibility for hardware acceleration
3. Verify all dependencies are correctly installed
4. Check system logs for detailed error messages

## License

[Your License Here]

## Current Features
- YouTube video download
- Automatic transcription
- Intelligent clip selection
- Video clip extraction
- Web interface for processing
- Real-time progress monitoring

## Project Structure
```
AI Clipper/
├── src/
│   ├── app.py              # Flask web server
│   ├── downloader.py       # YouTube download logic
│   ├── transcriber.py      # Whisper transcription
│   ├── clip_selector.py    # Clip selection logic
│   └── video_processor.py  # Video processing
├── output/
│   └── clips/             # Extracted video clips
└── README.md
```

## Next Steps
1. Implement subtitle generation for clips (Step 5)
2. Add caption burning functionality (Step 6)
3. Test and refine caption alignment
4. Add visual confirmation of caption embedding

## Technical Stack
- Python 3.x
- Flask
- yt-dlp
- Whisper
- moviepy
- ffmpeg

## Recent Updates
- Fixed video clip serving in web interface
- Implemented moviepy-based clip extraction
- Added CORS support for local development
- Improved error handling in file serving
- Successfully tested end-to-end pipeline from video download to clip viewing
- Resolved x264 encoding dimension issues with proper scaling
- Optimized caption burning process for HD quality
- Fixed FFmpeg pipeline for vertical video processing

## Immediate Next Steps

### 1. UI/UX Improvements
- [ ] Add loading indicators for clip processing
- [ ] Implement clip preview before download
- [ ] Add progress bars for video download and processing
- [ ] Improve error messaging and user feedback
- [ ] Add clip reordering capability
- [ ] Implement manual clip adjustment tools

### 2. Video Quality & Format Options
- [ ] Add quality selection for clip extraction
- [ ] Support multiple output formats
- [ ] Implement different aspect ratios (9:16, 1:1, 16:9)
- [ ] Add watermark options
- [ ] Include custom resolution settings

### 3. Caption Integration
- [ ] Add automatic caption overlay on clips
- [ ] Implement caption styling options
- [ ] Add caption positioning controls
- [ ] Support multiple caption formats

### 4. Batch Processing
- [ ] Enable processing multiple videos simultaneously
- [ ] Add queue management system
- [ ] Implement batch export options
- [ ] Add progress tracking for multiple videos

### 5. Performance Optimization
- [ ] Implement caching for processed videos
- [ ] Optimize clip extraction speed
- [ ] Add background job processing
- [ ] Implement cleanup for temporary files

## Future Enhancements
- Social media direct upload integration
- Custom clip selection algorithms
- User accounts and clip history
- Advanced video editing features
- API for programmatic access
- Mobile-responsive interface

## Dependencies
- Flask & Flask-CORS for web interface
- SentenceTransformer for semantic analysis
- NLTK for text processing
- yt-dlp for video download
- whisper for transcription
- moviepy for video processing

## Critical Implementation Details

### FFmpeg Caption Pipeline
```bash
# Current working FFmpeg command structure:
ffmpeg -i input.mp4 \
  -vf "crop=in_h*9/16:in_h,scale=1080:1920:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2,ass=subtitles.ass" \
  -c:v h264 -preset veryfast -tune zerolatency \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  output.mp4
```

### Key Configuration Values
- Video Resolution: 1080x1920 (9:16 vertical)
- Font Size: 72 (scaled relative to 1080p)
- Border Size: 4px for readability
- Default Margins: Vertical=20, Horizontal=100

### Recent Bug Fixes
- Resolved odd-dimension encoding error with trunc(iw/2)*2 scaling
- Fixed caption quality loss by applying captions after initial scaling
- Optimized processing speed with single-pass FFmpeg operation
- Corrected font rendering issues with proper ASS style configuration

### Current Processing Pipeline
1. Download (yt-dlp with format selection)
2. Transcribe (faster-whisper with GPU acceleration)
3. Select clips (optimized algorithm)
4. Process video (FFmpeg with hardware acceleration)
5. Burn captions (single-pass FFmpeg operation)

### Performance Metrics
- Download: ~30s for 10min video
- Transcription: ~1min for 10min video
- Processing: ~2-3min for 5min video
- Caption Burning: ~1min per clip 