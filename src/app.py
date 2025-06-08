import os
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from flask import Flask, request, jsonify, send_from_directory, render_template, session, current_app, copy_current_request_context
from flask_cors import CORS
import werkzeug.utils
import subprocess
import logging
import secrets
import sys
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for Render
)
logger = logging.getLogger(__name__)

# Local imports
try:
    from src.downloader import download_youtube_video, save_uploaded_video
    from src.transcriber import transcribe_video
    from src.clip_selector import ClipSelector
    from src.video_processor import VideoProcessor, ClipRequest
    from src.caption_burner import CaptionStyle
    logger.info("Successfully imported all local modules")
except Exception as e:
    logger.error(f"Error importing local modules: {str(e)}")
    raise

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Add startup status tracking
startup_status = {
    "ready": False,
    "message": "Server is starting...",
    "start_time": time.time(),
    "errors": []
}

def initialize_app():
    """Initialize application components"""
    try:
        logger.info("Starting application initialization...")
        
        # Ensure output directory exists
        output_dir = os.path.abspath('output')
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ensured at {output_dir}")
        
        # Initialize VideoProcessor (or any other components)
        global video_processor
        video_processor = VideoProcessor()
        logger.info("VideoProcessor initialized")
        
        # Mark as ready
        startup_status["ready"] = True
        startup_status["message"] = "Server is ready"
        logger.info("Application initialization complete")
        
    except Exception as e:
        error_msg = f"Error during initialization: {str(e)}"
        logger.error(error_msg)
        startup_status["errors"].append(error_msg)
        startup_status["message"] = error_msg
        raise

# Start initialization in the main thread
try:
    initialize_app()
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}")

def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = 'https://init-12295.web.app'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

@app.route('/health')
def health_check():
    """Enhanced health check endpoint"""
    uptime = time.time() - startup_status["start_time"]
    status_info = {
        "status": "ready" if startup_status["ready"] else "starting",
        "message": startup_status["message"],
        "uptime": f"{uptime:.2f} seconds",
        "errors": startup_status["errors"],
        "environment": {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "output_directory": os.path.abspath('output')
        }
    }
    response = jsonify(status_info)
    return add_cors_headers(response)

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        "status": "ok",
        "message": "AI Clipper API is running",
        "ready": startup_status["ready"]
    })

# Initialize models in a background thread
def init_models():
    try:
        logger.info("Initializing models...")
        global clip_selector
        clip_selector = ClipSelector()
        clip_selector.wait_for_models(timeout=60)
        startup_status["ready"] = True
        startup_status["message"] = "Server is ready"
        logger.info("Models initialized successfully")
    except Exception as e:
        startup_status["message"] = f"Error initializing models: {str(e)}"
        logger.error(f"Error initializing models: {str(e)}")

# Start model initialization in background
threading.Thread(target=init_models).start()

# Initialize clip selector
clip_selector = ClipSelector()

# Task progress tracking
task_progress = {}
task_lock = Lock()
task_results: Dict[str, dict] = {}  # Add storage for results

# Ad state tracking
ad_states: Dict[str, dict] = {}
ad_lock = Lock()

def init_ad_state(task_id: str):
    """Initialize ad state for a task"""
    with ad_lock:
        ad_states[task_id] = {
            'ads_completed': 0,
            'last_ad_time': 0,
            'ads_required': 3,
            'verified': False
        }

def verify_ad_completion(task_id: str) -> bool:
    """Check if all required ads have been viewed"""
    with ad_lock:
        if task_id not in ad_states:
            return False
        return ad_states[task_id]['verified']

def estimate_processing_time(video_length_seconds: float, num_clips: int = 3) -> dict:
    """
    Calculate estimated processing times based on real-world performance data
    Returns a dict of processing phases with their estimated times and weights
    """
    video_length_minutes = video_length_seconds / 60
    
    # Real-world calibration:
    # A 3:42 (222s) video takes about 75s total to process
    # Using this as our baseline for scaling
    
    # Download: Very fast, about 5-10s total regardless of video length
    download_time = 5 + (video_length_minutes * 2)  # Base 5s + 2s per minute
    
    # Transcription: About 0.5x realtime (twice as fast as realtime)
    # For a 222s video, transcription took about 30s
    transcribe_time = video_length_seconds * 0.5
    
    # Analysis: Quick, about 10-15s total
    # Base loading time + quick processing
    analysis_time = 10 + (video_length_minutes * 1)  # Base 10s + 1s per minute
    
    # Processing: About 5s per clip
    # Each clip takes about 5s total (including extraction and caption burning)
    process_time = num_clips * 5
    
    # Calculate total time and weights
    total_time = download_time + transcribe_time + analysis_time + process_time
    
    return {
        "download": {
            "time": download_time,
            "weight": download_time / total_time
        },
        "transcribe": {
            "time": transcribe_time,
            "weight": transcribe_time / total_time
        },
        "analyze": {
            "time": analysis_time,
            "weight": analysis_time / total_time
        },
        "process": {
            "time": process_time,
            "weight": process_time / total_time
        },
        "error": {  # Add error phase with zero weight
            "time": 0,
            "weight": 0
        }
    }

def calculate_eta(task_data: dict) -> str:
    """Calculate ETA based on current phase and progress"""
    current_time = time.time()
    start_time = task_data["start_time"]
    progress = task_data["progress"]
    current_phase = task_data["current_phase"]
    phase_start_time = task_data["phase_start_time"]
    processing_phases = task_data["processing_phases"]
    
    # If in error state, return immediately
    if current_phase == "error":
        return "Error occurred"
    
    # Calculate total estimated time
    total_estimated_seconds = sum(
        phase["time"] for phase_name, phase in processing_phases.items()
        if phase_name != "error"  # Exclude error phase from calculations
    )
    
    if progress <= 0:
        # Return the initial total estimate
        if total_estimated_seconds < 60:
            return f"{int(total_estimated_seconds)}s total"
        elif total_estimated_seconds < 3600:
            return f"{int(total_estimated_seconds/60)}m total"
        else:
            hours = int(total_estimated_seconds/3600)
            minutes = int((total_estimated_seconds % 3600) / 60)
            return f"{hours}h {minutes}m total"
    
    # Calculate elapsed time for current phase
    phase_elapsed_time = current_time - phase_start_time
    
    # Calculate overall progress including weighted phase progress
    completed_phases_weight = sum(
        phase["weight"] 
        for phase_name, phase in processing_phases.items()
        if phase_name in task_data["completed_phases"] and phase_name != "error"
    )
    
    current_phase_progress = (progress / 100.0) * processing_phases[current_phase]["weight"]
    total_progress = (completed_phases_weight + current_phase_progress) * 100
    
    if total_progress >= 100:
        return "Complete!"
    
    # Calculate remaining time based on progress and phase timing
    elapsed_total_time = current_time - start_time
    progress_rate = total_progress / elapsed_total_time if elapsed_total_time > 0 else 0
    
    if progress_rate <= 0:
        return "Calculating..."
    
    remaining_progress = 100 - total_progress
    remaining_seconds = remaining_progress / progress_rate
    
    # Format remaining time string
    if remaining_seconds < 60:
        return f"{int(remaining_seconds)}s remaining"
    elif remaining_seconds < 3600:
        return f"{int(remaining_seconds/60)}m remaining"
    else:
        hours = int(remaining_seconds/3600)
        minutes = int((remaining_seconds % 3600) / 60)
        return f"{hours}h {minutes}m remaining"

def update_task_progress(task_id: str, progress: float, status: Optional[str] = None, phase: Optional[str] = None):
    """Update the progress of a task"""
    with task_lock:
        if task_id not in task_progress:
            # Default to 3-minute video length if not provided
            # This will be updated when we get the actual video length
            processing_phases = estimate_processing_time(180, 3)
            
            task_progress[task_id] = {
                "progress": 0,
                "eta": "Estimating...",
                "status": "Initializing...",
                "start_time": time.time(),
                "last_update_time": time.time(),
                "current_phase": "download",
                "phase_start_time": time.time(),
                "completed_phases": [],  # Changed from set() to list
                "last_progress": 0,
                "processing_phases": processing_phases,
                "video_length": 180,  # Will be updated when we get the actual length
                "num_clips": 3        # Will be updated when specified
            }
        
        current_time = time.time()
        task_data = task_progress[task_id]
        
        # Update phase if provided
        if phase and phase != task_data["current_phase"]:
            logger.info(f"Task {task_id} changing phase from {task_data['current_phase']} to {phase}")
            # Mark previous phase as completed if progress was at 100%
            if task_data["progress"] >= 100 and task_data["current_phase"] not in task_data["completed_phases"]:
                task_data["completed_phases"].append(task_data["current_phase"])
            # Start new phase
            task_data["current_phase"] = phase
            task_data["phase_start_time"] = current_time
            task_data["progress"] = 0  # Reset progress for new phase
        
        # Update progress and status
        if progress is not None:
            task_data["progress"] = progress
            task_data["last_update_time"] = current_time
            
        if status:
            task_data["status"] = status
            logger.info(f"Task {task_id} status update: {status}")
            
        # Calculate total progress
        completed_phases_weight = sum(
            phase["weight"] 
            for phase_name, phase in task_data["processing_phases"].items()
            if phase_name in task_data["completed_phases"] and phase_name != "error"
        )
        
        current_phase_progress = (progress / 100.0) * task_data["processing_phases"][task_data["current_phase"]]["weight"]
        task_data["total_progress"] = (completed_phases_weight + current_phase_progress) * 100
        
        # Update ETA
        task_data["eta"] = calculate_eta(task_data)
        
        logger.info(f"Task {task_id} progress update: {progress}% ({task_data['total_progress']:.1f}% total) - {status}")

def update_task_video_info(task_id: str, video_length_seconds: float, num_clips: int):
    """Update task with actual video length and number of clips"""
    with task_lock:
        if task_id in task_progress:
            task_data = task_progress[task_id]
            task_data["video_length"] = video_length_seconds
            task_data["num_clips"] = num_clips
            # Recalculate processing phases with actual video length
            task_data["processing_phases"] = estimate_processing_time(video_length_seconds, num_clips)
            # Update ETA with new estimates
            task_data["eta"] = calculate_eta(task_data)

def update_task_results(task_id: str, results: dict):
    """Store results for a task"""
    with task_lock:
        logger.info(f"Storing results for task {task_id}: {results}")
        task_results[task_id] = results

@app.route('/models-status')
def models_status():
    """Check if models are loaded."""
    try:
        # Try to wait for models with a short timeout
        clip_selector.wait_for_models(timeout=0.1)
        return jsonify({'loaded': True})
    except TimeoutError:
        # Models are still loading
        return jsonify({'loaded': False})
    except Exception as e:
        # Something went wrong during loading
        return jsonify({'loaded': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload():
    """Handle video file uploads."""
    try:
        task_id = str(time.time())  # Simple task ID generation
        update_task_progress(task_id, 0, "Starting video processing...", "download")
        
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
            
        video_file = request.files['video']
        should_transcribe = request.form.get('transcribe', 'false').lower() == 'true'
        num_clips = int(request.form.get('num_clips', 3))
        quality = request.form.get('quality', 'high')
        
        if not video_file.filename:
            return jsonify({'error': 'No video file selected'}), 400
            
        # Start processing in a background thread
        def process_task():
            try:
                # Save the uploaded file
                video_path = save_uploaded_video(video_file)
                
                # Get video length and update estimates
                video_length = get_video_length(video_path)
                update_task_video_info(task_id, video_length, num_clips)
                
                # Process the video (transcribe, select clips, etc.)
                if should_transcribe:
                    # Transcription phase
                    update_task_progress(task_id, 0, "Transcribing video...", "transcribe")
                    words = transcribe_video(video_path)
                    update_task_progress(task_id, 100, "Transcription complete", "transcribe")
                    
                    # Select clips
                    try:
                        update_task_progress(task_id, 0, "Analyzing video content...", "analyze")
                        clips = clip_selector.select_clips(words, target_num_clips=num_clips)
                        update_task_progress(task_id, 100, "Analysis complete", "analyze")
                    except Exception as e:
                        logger.error(f"Error selecting clips: {str(e)}")
                        update_task_progress(task_id, 0, f"Error selecting clips: {str(e)}", "error")
                        return
                else:
                    # If not transcribing, just split into equal segments
                    clips = split_video_duration(video_length, num_clips)
                    words = None
                
                # Processing phase
                update_task_progress(task_id, 0, "Initializing video processor...", "process")
                processor = VideoProcessor(video_path)
                
                # Create clips directory
                clips_dir = os.path.join('output', 'clips')
                os.makedirs(clips_dir, exist_ok=True)
                
                # Create default caption style
                caption_style = CaptionStyle(
                    font_family='Impact',
                    font_size=72,
                    font_color='&H00FFFFFF',
                    font_bold=True,
                    border_color='&H000000000',
                    border_size=4,
                    background_color='&H00000000',
                    position='middle',
                    alignment='center',
                    margin_vertical=20,
                    margin_horizontal=100,
                    scale_x=100,
                    scale_y=100,
                    spacing=0,
                    fade_in_ms=200,
                    fade_out_ms=150,
                    min_gap_ms=50
                )
                
                update_task_progress(task_id, 30, "Extracting and processing clips...", "process")
                
                # Extract clips with captions
                extracted_clips = []
                for i, ((start, end), _) in enumerate(zip(clips, range(len(clips)))):
                    update_task_progress(
                        task_id, 
                        30 + ((i + 1) / len(clips) * 60),  # Progress from 30% to 90%
                        f"Processing clip {i + 1} of {len(clips)}...",
                        "process"
                    )
                    
                    # Generate output filename
                    base_name = os.path.splitext(os.path.basename(video_path))[0]
                    # Clean the filename to be URL-safe
                    base_name = werkzeug.utils.secure_filename(base_name)
                    clip_path = os.path.join(clips_dir, f"{base_name}_clip_{i+1}.mp4")
                    
                    # Create clip request
                    clip_request = ClipRequest(
                        start_time=start,
                        end_time=end,
                        output_path=clip_path,
                        quality=quality,
                        transcript=words,
                        caption_style=caption_style
                    )
                    
                    # Extract the clip
                    if processor.extract_clip(clip_request):
                        extracted_clips.append(clip_path)
                
                update_task_progress(task_id, 90, "Finalizing clips...", "process")
                
                # Store results
                results = {
                    'clips': [
                        {
                            'file': os.path.basename(clip_path),
                            'duration': end - start,
                            'text': " ".join(word['word'] for word in words if start <= word['start'] <= end) if words else None
                        }
                        for clip_path, (start, end) in zip(extracted_clips, clips)
                    ]
                }
                update_task_results(task_id, results)
                
                update_task_progress(task_id, 100, "Processing complete!", "process")
                
            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                update_task_progress(task_id, 0, f"Error: {str(e)}", "error")
        
        # Start the background task
        thread = threading.Thread(target=process_task)
        thread.start()
        
        return jsonify({
            'task_id': task_id
        })
        
    except Exception as e:
        logger.error(f"Error handling upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download():
    """Handle YouTube URL downloads."""
    try:
        task_id = str(time.time())  # Simple task ID generation
        update_task_progress(task_id, 0, "Starting video processing...", "download")
        
        url = request.json.get('url')
        should_transcribe = request.json.get('transcribe', False)
        language = request.json.get('language')
        num_clips = int(request.json.get('num_clips', 3))
        quality = request.json.get('quality', 'high')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
            
        def download_progress_callback(progress: float, status: str):
            update_task_progress(task_id, progress, status, "download")
            
        # Start processing in a background thread
        def process_task():
            try:
                # Download the video with progress tracking
                video_path = download_youtube_video(url, progress_callback=download_progress_callback)
                
                # Get video length and update estimates
                video_length = get_video_length(video_path)
                update_task_video_info(task_id, video_length, num_clips)
                
                # Process the video (transcribe, select clips, etc.)
                if should_transcribe:
                    # Transcription phase
                    update_task_progress(task_id, 0, "Transcribing video...", "transcribe")
                    words = transcribe_video(video_path, language)
                    update_task_progress(task_id, 100, "Transcription complete", "transcribe")
                    
                    # Select clips
                    try:
                        # Analysis phase
                        update_task_progress(task_id, 0, "Loading AI models...", "analyze")
                        clip_selector.wait_for_models(timeout=60)
                        update_task_progress(task_id, 30, "Models loaded, analyzing content...", "analyze")
                        clips = clip_selector.select_clips(words, target_num_clips=num_clips)
                        update_task_progress(task_id, 100, "Analysis complete", "analyze")
                        
                        # Processing phase
                        update_task_progress(task_id, 0, "Initializing video processor...", "process")
                        processor = VideoProcessor(video_path)
                        update_task_progress(task_id, 10, "Creating output directories...", "process")
                        
                        # Create clips directory
                        clips_dir = os.path.join('output', 'clips')
                        os.makedirs(clips_dir, exist_ok=True)
                        
                        update_task_progress(task_id, 20, "Preparing to extract clips...", "process")
                        
                        # Create default caption style
                        caption_style = CaptionStyle(
                            font_family='Impact',
                            font_size=72,
                            font_color='&H00FFFFFF',
                            font_bold=True,
                            border_color='&H000000000',
                            border_size=4,
                            background_color='&H00000000',
                            position='middle',
                            alignment='center',
                            margin_vertical=20,
                            margin_horizontal=100,
                            scale_x=100,
                            scale_y=100,
                            spacing=0,
                            fade_in_ms=200,
                            fade_out_ms=150,
                            min_gap_ms=50
                        )
                        
                        update_task_progress(task_id, 30, "Extracting and processing clips...", "process")
                        
                        # Extract clips with captions
                        extracted_clips = []
                        for i, ((start, end), _) in enumerate(zip(clips, range(len(clips)))):
                            update_task_progress(
                                task_id, 
                                30 + ((i + 1) / len(clips) * 60),  # Progress from 30% to 90%
                                f"Processing clip {i + 1} of {len(clips)}...",
                                "process"
                            )
                            
                            # Generate output filename
                            base_name = os.path.splitext(os.path.basename(video_path))[0]
                            # Clean the filename to be URL-safe
                            base_name = werkzeug.utils.secure_filename(base_name)
                            clip_path = os.path.join(clips_dir, f"{base_name}_clip_{i+1}.mp4")
                            
                            # Create clip request
                            clip_request = ClipRequest(
                                start_time=start,
                                end_time=end,
                                output_path=clip_path,
                                quality='high',
                                transcript=words,
                                caption_style=caption_style
                            )
                            
                            # Extract the clip
                            if processor.extract_clip(clip_request):
                                extracted_clips.append(clip_path)
                        
                        update_task_progress(task_id, 90, "Finalizing clips...", "process")
                        
                        # Store results
                        clip_results = {
                            'clips': [
                                {
                                    'start': start,
                                    'end': end,
                                    'duration': end - start,
                                    'text': ' '.join(
                                        word['word'] for word in words 
                                        if start <= word['start'] <= end
                                    ),
                                    'file': os.path.basename(clip_path) if clip_path else None
                                }
                                for (start, end), clip_path in zip(clips, extracted_clips)
                            ]
                        }
                        update_task_results(task_id, clip_results)
                        update_task_progress(task_id, 100, "Processing complete!", "process")
                        
                    except TimeoutError:
                        update_task_progress(task_id, 100, "Error: Models are still loading", "error")
                    except Exception as e:
                        update_task_progress(task_id, 100, f"Error: {str(e)}", "error")
                
            except Exception as e:
                update_task_progress(task_id, 100, f"Error: {str(e)}", "error")
        
        # Get the filename from the path for the initial response
        result = {
            'success': True,
            'message': 'Starting video download...',
            'task_id': task_id
        }
        
        # Start the background processing
        thread = threading.Thread(target=process_task)
        thread.start()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/output/<path:filename>')
def serve_file(filename):
    """Serve files from the output directory."""
    try:
        # Get the directory relative to output/
        directory = os.path.dirname(filename)
        basename = os.path.basename(filename)
        
        # Get the absolute path to the output directory
        output_dir = os.path.abspath('output')
        
        # If the file is in a subdirectory of output
        if directory:
            # Construct the full directory path
            full_dir = os.path.join(output_dir, directory)
            if not os.path.exists(full_dir):
                return jsonify({'error': f'Directory not found: {directory}'}), 404
            return send_from_directory(full_dir, basename)
        
        # If the file is directly in output
        return send_from_directory(output_dir, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/output/clips/<path:filename>')
def serve_clip(filename):
    """Serve generated video clips."""
    try:
        # Extract task_id from filename
        task_id = request.args.get('task_id')
        if not task_id:
            logger.error("No task_id provided for clip access")
            return jsonify({'error': 'No task ID provided'}), 400
            
        # Check ad completion
        if not verify_ad_completion(task_id):
            logger.error(f"Attempted to access clip without completing ads: {task_id}")
            return jsonify({'error': 'Please complete required advertisements first'}), 403
            
        # Get absolute path to clips directory
        clips_dir = os.path.abspath(os.path.join('output', 'clips'))
        if not os.path.exists(clips_dir):
            logger.error(f"Clips directory does not exist: {clips_dir}")
            return jsonify({'error': 'Clips directory not found'}), 404
            
        # URL decode the filename
        filename = werkzeug.utils.secure_filename(filename)
        file_path = os.path.join(clips_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Clip file not found: {file_path}")
            return jsonify({'error': 'Clip file not found'}), 404
            
        logger.info(f"Serving clip: {filename} from {file_path}")
        
        # Use absolute path with send_from_directory
        directory = os.path.dirname(file_path)
        basename = os.path.basename(file_path)
        return send_from_directory(directory, basename, as_attachment=False)
    except Exception as e:
        logger.error(f"Error serving clip {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<task_id>', methods=['GET', 'OPTIONS'])
def get_progress(task_id):
    """Get progress for a specific task"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return add_cors_headers(response)

    try:
        logger.info(f"Getting progress for task {task_id}")
        progress = task_manager.get_task(task_id)
        response = jsonify(progress)
        return add_cors_headers(response)
    except Exception as e:
        logger.error(f"Error getting progress for task {task_id}: {str(e)}")
        response = jsonify({
            'status': 'error',
            'message': f'Error getting progress: {str(e)}'
        }), 500
        return add_cors_headers(response)

@app.route('/process', methods=['POST', 'OPTIONS'])
def process_video():
    """Process video endpoint"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return add_cors_headers(response)
    
    try:
        # Create task first
        task_id = task_manager.create_task()
        logger.info(f"Created new task: {task_id}")
        
        # Get request data before starting the thread
        request_data = request.get_json() if request.is_json else request.form.to_dict()
        files = request.files if request.files else None
        
        @copy_current_request_context
        def process_in_background(task_id, data, files):
            try:
                with app.app_context():
                    logger.info(f"Starting background processing for task {task_id}")
                    task_manager.update_task(task_id, 'processing', 0, 'Starting video processing')
                    
                    # Your processing code here...
                    # Example:
                    task_manager.update_task(task_id, 'processing', 50, 'Processing video...')
                    time.sleep(2)  # Simulate work
                    task_manager.update_task(task_id, 'completed', 100, 'Processing complete')
                    
            except Exception as e:
                logger.error(f"Error in background task {task_id}: {str(e)}")
                task_manager.update_task(task_id, 'error', message=str(e))
        
        # Start the background thread with the data
        thread = threading.Thread(
            target=process_in_background,
            args=(task_id, request_data, files)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started background thread for task {task_id}")
        response = jsonify({'task_id': task_id})
        return add_cors_headers(response)
        
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        response = jsonify({
            'status': 'error',
            'message': f'Error creating task: {str(e)}'
        }), 500
        return add_cors_headers(response)

@app.route('/results/<task_id>')
def get_results(task_id):
    """Get the results of a completed task"""
    with task_lock:
        logger.info(f"Getting results for task {task_id}")
        if task_id not in task_results:
            logger.warning(f"No results found for task {task_id}")
            return jsonify({
                'error': 'Results not found'
            }), 404
            
        logger.info(f"Found results for task {task_id}: {task_results[task_id]}")
        return jsonify(task_results[task_id])

@app.route('/api/ad/complete', methods=['POST'])
def complete_ad():
    """Mark an ad as completed"""
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        ad_id = data.get('ad_id')
        
        if not task_id or task_id not in ad_states:
            return jsonify({'error': 'Invalid task ID'}), 400
            
        with ad_lock:
            state = ad_states[task_id]
            current_time = time.time()
            
            # Verify minimum time between ads (25 seconds)
            if current_time - state['last_ad_time'] < 25:
                return jsonify({'error': 'Ad completion too quick'}), 400
                
            # Update ad state
            state['ads_completed'] += 1
            state['last_ad_time'] = current_time
            
            # Check if all ads are completed
            if state['ads_completed'] >= state['ads_required']:
                state['verified'] = True
                
            return jsonify({
                'ads_completed': state['ads_completed'],
                'ads_required': state['ads_required'],
                'verified': state['verified']
            })
            
    except Exception as e:
        logger.error(f"Error completing ad: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ad/status/<task_id>')
def get_ad_status(task_id):
    """Get current ad viewing status"""
    try:
        if task_id not in ad_states:
            return jsonify({'error': 'Invalid task ID'}), 400
            
        with ad_lock:
            state = ad_states[task_id]
            return jsonify({
                'ads_completed': state['ads_completed'],
                'ads_required': state['ads_required'],
                'verified': state['verified']
            })
            
    except Exception as e:
        logger.error(f"Error getting ad status: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_video_length(video_path: str) -> float:
    """Get video length in seconds using ffmpeg"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout)
    except Exception as e:
        print(f"Error getting video length: {e}")
        return 180  # Default to 3 minutes if we can't get the length

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    response = jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'error': str(error)
    }), 500
    return add_cors_headers(response)

@app.errorhandler(404)
def not_found_error(error):
    logger.error(f"Not found error: {str(error)}")
    response = jsonify({
        'status': 'error',
        'message': 'Not found',
        'error': str(error)
    }), 404
    return add_cors_headers(response)

if __name__ == '__main__':
    app.run(debug=True) 