import faster_whisper
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_video(video_path: str, language: str = None) -> list:
    """
    Transcribe a video file using faster-whisper and return word-level timestamps.
    
    Args:
        video_path (str): Path to the video file
        language (str, optional): Language code (e.g., 'en', 'es'). If None, will auto-detect
        
    Returns:
        list: List of dictionaries containing word-level transcriptions with timestamps
    """
    try:
        logger.info(f"Starting transcription of {video_path}")
        
        # Use CUDA if available, with optimized compute type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"Using device: {device} with compute type: {compute_type}")
        
        # Load the faster-whisper model with optimized settings
        model = faster_whisper.WhisperModel(
            "base",
            device=device,
            compute_type=compute_type,
            cpu_threads=8,  # Optimize CPU threading
            num_workers=4,  # Enable parallel processing
            download_root=None,  # Use default cache
        )
        
        # Transcribe with optimized settings
        segments, _ = model.transcribe(
            video_path,
            language=language,
            word_timestamps=True,
            beam_size=5,  # Faster beam search
            best_of=2,    # Reduced candidates for speed
            condition_on_previous_text=False,  # Faster processing
            initial_prompt=None,
            temperature=0.0,  # Deterministic output
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        
        # Extract word-level information
        words = []
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    words.append({
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                        "confidence": 1.0  # faster-whisper doesn't provide word-level confidence
                    })
        
        logger.info(f"Transcription completed. Found {len(words)} words.")
        return words
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise 