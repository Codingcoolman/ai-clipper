from src.clip_selector import ClipSelector
from test_transcript import get_full_transcript, get_cooking_transcript, get_gaming_transcript
import time

def test_clip_selection(transcript_words: list, description: str):
    print(f"\nTesting clip selection on {description}...")
    
    # Time the initialization
    start_time = time.time()
    selector = ClipSelector(use_llm=False)  # Using statistical only
    
    # Time the clip selection
    selection_start = time.time()
    clips = selector.select_clips(transcript_words, target_num_clips=3)
    selection_end = time.time()
    
    # Print timing info
    init_time = selection_start - start_time
    selection_time = selection_end - selection_start
    print(f"\nTiming:")
    print(f"Initialization: {init_time:.1f} seconds")
    print(f"Clip Selection: {selection_time:.1f} seconds")
    print(f"Total Time: {selection_end - start_time:.1f} seconds")
    
    # Print selected clips
    print("\nSelected Clips:")
    for i, (start, end) in enumerate(clips, 1):
        # Get the text for this clip
        clip_text = " ".join(
            word["word"] for word in transcript_words 
            if start <= word["start"] <= end
        )
        print(f"\nClip {i}:")
        print(f"Time: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
        print(f"Text: {clip_text}")
    
    return clips

if __name__ == "__main__":
    # Test with philosophical content
    print("\n=== Testing with Philosophical Content (Life/Death) ===")
    philosophical_clips = test_clip_selection(
        get_full_transcript(),
        "philosophical content"
    )
    
    # Test with cooking content
    print("\n=== Testing with Cooking Tutorial Content ===")
    cooking_clips = test_clip_selection(
        get_cooking_transcript(),
        "cooking tutorial"
    )
    
    # Test with gaming discussion
    print("\n=== Testing with Gaming Discussion Content ===")
    gaming_clips = test_clip_selection(
        get_gaming_transcript(),
        "gaming discussion"
    ) 