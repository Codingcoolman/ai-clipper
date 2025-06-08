from src.clip_selector import ClipSelector
from test_transcript import get_full_transcript

def main():
    # Initialize the clip selector
    selector = ClipSelector()
    
    # Get the test transcript
    words = get_full_transcript()
    
    print("\nTesting clip selection with real transcript about mortality...")
    
    # Run the algorithm
    clips = selector.select_clips(words, target_num_clips=3)
    
    # Print results
    print("\nSelected Clips:")
    for i, (start, end) in enumerate(clips, 1):
        # Get the text for this clip
        clip_text = " ".join(
            word["word"] for word in words 
            if start <= word["start"] <= end
        )
        print(f"\nClip {i}:")
        print(f"Time: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
        print(f"Text: {clip_text}")

if __name__ == "__main__":
    main() 