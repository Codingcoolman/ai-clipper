import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from dataclasses import dataclass
import threading
import queue
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

@dataclass
class ClipSegment:
    start_time: float
    end_time: float
    text: str
    importance_score: float
    topic: int
    is_complete_thought: bool = True

class ClipSelector:
    def __init__(self, 
                 min_clip_duration: float = 8.0,
                 max_clip_duration: float = 45.0,
                 min_segment_gap: float = 2.0,
                 use_llm: bool = False,
                 llm_model_name: str = "microsoft/phi-2"):  # Much smaller model, better for CPU
        """
        Initialize the clip selector.
        
        Args:
            min_clip_duration: Minimum duration of a clip in seconds
            max_clip_duration: Maximum duration of a clip in seconds
            min_segment_gap: Minimum gap between segments in seconds
            use_llm: Whether to use LLM for clip selection
            llm_model_name: Name of the HuggingFace model to use
        """
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.min_segment_gap = min_segment_gap
        self.use_llm = use_llm
        self.llm_model_name = llm_model_name
        
        # Initialize models as None
        self.sentence_model = None
        self.topic_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.models_ready = threading.Event()
        self.loading_error = None
        
        # Configure sentence tokenizer for better boundary detection
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = {'dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'etc', 'eg', 'ie'}
        self.sentence_tokenizer = PunktSentenceTokenizer(punkt_param)
        
        # Start loading models in background
        self.load_models_async()
        
    def load_models_async(self):
        """Load models in a background thread."""
        def load_models():
            try:
                logger.info("Loading NLP models in background...")
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.topic_model = BERTopic(embedding_model=self.sentence_model)
                
                if self.use_llm:
                    logger.info(f"Loading LLM model: {self.llm_model_name}")
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(
                        self.llm_model_name,
                        low_cpu_mem_usage=True
                    )
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        self.llm_model_name,
                        torch_dtype=torch.float32,  # Use full precision for CPU
                        device_map='cpu',          # Force CPU
                        low_cpu_mem_usage=True     # Optimize memory usage
                    )
                    logger.info("LLM model loaded successfully!")
                
                logger.info("All models loaded successfully!")
                self.models_ready.set()
            except Exception as e:
                self.loading_error = str(e)
                logger.error(f"Error loading models: {e}")
                self.models_ready.set()
        
        thread = threading.Thread(target=load_models, daemon=True)
        thread.start()

    def wait_for_models(self, timeout=180):  # Increased timeout to 3 minutes for first load
        """Wait for models to be loaded."""
        if not self.models_ready.wait(timeout):
            raise TimeoutError("Timeout waiting for models to load")
        if self.loading_error:
            raise RuntimeError(f"Error loading models: {self.loading_error}")

    def _is_complete_thought(self, text: str) -> bool:
        """Check if a text segment represents a complete thought."""
        # Basic checks for completeness
        if not text:
            return False
            
        try:
            # Must have both subject and predicate
            words = nltk.word_tokenize(text)
            tags = nltk.pos_tag(words)  # This uses the default English tagger
            
            has_noun = any(tag.startswith('NN') for _, tag in tags)
            has_verb = any(tag.startswith('VB') for _, tag in tags)
            
            # Check for incomplete starts
            incomplete_starts = ['and ', 'but ', 'or ', 'so ', 'because ', 'while ']
            starts_incomplete = any(text.lower().startswith(start) for start in incomplete_starts)
            
            # Check for proper ending
            ends_properly = text.strip().endswith(('.', '!', '?'))
            
            return has_noun and has_verb and ends_properly and not starts_incomplete
        except Exception as e:
            logger.warning(f"Error checking complete thought: {e}")
            # If there's an error in NLP processing, assume it's a complete thought
            # This allows the system to continue functioning even if NLP fails
            return True
        
    def _group_words_into_sentences(self, words: List[Dict]) -> List[Dict]:
        """Group word-level transcriptions into complete sentences."""
        if not words:
            return []
            
        sentences = []
        current_sentence = {
            'text': '',
            'start': words[0]['start'],
            'end': words[0]['end'],
            'words': []
        }
        
        for i, word in enumerate(words):
            # Add word to current sentence
            if current_sentence['text']:
                current_sentence['text'] += ' '
            current_sentence['text'] += word['word']
            current_sentence['end'] = word['end']
            current_sentence['words'].append(word)
            
            # Check if this is a sentence boundary
            is_boundary = False
            
            # End of punctuation
            if word['word'].endswith(('.', '!', '?')):
                is_boundary = True
            
            # Long pause (> 1.5 seconds)
            next_word = words[i + 1] if i < len(words) - 1 else None
            if next_word and (next_word['start'] - word['end']) > 1.5:
                is_boundary = True
            
            # Natural breaking points
            breaking_words = {'and', 'but', 'or', 'so', 'because', 'however'}
            if next_word and next_word['word'].lower() in breaking_words:
                is_boundary = True
            
            # Maximum length reached (30 words)
            if len(current_sentence['words']) >= 30:
                is_boundary = True
            
            if is_boundary or i == len(words) - 1:
                # Check if sentence is complete enough
                if len(current_sentence['words']) >= 3:  # Minimum 3 words
                    current_sentence['is_complete'] = True
                    sentences.append(current_sentence)
                
                # Start new sentence if not at end
                if i < len(words) - 1:
                    current_sentence = {
                        'text': '',
                        'start': next_word['start'] if next_word else word['end'],
                        'end': next_word['end'] if next_word else word['end'],
                        'words': []
                    }
        
        logger.info(f"Grouped {len(words)} words into {len(sentences)} sentences")
        return sentences
    
    def _calculate_importance_scores(self, sentences: List[Dict]) -> np.ndarray:
        """Calculate importance scores for sentences using semantic similarity and content analysis."""
        texts = [s['text'] for s in sentences]
        
        # Calculate embeddings
        embeddings = self.sentence_model.encode(texts)
        
        # Initialize importance scores
        importance_scores = np.zeros(len(sentences))
        
        # Universal content markers (not specific to any topic)
        content_markers = {
            # Teaching/instruction markers
            'how to': 1.2,
            'important': 1.2,
            'key': 1.2,
            'tip': 1.2,
            'secret': 1.2,
            'best': 1.2,
            'perfect': 1.2,
            
            # Problem/solution markers
            'mistake': 1.2,
            'wrong': 1.2,
            'instead': 1.1,
            'better': 1.1,
            
            # Result/outcome markers
            'result': 1.1,
            'finally': 1.1,
            'now': 1.0,
            'then': 1.0,
            
            # Emphasis markers
            'must': 1.2,
            'always': 1.2,
            'never': 1.2,
            'exactly': 1.2,
            
            # Demonstration markers
            'look': 1.1,
            'see': 1.1,
            'watch': 1.1,
            'notice': 1.1,
            'here': 1.0
        }
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            text_lower = text.lower()
            
            # 1. Content Marker Score (25%)
            marker_score = 1.0
            for marker, boost in content_markers.items():
                if marker in text_lower:
                    marker_score *= boost
            
            # 2. Semantic Coherence Score (25%)
            # How well this segment connects with its neighbors
            semantic_score = 0.0
            neighbor_count = 0
            
            if i > 0:
                prev_emb = embeddings[i-1]
                semantic_score += np.dot(embedding, prev_emb) / (np.linalg.norm(embedding) * np.linalg.norm(prev_emb))
                neighbor_count += 1
                
            if i < len(embeddings) - 1:
                next_emb = embeddings[i+1]
                semantic_score += np.dot(embedding, next_emb) / (np.linalg.norm(embedding) * np.linalg.norm(next_emb))
                neighbor_count += 1
            
            if neighbor_count > 0:
                semantic_score /= neighbor_count
            
            # 3. Structural Score (25%)
            structural_score = 1.0
            
            # Complete sentence bonus
            if text.strip().endswith(('.', '!', '?')):
                structural_score *= 1.2
            
            # Length penalty
            words = text.split()
            if len(words) < 5:  # Too short
                structural_score *= 0.5
            elif len(words) > 30:  # Too long
                structural_score *= 0.7
            
            # 4. Engagement Score (25%)
            engagement_score = 1.0
            
            # Direct address
            if 'you' in text_lower:
                engagement_score *= 1.1
            
            # Action words
            action_words = ['make', 'do', 'try', 'get', 'put', 'take', 'use', 'keep', 'let']
            if any(word in text_lower for word in action_words):
                engagement_score *= 1.1
            
            # Questions or commands
            if '?' in text or text.strip().endswith('!'):
                engagement_score *= 1.1
            
            # Combine scores
            importance_scores[i] = (
                marker_score * 0.25 +      # Content markers
                semantic_score * 0.25 +     # Semantic coherence
                structural_score * 0.25 +   # Structural completeness
                engagement_score * 0.25     # Engagement features
            )
        
        # Normalize scores
        if len(importance_scores) > 0:
            importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-10)
        
        logger.info(f"Calculated importance scores ranging from {importance_scores.min():.2f} to {importance_scores.max():.2f}")
        return importance_scores
    
    def _merge_nearby_segments(self, segments: List[ClipSegment]) -> List[ClipSegment]:
        """Merge nearby segments into coherent thoughts."""
        if not segments:
            return []
            
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            gap = next_segment.start_time - current.end_time
            merged_duration = next_segment.end_time - current.start_time
            
            # Check if merging would create a more complete thought
            merged_text = current.text + " " + next_segment.text
            
            # Only merge if:
            # 1. The gap is small enough
            # 2. The resulting duration is acceptable
            # 3. The segments are semantically related
            # 4. The combination makes a complete thought
            if (gap <= self.min_segment_gap * 3 and  # Increased gap tolerance
                merged_duration <= self.max_clip_duration and
                len(merged_text.split()) <= 100):  # Increased word limit
                
                # Calculate semantic similarity between segments
                embeddings = self.sentence_model.encode([current.text, next_segment.text])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                
                # More lenient similarity threshold for longer segments
                min_similarity = 0.5 if merged_duration < 20 else 0.4
                
                # Only merge if segments are semantically related
                if similarity > min_similarity:
                    current = ClipSegment(
                        start_time=current.start_time,
                        end_time=next_segment.end_time,
                        text=merged_text,
                        importance_score=max(current.importance_score, next_segment.importance_score),
                        topic=current.topic,
                        is_complete_thought=True
                    )
                    continue
            
            merged.append(current)
            current = next_segment
        
        merged.append(current)
        
        # Additional pass to merge very short segments with their neighbors
        if len(merged) > 1:
            final_merged = []
            current = merged[0]
            
            for next_segment in merged[1:]:
                duration = current.end_time - current.start_time
                if duration < self.min_clip_duration:
                    # Try to merge with next segment
                    merged_duration = next_segment.end_time - current.start_time
                    if merged_duration <= self.max_clip_duration:
                        current = ClipSegment(
                            start_time=current.start_time,
                            end_time=next_segment.end_time,
                            text=current.text + " " + next_segment.text,
                            importance_score=max(current.importance_score, next_segment.importance_score),
                            topic=current.topic,
                            is_complete_thought=True
                        )
                        continue
                
                final_merged.append(current)
                current = next_segment
            
            final_merged.append(current)
            merged = final_merged
        
        logger.info(f"Merged {len(segments)} segments into {len(merged)} complete thoughts")
        return merged
    
    def _get_llm_clip_suggestions(self, transcript_text: str, target_num_clips: int) -> List[Dict]:
        """Use LLM to suggest viral-worthy clips from the transcript."""
        if not self.llm_model or not self.llm_tokenizer:
            logger.warning("LLM model not loaded, skipping LLM suggestions")
            return []
            
        # Truncate transcript if too long to save processing time
        max_chars = 6000  # About 1000 words
        if len(transcript_text) > max_chars:
            words = transcript_text.split()
            transcript_text = " ".join(words[:500]) + "\n...\n" + " ".join(words[-500:])
            logger.info("Truncated long transcript for LLM processing")
            
        prompt = f"""Analyze this transcript and identify exactly {target_num_clips} segments that would make great viral short-form video clips.

Focus on moments that are:
1. Emotionally impactful or thought-provoking
2. Self-contained and make sense without context
3. Relatable and shareable
4. Between 5-30 seconds when spoken

For each clip, provide:
1. The exact text to include
2. A score from 1-10 indicating viral potential
3. A brief explanation of why it would be viral

Format your response exactly like this example:
---
Clip 1:
Text: "exact text to use in the clip"
Score: 8
Reason: brief explanation of viral potential
---

Transcript:
{transcript_text}"""

        try:
            # Use shorter maximum lengths for faster processing
            inputs = self.llm_tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024  # Reduced from 2048
            ).to(self.llm_model.device)
            
            outputs = self.llm_model.generate(
                **inputs,
                max_length=512,    # Reduced from 1024
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                do_sample=True,    # Enable sampling for faster generation
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse LLM response
            suggestions = []
            clip_sections = response.split('---')[1:-1]  # Split by --- and remove empty first/last sections
            
            for section in clip_sections:
                try:
                    # Extract text between 'Text:' and 'Score:'
                    text_start = section.find('Text:') + 5
                    text_end = section.find('Score:')
                    text = section[text_start:text_end].strip().strip('"')
                    
                    # Extract score
                    score_start = section.find('Score:') + 6
                    score_end = section.find('Reason:')
                    score = float(section[score_start:score_end].strip())
                    
                    # Extract reason
                    reason = section[score_end + 7:].strip()
                    
                    suggestions.append({
                        'text': text,
                        'viral_score': score / 10.0,  # Normalize to 0-1
                        'reason': reason
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse clip suggestion: {e}")
                    continue
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting LLM suggestions: {e}")
            return []

    def _boost_scores_with_llm_suggestions(self, segments: List[ClipSegment], llm_suggestions: List[Dict]) -> None:
        """Boost importance scores based on LLM suggestions."""
        if not llm_suggestions:
            return
            
        # For each LLM suggestion, find matching segments and boost their scores
        for suggestion in llm_suggestions:
            suggestion_text = suggestion['text'].lower()
            viral_score = suggestion['viral_score']
            
            # Find segments that contain this text
            for segment in segments:
                segment_text = segment.text.lower()
                
                # Calculate text similarity ratio
                # If the suggestion is fully contained in the segment or vice versa
                if suggestion_text in segment_text or segment_text in suggestion_text:
                    # Boost the segment's score
                    # We use a weighted average between current score and viral score
                    segment.importance_score = (
                        segment.importance_score * 0.6 +  # Keep 60% of original score
                        viral_score * 0.4                 # Add 40% influence from LLM
                    )
                    logger.info(f"Boosted segment score to {segment.importance_score:.2f} based on LLM suggestion")
                    logger.info(f"Reason: {suggestion['reason']}")

    def select_clips(self, words: List[Dict], target_num_clips: int = 3) -> List[Tuple[float, float]]:
        """Select the most important complete-thought clips from the transcript."""
        try:
            # First, ensure models are loaded
            logger.info("Waiting for NLP models to be ready...")
            self.wait_for_models(timeout=60)  # Wait up to 60 seconds
            
            logger.info(f"Selecting approximately {target_num_clips} clips...")
            
            if not words:
                logger.warning("No words provided in transcript")
                return []
                
            # Group words into sentences
            sentences = self._group_words_into_sentences(words)
            if not sentences:
                logger.warning("No sentences found in transcript")
                return []
            
            # Calculate importance scores
            importance_scores = self._calculate_importance_scores(sentences)
            
            # Create initial segments
            segments = [
                ClipSegment(
                    start_time=sent['start'],
                    end_time=sent['end'],
                    text=sent['text'],
                    importance_score=score,
                    topic=0,  # We'll set this later
                    is_complete_thought=sent['is_complete']
                )
                for sent, score in zip(sentences, importance_scores)
            ]
            
            # Filter for complete thoughts first
            complete_segments = [seg for seg in segments if seg.is_complete_thought]
            
            if not complete_segments:
                logger.warning("No complete thoughts found, falling back to all segments")
                complete_segments = segments
            
            # Sort by importance score
            complete_segments.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Take top candidates (2x target for merging flexibility)
            candidates = complete_segments[:int(target_num_clips * 3)]  # Increased from 2x to 3x
            
            # Sort by time and merge nearby segments
            candidates.sort(key=lambda x: x.start_time)
            merged_segments = self._merge_nearby_segments(candidates)
            
            # Select final clips with length requirements
            merged_segments.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Initialize final clips list
            selected_segments = []
            
            # Process segments to ensure good length distribution
            for i in range(target_num_clips):
                remaining_segments = [s for s in merged_segments 
                                   if s not in selected_segments]
                
                if not remaining_segments:
                    break
                    
                # For the last clip, prioritize longer segments
                if i == target_num_clips - 1:
                    # Filter for segments of reasonable length
                    long_segments = [s for s in remaining_segments 
                                   if (s.end_time - s.start_time) >= self.min_clip_duration * 1.5]
                    
                    if long_segments:
                        # Choose the most important long segment
                        selected_segments.append(max(long_segments, 
                            key=lambda x: x.importance_score))
                    else:
                        # If no long segments, try to merge adjacent segments
                        remaining_segments.sort(key=lambda x: x.start_time)
                        for j in range(len(remaining_segments) - 1):
                            current = remaining_segments[j]
                            next_seg = remaining_segments[j + 1]
                            merged_duration = next_seg.end_time - current.start_time
                            
                            if (merged_duration <= self.max_clip_duration and 
                                merged_duration >= self.min_clip_duration * 1.5):
                                merged = ClipSegment(
                                    start_time=current.start_time,
                                    end_time=next_seg.end_time,
                                    text=current.text + " " + next_seg.text,
                                    importance_score=max(current.importance_score, 
                                                      next_seg.importance_score),
                                    topic=current.topic,
                                    is_complete_thought=True
                                )
                                selected_segments.append(merged)
                                break
                        else:
                            # If we couldn't merge, just take the most important remaining
                            selected_segments.append(max(remaining_segments, 
                                key=lambda x: x.importance_score))
                else:
                    # For other clips, balance importance and length
                    scored_segments = [
                        (s, s.importance_score * min(1.0, 
                            (s.end_time - s.start_time) / self.min_clip_duration))
                        for s in remaining_segments
                    ]
                    selected_segments.append(max(scored_segments, key=lambda x: x[1])[0])
            
            # Sort by time for output
            selected_segments.sort(key=lambda x: x.start_time)
            
            # Convert to time tuples
            clips = [(seg.start_time, seg.end_time) for seg in selected_segments]
            
            return clips
            
        except TimeoutError:
            logger.error("Timeout waiting for models to load")
            return []
        except Exception as e:
            logger.error(f"Error selecting clips: {e}")
            return []

    def test_with_sample(self) -> List[Tuple[float, float, str]]:
        """Test function using a sample transcript for quick algorithm testing."""
        # Sample transcript with timestamps and known key points
        sample_words = [
            {"word": "I", "start": 0.0, "end": 0.1},
            {"word": "want", "start": 0.1, "end": 0.2},
            {"word": "to", "start": 0.2, "end": 0.3},
            {"word": "share", "start": 0.3, "end": 0.5},
            {"word": "something", "start": 0.5, "end": 0.7},
            {"word": "important", "start": 0.7, "end": 1.0},
            {"word": "with", "start": 1.0, "end": 1.1},
            {"word": "you", "start": 1.1, "end": 1.2},
            {"word": "today.", "start": 1.2, "end": 1.5},
            
            {"word": "The", "start": 5.0, "end": 5.1},
            {"word": "first", "start": 5.1, "end": 5.3},
            {"word": "key", "start": 5.3, "end": 5.5},
            {"word": "point", "start": 5.5, "end": 5.7},
            {"word": "is", "start": 5.7, "end": 5.8},
            {"word": "that", "start": 5.8, "end": 5.9},
            {"word": "time", "start": 5.9, "end": 6.1},
            {"word": "is", "start": 6.1, "end": 6.2},
            {"word": "precious.", "start": 6.2, "end": 6.5},
            
            {"word": "We", "start": 10.0, "end": 10.1},
            {"word": "often", "start": 10.1, "end": 10.3},
            {"word": "forget", "start": 10.3, "end": 10.5},
            {"word": "this", "start": 10.5, "end": 10.7},
            {"word": "simple", "start": 10.7, "end": 10.9},
            {"word": "truth.", "start": 10.9, "end": 11.2},
            
            {"word": "Therefore,", "start": 15.0, "end": 15.3},
            {"word": "we", "start": 15.3, "end": 15.4},
            {"word": "must", "start": 15.4, "end": 15.6},
            {"word": "make", "start": 15.6, "end": 15.8},
            {"word": "every", "start": 15.8, "end": 16.0},
            {"word": "moment", "start": 16.0, "end": 16.2},
            {"word": "count.", "start": 16.2, "end": 16.5},
            
            {"word": "The", "start": 20.0, "end": 20.1},
            {"word": "second", "start": 20.1, "end": 20.3},
            {"word": "important", "start": 20.3, "end": 20.6},
            {"word": "lesson", "start": 20.6, "end": 20.8},
            {"word": "is", "start": 20.8, "end": 20.9},
            {"word": "about", "start": 20.9, "end": 21.1},
            {"word": "growth.", "start": 21.1, "end": 21.5},
            
            {"word": "Finally,", "start": 25.0, "end": 25.3},
            {"word": "remember", "start": 25.3, "end": 25.6},
            {"word": "that", "start": 25.6, "end": 25.7},
            {"word": "success", "start": 25.7, "end": 26.0},
            {"word": "takes", "start": 26.0, "end": 26.2},
            {"word": "time", "start": 26.2, "end": 26.4},
            {"word": "and", "start": 26.4, "end": 26.5},
            {"word": "patience.", "start": 26.5, "end": 27.0}
        ]
        
        # Select clips
        clips = self.select_clips(sample_words, target_num_clips=3)
        
        # Return clips with their text for easier evaluation
        result = []
        for start, end in clips:
            text = " ".join(
                word["word"] for word in sample_words 
                if start <= word["start"] <= end
            )
            result.append((start, end, text))
            
        return result 