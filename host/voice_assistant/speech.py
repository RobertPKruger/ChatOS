# voice_assistant/speech.py - FIXED VERSION with acknowledgement exceptions
"""
Speech-to-text and text-to-speech processing with improved validation
"""

import asyncio
import io
import logging
import os
import sys
import time
from typing import Optional
import sounddevice as sd
import soundfile as sf

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except:
    PYGAME_AVAILABLE = False

from .state import AssistantState, AssistantMode
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)

# Valid acknowledgements and short responses that should always be accepted
VALID_ACKNOWLEDGEMENTS = {
    # Basic responses
    "yes", "yeah", "yep", "yup", "yes sir", "yes ma'am", "absolutely", "definitely", 
    "sure", "ok", "okay", "alright", "all right", "right", "correct", "true",
    "no", "nope", "nah", "no way", "not really", "negative", "false", "wrong",
    
    # Politeness
    "please", "thank you", "thanks", "thank you very much", "much appreciated",
    "you're welcome", "no problem", "no worries", "my pleasure", "anytime",
    "sorry", "excuse me", "pardon me", "my apologies", "i apologize",
    
    # Confirmations
    "i agree", "agreed", "sounds good", "that works", "perfect", "excellent",
    "great", "awesome", "wonderful", "fantastic", "amazing", "brilliant",
    "i understand", "understood", "got it", "i see", "makes sense", "i get it",
    
    # Directions and commands
    "stop", "wait", "hold on", "pause", "continue", "go ahead", "proceed",
    "start", "begin", "end", "finish", "done", "complete", "next", "previous",
    "up", "down", "left", "right", "back", "forward", "here", "there",
    
    # Questions
    "what", "where", "when", "why", "how", "who", "which", "whose",
    "what's that", "what is that", "who's that", "who is that",
    "how much", "how many", "how long", "how far", "how often",
    
    # Greetings
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "good night", "goodbye", "bye", "see you later", "talk to you later",
    
    # Help and assistance
    "help", "help me", "i need help", "can you help", "assist me",
    "what can you do", "what do you do", "how do you work",
    
    # Emotional responses
    "wow", "cool", "nice", "good", "bad", "terrible", "horrible", "lovely",
    "interesting", "boring", "exciting", "scary", "funny", "weird", "strange",
    
    # Numbers and basic words
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "last", "final", "initial", "original",
    
    # Actions
    "show me", "tell me", "give me", "find", "search", "look", "check",
    "open", "close", "start", "stop", "play", "pause", "skip", "repeat",
    
    # Common short phrases
    "i think", "i believe", "i know", "i don't know", "i'm not sure",
    "maybe", "perhaps", "possibly", "probably", "definitely not",
    "of course", "certainly", "obviously", "clearly", "exactly",
    
    # Technical/computer terms
    "computer", "laptop", "phone", "tablet", "screen", "monitor", "keyboard",
    "mouse", "internet", "wifi", "bluetooth", "usb", "file", "folder",
    
    # Time references
    "now", "later", "soon", "today", "tomorrow", "yesterday", "tonight",
    "morning", "afternoon", "evening", "night", "minute", "hour", "day"
}

# Convert to lowercase set for faster lookup
VALID_ACKNOWLEDGEMENTS_SET = {phrase.lower() for phrase in VALID_ACKNOWLEDGEMENTS}

async def transcribe_audio(audio_buffer: io.BytesIO, state: AssistantState, config, check_stuck_phrase: bool = False) -> Optional[str]:
    """Transcribe audio with retry logic and improved validation"""
    async def do_transcribe():
        try:
            # Force English language hint
            response = state.openai_client.audio.transcriptions.create(
                model=config.stt_model,
                file=audio_buffer,
                language="en"  # Force English transcription
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"STT error: {e}")
            raise
    
    try:
        text = await retry_with_backoff(do_transcribe(), config.max_retries)
        
        # If checking for stuck phrase, do minimal validation
        if check_stuck_phrase:
            return text
        
        # Validate transcription with acknowledgement exceptions
        if not text or len(text.strip()) == 0:
            logger.info("Empty transcription, ignoring")
            return None
        
        # Clean and normalize the text
        clean_text = text.strip().lower()
        normalized_text = clean_text.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
        
        # FIRST: Check if it's a valid acknowledgement or short response
        if normalized_text in VALID_ACKNOWLEDGEMENTS_SET:
            logger.info(f"Valid acknowledgement detected: {text}")
            return text
        
        # SECOND: Check if any words in the text are valid acknowledgements
        words = normalized_text.split()
        if len(words) <= 3:  # Short phrases only
            for word in words:
                if word in VALID_ACKNOWLEDGEMENTS_SET:
                    logger.info(f"Contains valid acknowledgement word: {text}")
                    return text
        
        # THIRD: Check for short but meaningful combinations
        if len(words) <= 2:
            # Allow combinations like "yes please", "no thanks", "ok sure", etc.
            word_set = set(words)
            if any(word in VALID_ACKNOWLEDGEMENTS_SET for word in word_set):
                logger.info(f"Short meaningful phrase: {text}")
                return text
        
        # Enhanced noise patterns - common false positives
        noise_patterns = [
            # Common STT artifacts
            "Thank you.", "Thanks for watching!", "Bye!", "Thanks.", 
            "Thank you for watching.", "Please subscribe.", "Bye-bye",
            
            # Music/media artifacts  
            "[Music]", "[Applause]", "[Laughter]", "[MUSIC]", "[music]",
            "♪", "♫", "♪♪", "♬", "♩",
            
            # Foreign language false positives
            "음악", "音楽", "嗯", "啊", "呃", "哦", "是", "的",
            "ありがとう", "こんにちは", "さようなら",
            "Merci", "Bonjour", "Au revoir", "Gracias", "Hola",
            
            # Single characters/punctuation
            "the", "a", ".", ",", "!", "?", "-", "...",
            
            # Common background noise transcriptions
            "Shh", "Shh.", "Shhh", "Hmm", "Hmm.", "Hm",
            "background noise", "inaudible", "[inaudible]",
            
            # Media playback artifacts
            "playing", "music playing", "video playing",
            "Transcribed by", "Subtitles by", "Captions by"
        ]
        
        # Check exact matches (case-insensitive) - but skip if it's a valid acknowledgement
        if clean_text in [p.lower() for p in noise_patterns]:
            # Double-check it's not actually a valid response
            if clean_text not in VALID_ACKNOWLEDGEMENTS_SET:
                logger.info(f"Ignoring noise transcription: {text}")
                return None
        
        # Check if it's mostly punctuation or special characters
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars < len(text) * 0.5 and len(words) > 1:  # Less than 50% letters and not a single word
            logger.info(f"Ignoring non-text transcription: {text}")
            return None
        
        # Check minimum word count - but be more lenient for acknowledged words
        if len(words) < config.min_confidence_length:
            # If it's very short, check if any word is meaningful
            if len(words) == 1 and words[0] not in VALID_ACKNOWLEDGEMENTS_SET:
                logger.info(f"Single word not in acknowledgements: {text}")
                return None
            elif len(words) == 2:
                # For two words, at least one should be meaningful
                if not any(word in VALID_ACKNOWLEDGEMENTS_SET for word in words):
                    logger.info(f"Two words, neither meaningful: {text}")
                    return None
        
        # Check for repeated characters (often indicates noise) - but allow short responses
        if len(set(text.replace(" ", ""))) < 3 and len(text) > 5:
            logger.info(f"Ignoring repetitive transcription: {text}")
            return None
        
        # Check for non-English characters (basic check) - but be more lenient for short text
        non_ascii_chars = sum(1 for c in text if ord(c) > 127)
        if non_ascii_chars > len(text) * 0.5 and len(text) > 10:  # More than 50% non-ASCII in longer text
            logger.info(f"Ignoring non-English transcription: {text}")
            return None
        
        # Language detection using simple heuristics - but skip for acknowledged phrases
        if normalized_text not in VALID_ACKNOWLEDGEMENTS_SET:
            common_english_words = {"the", "is", "are", "and", "or", "but", "in", "on", "at", 
                                   "to", "for", "of", "with", "as", "by", "that", "this",
                                   "what", "how", "when", "where", "why", "who", "which",
                                   "can", "will", "would", "should", "could", "have", "has", 
                                   "launch", "open", "start", "play", "run", "create", "delete", "edit", "save"}
            
            words_lower = [w.lower() for w in words]
            english_word_count = sum(1 for w in words_lower if w in common_english_words)
            acknowledgement_count = sum(1 for w in words_lower if w in VALID_ACKNOWLEDGEMENTS_SET)
            
            # Require at least one common English word OR acknowledgement for longer phrases
            if len(words) >= 3 and english_word_count == 0 and acknowledgement_count == 0:
                logger.info(f"No common English or acknowledgement words found: {text}")
                return None
        
        logger.info(f"Valid transcription: {text[:100]}...")
        return text
        
    except Exception as e:
        logger.error(f"Failed to transcribe after retries: {e}")
        return None

async def speak_text(text: str, state: AssistantState, config) -> bool:
    """Convert text to speech and play it with interruption support"""
    if not text or text.isspace():
        text = "Okay."
    
    # Set speaking mode
    state.set_mode(AssistantMode.SPEAKING)
    state.interrupt_flag.clear()
    
    try:
        logger.info(f"Speaking: {text[:50]}...")
        
        # Generate speech
        loop = asyncio.get_event_loop()
        audio_response = await loop.run_in_executor(
            None, 
            lambda: state.openai_client.audio.speech.create(
                model=config.tts_model,
                voice=config.tts_voice,
                input=text,
                response_format="wav"
            )
        )
        
        # Check for interruption before playing
        if state.interrupt_flag.is_set():
            logger.info("Speech interrupted before playback")
            state.set_mode(AssistantMode.LISTENING)
            return False
        
        # Try different audio playback methods
        audio_data = audio_response.read()
        success = False
        
        # Method 1: Try sounddevice first
        try:
            def play_with_sounddevice():
                audio_buffer = io.BytesIO(audio_data)
                data, sample_rate = sf.read(audio_buffer, dtype="float32")
                
                # Try to use default output device
                try:
                    sd.default.device = None  # Reset to default
                    sd.play(data, sample_rate)
                    sd.wait()
                    return True
                except sd.PortAudioError:
                    # Try alternative device
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        if device['max_output_channels'] > 0:
                            try:
                                sd.default.device = i
                                sd.play(data, sample_rate)
                                sd.wait()
                                return True
                            except:
                                continue
                    return False
            
            success = await loop.run_in_executor(None, play_with_sounddevice)
            
        except Exception as e:
            logger.warning(f"Sounddevice playback failed: {e}")
        
        # Method 2: Fallback to pygame if available
        if not success and PYGAME_AVAILABLE:
            try:
                def play_with_pygame():
                    # Save to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(audio_data)
                        tmp_path = tmp_file.name
                    
                    # Play with pygame
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    
                    # Wait for completion with interruption check
                    while pygame.mixer.music.get_busy():
                        if state.interrupt_flag.is_set():
                            pygame.mixer.music.stop()
                            os.unlink(tmp_path)
                            return False
                        time.sleep(0.1)
                    
                    os.unlink(tmp_path)
                    return True
                
                success = await loop.run_in_executor(None, play_with_pygame)
                if success:
                    logger.info("Audio played using pygame fallback")
                    
            except Exception as e:
                logger.warning(f"Pygame playback failed: {e}")
        
        # Method 3: System command fallback (Windows)
        if not success and sys.platform == "win32":
            try:
                import tempfile
                import winsound
                
                def play_with_winsound():
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(audio_data)
                        tmp_path = tmp_file.name
                    
                    winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
                    os.unlink(tmp_path)
                    return True
                
                success = await loop.run_in_executor(None, play_with_winsound)
                if success:
                    logger.info("Audio played using winsound fallback")
                    
            except Exception as e:
                logger.warning(f"Winsound playback failed: {e}")
        
        if not success:
            logger.error("All audio playback methods failed")
            state.set_mode(AssistantMode.LISTENING)
            return False
            
        logger.info("Audio playback completed")
        
        # Return to listening mode
        state.set_mode(AssistantMode.LISTENING)
        return True
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        state.set_mode(AssistantMode.LISTENING)
        return False