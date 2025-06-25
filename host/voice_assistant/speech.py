# voice_assistant/speech.py
"""
Speech-to-text and text-to-speech processing
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

async def transcribe_audio(audio_buffer: io.BytesIO, state: AssistantState, config, check_stuck_phrase: bool = False) -> Optional[str]:
    """Transcribe audio with retry logic and validation"""
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
        
        # Validate transcription
        if not text or len(text.strip()) == 0:
            logger.info("Empty transcription, ignoring")
            return None
        
        # Enhanced noise patterns - common false positives
        noise_patterns = [
            # Common STT artifacts
            "Thank you.", "Thanks for watching!", "Bye!", "Thanks.", 
            "Thank you for watching.", "Please subscribe.", "Bye-bye",
            "Hello?", "Yeah.", "Uh-huh.", "Mm-hmm.", "Okay.", "Alright.",
            
            # Music/media artifacts  
            "[Music]", "[Applause]", "[Laughter]", "[MUSIC]", "[music]",
            "♪", "♫", "♪♪", "♬", "♩",
            
            # Foreign language false positives
            "음악", "音楽", "嗯", "啊", "呃", "哦", "是", "的",
            "ありがとう", "こんにちは", "さようなら",
            "Merci", "Bonjour", "Au revoir", "Gracias", "Hola",
            
            # Single characters/punctuation
            "you", "the", "a", ".", ",", "!", "?", "-", "...",
            
            # Common background noise transcriptions
            "Shh", "Shh.", "Shhh", "Hmm", "Hmm.", "Hm",
            "background noise", "inaudible", "[inaudible]",
            
            # Media playback artifacts
            "playing", "music playing", "video playing",
            "Transcribed by", "Subtitles by", "Captions by"
        ]
        
        # Check exact matches (case-insensitive)
        if text.strip().lower() in [p.lower() for p in noise_patterns]:
            logger.info(f"Ignoring noise transcription: {text}")
            return None
        
        # Check if it's mostly punctuation or special characters
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars < len(text) * 0.5:  # Less than 50% letters
            logger.info(f"Ignoring non-text transcription: {text}")
            return None
        
        # Check minimum word count
        words = [w for w in text.split() if w.strip() and any(c.isalpha() for c in w)]
        if len(words) < config.min_confidence_length:
            logger.info(f"Transcription too short ({len(words)} words): {text}")
            return None
        
        # Check for repeated characters (often indicates noise)
        if len(set(text.replace(" ", ""))) < 3:
            logger.info(f"Ignoring repetitive transcription: {text}")
            return None
        
        # Check for non-English characters (basic check)
        non_ascii_chars = sum(1 for c in text if ord(c) > 127)
        if non_ascii_chars > len(text) * 0.3:  # More than 30% non-ASCII
            logger.info(f"Ignoring non-English transcription: {text}")
            return None
        
        # Language detection using simple heuristics
        common_english_words = {"the", "is", "are", "and", "or", "but", "in", "on", "at", 
                               "to", "for", "of", "with", "as", "by", "that", "this",
                               "what", "how", "when", "where", "why", "who", "which",
                               "can", "will", "would", "should", "could", "have", "has", 
                               "launch", "open", "start", "play", "run", "create", "delete", "edit", "save"}
        
        words_lower = [w.lower() for w in words]
        english_word_count = sum(1 for w in words_lower if w in common_english_words)
        
        # Require at least one common English word for short phrases
        if len(words) < 10 and english_word_count == 0:
            logger.info(f"No common English words found in short phrase: {text}")
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