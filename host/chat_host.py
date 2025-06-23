
# enhanced_chat_host_v2.py
"""
Enhanced MCP Chat Host - Always-on voice assistant with interruption support
New features:
- Continuous listening (no push-to-talk)
- Voice Activity Detection (VAD) support
- Interrupt assistant while speaking
- Proper shutdown of both server and client
- Improved audio handling
"""

import asyncio
import json
import subprocess
import queue
import threading
import sys
import io
import time
import os
import uuid
import logging
import signal
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
import numpy as np

import sounddevice as sd
import soundfile as sf
import pygame  # Fallback audio player
from openai import OpenAI
from fastmcp import Client
from dotenv import load_dotenv
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    logger.warning("webrtcvad not available, falling back to energy-based detection")
    VAD_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class Config:
    """Configuration settings for the voice assistant"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    stt_model: str = "gpt-4o-transcribe"
    chat_model: str = "gpt-4o"
    tts_model: str = "tts-1"
    tts_voice: str = "nova"
    sample_rate: int = 16000
    min_audio_size: int = 10000
    tool_timeout: float = 30.0
    reconnect_delay: float = 2.0
    max_retries: int = 3
    log_level: str = "INFO"
    log_file: str = "voice_assistant.log"
    enable_vad: bool = True
    vad_aggressiveness: int = 3  # 0-3, higher = more aggressive (increased from 2)
    silence_threshold: float = 0.03  # Energy threshold for silence detection (increased)
    silence_duration: float = 1.5  # Seconds of silence before processing
    min_speech_duration: float = 0.8  # Minimum speech duration in seconds (increased)
    energy_threshold_multiplier: float = 2.0  # Dynamic threshold multiplier (increased)
    min_confidence_length: int = 2  # Minimum words for valid transcription
    max_energy_threshold: float = 0.5  # Maximum energy threshold to prevent oversensitivity
    english_confidence_threshold: float = 0.7  # Confidence threshold for language detection
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            stt_model=os.getenv("STT_MODEL", "gpt-4o-transcribe"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o"),
            tts_model=os.getenv("TTS_MODEL", "tts-1"),
            tts_voice=os.getenv("TTS_VOICE", "nova"),
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            min_audio_size=int(os.getenv("MIN_AUDIO_SIZE", "10000")),
            tool_timeout=float(os.getenv("TOOL_TIMEOUT", "30.0")),
            reconnect_delay=float(os.getenv("RECONNECT_DELAY", "2.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "voice_assistant.log"),
            enable_vad=os.getenv("ENABLE_VAD", "true").lower() == "true",
            vad_aggressiveness=int(os.getenv("VAD_AGGRESSIVENESS", "3")),
            silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.03")),
            silence_duration=float(os.getenv("SILENCE_DURATION", "1.5")),
            min_speech_duration=float(os.getenv("MIN_SPEECH_DURATION", "0.8")),
            energy_threshold_multiplier=float(os.getenv("ENERGY_THRESHOLD_MULTIPLIER", "2.0")),
            min_confidence_length=int(os.getenv("MIN_CONFIDENCE_LENGTH", "3")),
            max_energy_threshold=float(os.getenv("MAX_ENERGY_THRESHOLD", "0.5")),
            english_confidence_threshold=float(os.getenv("ENGLISH_CONFIDENCE_THRESHOLD", "0.7"))
        )

# Global configuration
config = Config.from_env()

# Initialize pygame mixer as fallback
try:
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except:
    PYGAME_AVAILABLE = False
    logger.warning("pygame not available for audio fallback")

# Setup logging
def setup_logging():
    """Configure logging with rotation and proper formatting"""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

# Global state
class AssistantState:
    """Global state management for the assistant"""
    def __init__(self):
        self.running = True
        self.muted = False
        self.is_speaking = False
        self.interrupt_flag = threading.Event()
        self.openai_client: Optional[OpenAI] = None
        self.mcp_client: Optional[Client] = None
        self.mcp_process: Optional[subprocess.Popen] = None
        self.tools_cache: List[Dict[str, Any]] = []
        self.tools_cache_time: float = 0
        self.conversation_history: List[Dict[str, Any]] = []
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.vad = webrtcvad.Vad(config.vad_aggressiveness) if (config.enable_vad and VAD_AVAILABLE) else None
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("Conversation history reset")

# Global state instance
state = AssistantState()

# System prompt
SYSTEM_PROMPT = """
You are my personal voice assistant. Keep responses conversational and natural, but concise.

When the user asks to open an application like "Open Notepad" or "Launch Calculator", use the launch_app tool and provide the appropriate app name as a parameter. For example:
- For "Open Notepad" → use launch_app with app="notepad"
- For "Open Calculator" → use launch_app with app="calc"
- For "Open File Explorer" → use launch_app with app="explorer"

Steam Game Commands:
- For "Open Steam" or "Launch Steam" → use open_steam tool
- For "Open Steam store/library/community" → use open_steam with the appropriate page parameter
- For "Play [game name]" or "Launch [game name]" → use launch_steam_game with game_name parameter
- For "What games do I have?" → use list_steam_games
- Common game examples:
  - "Play Counter-Strike" → launch_steam_game(game_name="Counter-Strike 2")
  - "Launch Dota" → launch_steam_game(game_name="Dota 2")
  - "Open Team Fortress 2" → launch_steam_game(game_name="Team Fortress 2")
  - "Play game 730" → launch_steam_game(app_id="730")

Always provide the appropriate parameters when using tools.

If you encounter any errors with tools, explain what went wrong and suggest alternatives when possible.

File-system examples
--------------------
- "Create a folder called Projects inside Documents"
  → create_folder(path="~/Documents/Projects")
- "Create a folder named Projects on my desktop"
  → create_folder(path="~/Desktop/Projects")
- "Open my desktop folder"
  → open_folder(path="~/Desktop")

When the user asks to shut down, exit, or quit, acknowledge and prepare to shut down the system.
"""

################################################################################
# Audio recording with continuous listening
################################################################################

class ContinuousAudioRecorder:
    """Manages continuous audio recording with VAD and interruption support"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.recording = False
        self.stream = None
        self._lock = threading.Lock()
        
        # List available audio devices for debugging
        try:
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                logger.info(f"  {i}: {device['name']} - In:{device['max_input_channels']} Out:{device['max_output_channels']}")
        except Exception as e:
            logger.warning(f"Could not query audio devices: {e}")
        
    def start(self):
        """Start continuous recording"""
        with self._lock:
            if self.recording:
                return
                
            self.recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.03),  # 30ms chunks for VAD
                dtype="int16",
                channels=1,
                callback=self._audio_callback
            )
            self.stream.start()
            logger.info("Started continuous audio recording")
    
    def stop(self):
        """Stop recording"""
        with self._lock:
            self.recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            logger.info("Stopped audio recording")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.recording:
            self.audio_queue.put(bytes(indata))
    
    def get_audio_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy of audio data"""
        return np.sqrt(np.mean(audio_data.astype(float)**2))
    
    async def record_until_silence(self) -> Optional[io.BytesIO]:
        """Record audio until silence is detected"""
        logger.info("Listening for speech...")
        
        frames = []
        silence_frames = 0
        speech_frames = 0
        speech_started = False
        required_silence_frames = int(config.silence_duration * self.sample_rate / (self.sample_rate * 0.03))
        min_speech_frames = int(config.min_speech_duration * self.sample_rate / (self.sample_rate * 0.03))
        
        # Dynamic noise floor estimation
        noise_samples = []
        noise_floor = config.silence_threshold
        
        # Track peak energy to detect actual speech
        peak_energy = 0
        
        while self.recording and not state.interrupt_flag.is_set():
            try:
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                energy = self.get_audio_energy(audio_data)
                
                # Update peak energy
                if energy > peak_energy:
                    peak_energy = energy
                
                # Update noise floor during initial silence
                if not speech_started and len(noise_samples) < 50:  # ~1.5 seconds of samples
                    noise_samples.append(energy)
                    if len(noise_samples) >= 20:
                        # Calculate noise floor with some headroom
                        noise_floor = np.percentile(noise_samples, 95) * config.energy_threshold_multiplier
                        noise_floor = max(noise_floor, config.silence_threshold)
                        noise_floor = min(noise_floor, config.max_energy_threshold)
                
                # Check if someone is speaking (to interrupt assistant)
                if state.is_speaking and energy > noise_floor * 1.5:
                    logger.info("User interruption detected")
                    state.interrupt_flag.set()
                    return None
                
                # Use VAD if enabled and available
                if config.enable_vad and state.vad and VAD_AVAILABLE:
                    # VAD requires specific frame size
                    frame_duration = 30  # ms
                    required_samples = int(self.sample_rate * frame_duration / 1000)
                    
                    if len(audio_data) == required_samples:
                        is_speech = state.vad.is_speech(audio_chunk, self.sample_rate)
                        # Also require energy threshold for VAD
                        is_speech = is_speech and energy > noise_floor
                    else:
                        is_speech = energy > noise_floor
                else:
                    is_speech = energy > noise_floor
                
                if is_speech:
                    if not speech_started:
                        # Require sustained energy above threshold
                        if energy > noise_floor * 1.5:  # Higher threshold for speech start
                            logger.info(f"Speech detected (energy: {energy:.4f}, threshold: {noise_floor:.4f})")
                            speech_started = True
                            frames.append(audio_chunk)
                            speech_frames += 1
                    else:
                        frames.append(audio_chunk)
                        speech_frames += 1
                    silence_frames = 0
                elif speech_started:
                    frames.append(audio_chunk)
                    silence_frames += 1
                    
                    if silence_frames >= required_silence_frames:
                        # Check if we have enough speech frames
                        if speech_frames >= min_speech_frames:
                            # Also check if peak energy was significant
                            if peak_energy > noise_floor * 2:
                                logger.info(f"Silence detected after {speech_frames} speech frames (peak energy: {peak_energy:.4f})")
                                break
                            else:
                                logger.info(f"Ignoring low-energy speech (peak: {peak_energy:.4f})")
                                frames = []
                                speech_frames = 0
                                speech_started = False
                                silence_frames = 0
                                peak_energy = 0
                        else:
                            logger.info(f"Ignoring short noise burst ({speech_frames} frames)")
                            frames = []
                            speech_frames = 0
                            speech_started = False
                            silence_frames = 0
                            peak_energy = 0
                        
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            except Exception as e:
                logger.error(f"Error in audio recording: {e}")
                break
        
        if not frames or speech_frames < min_speech_frames:
            return None
        
        # Final peak energy check
        if peak_energy < noise_floor * 1.5:
            logger.info(f"Rejecting low-energy audio (peak: {peak_energy:.4f})")
            return None
        
        # Convert frames to WAV
        wav_buffer = io.BytesIO()
        try:
            with sf.SoundFile(
                wav_buffer,
                mode="w",
                samplerate=self.sample_rate,
                channels=1,
                subtype="PCM_16",
                format="WAV"
            ) as sound_file:
                for frame in frames:
                    sound_file.buffer_write(frame, dtype="int16")
        except Exception as e:
            logger.error(f"Error creating WAV file: {e}")
            return None
        
        wav_buffer.seek(0)
        wav_buffer.name = "speech.wav"
        return wav_buffer

# Global audio recorder
audio_recorder = ContinuousAudioRecorder(config.sample_rate)

################################################################################
# MCP Client management with subprocess control
################################################################################

@asynccontextmanager
async def get_mcp_client():
    """Context manager for MCP client with subprocess management"""
    server_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mcp_os", "server.py")
    )
    
    client = None
    try:
        # Start the MCP server as a subprocess if not already running
        if not state.mcp_process or state.mcp_process.poll() is not None:
            logger.info(f"Starting MCP server: {server_path}")
            state.mcp_process = subprocess.Popen(
                [sys.executable, server_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Give the server time to start
            await asyncio.sleep(1)
        
        client = Client(server_path)
        async with client:
            logger.info("MCP client connected successfully")
            yield client
    except Exception as e:
        logger.error(f"MCP client error: {e}")
        if client:
            try:
                await client.close()
            except:
                pass
        raise

async def get_tools(mcp_client: Client, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Get tools from MCP server with caching"""
    current_time = time.time()
    
    # Use cache if available and recent
    if (use_cache and 
        state.tools_cache and 
        current_time - state.tools_cache_time < 60):  # 1 minute cache
        return state.tools_cache
    
    tools = []
    try:
        for tool in await mcp_client.list_tools():
            try:
                # Try to get OpenAI schema
                if hasattr(tool, 'openai_schema'):
                    spec = tool.openai_schema()
                else:
                    # Fallback schema construction
                    spec = {
                        "name": tool.name,
                        "description": getattr(tool, "description", ""),
                        "parameters": getattr(tool, "inputSchema", {
                            "type": "object", 
                            "properties": {}
                        })
                    }
                
                # Ensure required fields
                if "name" not in spec:
                    spec["name"] = tool.name
                if "description" not in spec:
                    spec["description"] = getattr(tool, "description", "")
                
                tools.append({"type": "function", "function": spec})
                logger.debug(f"Added tool: {tool.name}")
                
            except Exception as e:
                logger.warning(f"Failed to process tool {tool.name}: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        # Return cached tools if available
        if state.tools_cache:
            logger.info("Using cached tools due to listing failure")
            return state.tools_cache
        raise
    
    # Update cache
    state.tools_cache = tools
    state.tools_cache_time = current_time
    logger.info(f"Loaded {len(tools)} tools")
    
    return tools

################################################################################
# Tool execution with timeout and error handling
################################################################################

async def call_tool_with_timeout(mcp_client: Client, call) -> str:
    """Execute a tool call with timeout and error handling"""
    # Normalize call format
    if hasattr(call, "function"):
        name = call.function.name
        args_json = call.function.arguments or "{}"
        call_id = getattr(call, "id", str(uuid.uuid4()))
    else:
        name = call.get("name") or call["function"]["name"]
        args_json = call.get("function", call).get("arguments", "{}")
        call_id = call.get("id", str(uuid.uuid4()))
    
    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in tool arguments: {e}"
        logger.error(error_msg)
        return error_msg
    
    logger.info(f"Calling tool: {name} with args: {args}")
    
    async def execute_tool():
        try:
            response = await mcp_client.call_tool(name, args)
            
            # Handle different response types
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "content"):
                if isinstance(response.content, list):
                    text_parts = []
                    for content in response.content:
                        if hasattr(content, "text"):
                            text_parts.append(content.text)
                        elif hasattr(content, "content"):
                            text_parts.append(str(content.content))
                        else:
                            text_parts.append(str(content))
                    return " ".join(text_parts)
                else:
                    return str(response.content)
            else:
                return str(response)
                
        except Exception as e:
            error_msg = f"Tool {name} failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    try:
        # Execute with timeout
        result = await asyncio.wait_for(execute_tool(), timeout=config.tool_timeout)
        logger.info(f"Tool {name} completed successfully")
        return result
    except asyncio.TimeoutError:
        error_msg = f"Tool {name} timed out after {config.tool_timeout}s"
        logger.error(error_msg)
        return error_msg

################################################################################
# STT with retry logic
################################################################################

async def transcribe_audio(audio_buffer: io.BytesIO) -> Optional[str]:
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
        text = await retry_with_backoff(do_transcribe())
        
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
                               "launch", "open", "start", "play", "run", "create", "delete", "edit", "save"   }
        
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

################################################################################
# TTS with interruption support
################################################################################

async def speak_text(text: str) -> bool:
    """Convert text to speech and play it with interruption support"""
    if not text or text.isspace():
        text = "Okay."
    
    # Clear any previous interrupt flag
    state.interrupt_flag.clear()
    state.is_speaking = True
    
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
            return False
            
        logger.info("Audio playback completed")
        return True
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False
    finally:
        state.is_speaking = False

################################################################################
# Error handling and retry utilities
################################################################################

async def retry_with_backoff(coro, max_retries: int = None, base_delay: float = 1.0):
    """Retry a coroutine with exponential backoff"""
    if max_retries is None:
        max_retries = config.max_retries
        
    for attempt in range(max_retries + 1):
        try:
            return await coro
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Final retry attempt failed: {e}")
                raise
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)

################################################################################
# Shutdown utilities
################################################################################

async def shutdown_system():
    """Properly shut down both server and client"""
    logger.info("Initiating system shutdown...")
    
    # Stop audio recording
    audio_recorder.stop()
    
    # Close MCP client
    if state.mcp_client:
        try:
            await state.mcp_client.close()
        except Exception as e:
            logger.error(f"Error closing MCP client: {e}")
    
    # Terminate MCP server subprocess
    if state.mcp_process:
        try:
            logger.info("Terminating MCP server process...")
            state.mcp_process.terminate()
            
            # Wait for graceful shutdown
            try:
                state.mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("MCP server didn't terminate gracefully, forcing kill...")
                state.mcp_process.kill()
                state.mcp_process.wait()
            
            logger.info("MCP server process terminated")
        except Exception as e:
            logger.error(f"Error terminating MCP server: {e}")
    
    state.running = False
    logger.info("System shutdown complete")

################################################################################
# Main conversation loop
################################################################################

async def conversation_loop():
    """Main conversation loop with continuous listening"""
    logger.info("Starting conversation loop with continuous listening")
    
    # Initialize conversation
    state.reset_conversation()
    
    # Initialize OpenAI client
    if not state.openai_client:
        try:
            state.openai_client = OpenAI(api_key=config.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    # Start continuous audio recording
    audio_recorder.start()
    
    async with get_mcp_client() as mcp_client:
        state.mcp_client = mcp_client
        
        # Load tools
        tools = await get_tools(mcp_client)
        
        # Initial greeting
        await speak_text("Hello! I'm listening. You can speak naturally, and I'll respond when you pause.")
        
        while state.running:
            try:
                if state.muted:
                    await asyncio.sleep(1)
                    continue
                
                # Record audio until silence
                audio_buffer = await audio_recorder.record_until_silence()
                
                if not audio_buffer:
                    continue
                
                # Transcribe
                user_text = await transcribe_audio(audio_buffer)
                if not user_text:
                    continue
                
                logger.info(f"User said: {user_text}")
                
                # Handle special commands
                lower_text = user_text.lower().strip()
                
                if lower_text in {"reset chat", "new chat", "clear history"}:
                    state.reset_conversation()
                    await speak_text("Starting a new conversation.")
                    continue
                
                if lower_text in {"mute", "quiet", "stop listening"}:
                    state.muted = True
                    await speak_text("I'm muted. Say 'unmute' to resume.")
                    continue
                
                if lower_text in {"unmute", "resume", "wake up"}:
                    state.muted = False
                    await speak_text("I'm listening again.")
                    continue
                
                if any(phrase in lower_text for phrase in ["exit", "quit", "goodbye", "shut down", "shutdown"]):
                    await speak_text("Goodbye! Shutting down the system...")
                    await shutdown_system()
                    break
                
                # Add user message to history
                state.conversation_history.append({"role": "user", "content": user_text})
                
                # Get AI response
                try:
                    completion = state.openai_client.chat.completions.create(
                        model=config.chat_model,
                        messages=state.conversation_history,
                        tools=tools,
                        tool_choice="auto"
                    )
                    
                    choice = completion.choices[0]
                    message = choice.message
                    
                    assistant_response = ""
                    
                    if choice.finish_reason == "tool_calls" and message.tool_calls:
                        # Handle tool calls
                        state.conversation_history.append({
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": message.tool_calls
                        })
                        
                        # Execute each tool call
                        for tool_call in message.tool_calls:
                            tool_result = await call_tool_with_timeout(mcp_client, tool_call)
                            
                            # Add tool result to history
                            state.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                        
                        # Get the model's follow-up response
                        follow_up = state.openai_client.chat.completions.create(
                            model=config.chat_model,
                            messages=state.conversation_history,
                            tools=tools,
                            tool_choice="auto"
                        )
                        
                        follow_up_message = follow_up.choices[0].message
                        assistant_response = follow_up_message.content or "Task completed."
                        
                        state.conversation_history.append({
                            "role": "assistant", 
                            "content": assistant_response
                        })
                    else:
                        # Regular text response
                        assistant_response = message.content or "I'm not sure how to respond to that."
                        state.conversation_history.append({
                            "role": "assistant", 
                            "content": assistant_response
                        })
                    
                    # Speak the response (with interruption support)
                    await speak_text(assistant_response)
                    
                except Exception as e:
                    logger.error(f"Error in chat completion: {e}")
                    error_response = "I encountered an error processing your request. Please try again."
                    await speak_text(error_response)
                    
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                await asyncio.sleep(config.reconnect_delay)
                continue

################################################################################
# Main application with restart capability
################################################################################

async def run_forever():
    """Run the assistant with automatic restart on failure"""
    restart_count = 0
    
    while state.running:
        try:
            logger.info(f"Starting voice assistant (restart #{restart_count})")
            await conversation_loop()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            await shutdown_system()
            break
            
        except Exception as e:
            restart_count += 1
            logger.exception(f"Fatal error in conversation loop (restart #{restart_count}): {e}")
            
            if restart_count > 10:  # Prevent infinite restart loops
                logger.error("Too many restarts, giving up...")
                break
                
            logger.info(f"Restarting in {config.reconnect_delay} seconds...")
            await asyncio.sleep(config.reconnect_delay)

def signal_handler(signum, frame):
    """Handle Ctrl-C / SIGTERM gracefully."""
    logger.info(f"Received signal {signum}; shutting down…")
    state.running = False
    loop = asyncio.get_event_loop()
    loop.call_soon_threadsafe(loop.stop)

def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Voice Assistant starting up...")
    logger.info(f"Configuration: STT={config.stt_model}, Chat={config.chat_model}, TTS={config.tts_model}")
    logger.info("Mode: Always listening (continuous)")
    
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        logger.info("\nShutdown complete")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    
    logger.info("Voice Assistant shutdown complete")

if __name__ == "__main__":
    main()