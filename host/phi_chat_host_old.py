# enhanced_chat_host_v4.py
"""
Enhanced MCP Chat Host - Version 4 with Local Model Integration
New features:
- Local model routing for system operations
- Hybrid local/cloud processing
- Smart fallback to cloud on local model failures
- Improved response times for common operations
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
from typing import Optional, Dict, Any, List, Literal, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum
import numpy as np

import sounddevice as sd
import soundfile as sf
import pygame  # Fallback audio player
from openai import OpenAI
from fastmcp import Client
from dotenv import load_dotenv

# Local model imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available, local model features disabled")

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    logging.warning("webrtcvad not available, falling back to energy-based detection")

# Load environment variables
load_dotenv()

# Assistant States
class AssistantMode(Enum):
    LISTENING = "listening"          # Actively listening for commands
    RECORDING = "recording"          # Recording user speech
    PROCESSING = "processing"        # Processing with LLM
    LOCAL_PROCESSING = "local_processing"  # Processing with local model
    SPEAKING = "speaking"            # Playing TTS response
    STUCK_CHECK = "stuck_check"      # Listening only for wake phrase
    ERROR = "error"                  # Error state

# Configuration
@dataclass
class Config:
    """Configuration settings for the voice assistant"""
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    stt_model: str = "gpt-4o-transcribe"
    chat_model: str = "gpt-4o"
    tts_model: str = "tts-1"
    tts_voice: str = "nova"
    
    # Local Model Configuration
    local_model: str = "mistral-small:22b-instruct-2409-q4_0"
    use_local_first: bool = True
    local_timeout: float = 10.0
    local_max_tokens: int = 512
    local_confidence_threshold: float = 0.7
    
    # Audio Configuration
    sample_rate: int = 16000
    min_audio_size: int = 10000
    
    # Processing Configuration
    tool_timeout: float = 30.0
    processing_timeout: float = 60.0
    stuck_phrase: str = "hello abraxas are you stuck"
    stuck_check_interval: float = 5.0
    reconnect_delay: float = 2.0
    max_retries: int = 3
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "voice_assistant.log"
    
    # VAD Configuration
    enable_vad: bool = True
    vad_aggressiveness: int = 3
    silence_threshold: float = 0.03
    silence_duration: float = 1.5
    min_speech_duration: float = 0.8
    energy_threshold_multiplier: float = 2.0
    min_confidence_length: int = 2
    max_energy_threshold: float = 0.5
    english_confidence_threshold: float = 0.7
    
    # Tool Classification Keywords
    tool_call_keywords: List[str] = field(default_factory=lambda: [
        "open", "launch", "start", "run", "create", "delete", "edit", 
        "save", "close", "play", "stop", "pause", "file", "folder",
        "steam", "game", "application", "app", "calculator", "notepad",
        "explorer", "browser", "terminal", "command", "execute", "make"
    ])
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            stt_model=os.getenv("STT_MODEL", "gpt-4o-transcribe"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o"),
            tts_model=os.getenv("TTS_MODEL", "tts-1"),
            tts_voice=os.getenv("TTS_VOICE", "nova"),
            local_model=os.getenv("LOCAL_MODEL", "mistral-small:22b-instruct-2409-q4_0"),
            use_local_first=os.getenv("USE_LOCAL_FIRST", "true").lower() == "true",
            local_timeout=float(os.getenv("LOCAL_TIMEOUT", "10.0")),
            local_max_tokens=int(os.getenv("LOCAL_MAX_TOKENS", "512")),
            local_confidence_threshold=float(os.getenv("LOCAL_CONFIDENCE_THRESHOLD", "0.7")),
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            min_audio_size=int(os.getenv("MIN_AUDIO_SIZE", "10000")),
            tool_timeout=float(os.getenv("TOOL_TIMEOUT", "30.0")),
            processing_timeout=float(os.getenv("PROCESSING_TIMEOUT", "60.0")),
            stuck_phrase=os.getenv("STUCK_PHRASE", "hello abraxas are you stuck").lower().replace(",", "").replace("?", ""),
            stuck_check_interval=float(os.getenv("STUCK_CHECK_INTERVAL", "5.0")),
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

# Setup logging
def setup_logging():
    """Configure logging with rotation and proper formatting"""
    file_handler = logging.FileHandler(config.log_file, encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    if sys.platform == "win32":
        import locale
        import codecs
        if sys.stdout.encoding != 'utf-8':
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
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
        self.mode = AssistantMode.LISTENING
        self.mode_lock = threading.Lock()
        self.processing_start_time: Optional[float] = None
        self.interrupt_flag = threading.Event()
        self.processing_request = False  # NEW: Flag to prevent overlapping requests
        self.openai_client: Optional[OpenAI] = None
        self.mcp_client: Optional[Client] = None
        self.mcp_process: Optional[subprocess.Popen] = None
        self.tools_cache: List[Dict[str, Any]] = []
        self.tools_cache_time: float = 0
        self.conversation_history: List[Dict[str, Any]] = []
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.vad = webrtcvad.Vad(config.vad_aggressiveness) if (config.enable_vad and VAD_AVAILABLE) else None
        
    def set_mode(self, new_mode: AssistantMode):
        """Thread-safe mode setter with logging"""
        with self.mode_lock:
            old_mode = self.mode
            self.mode = new_mode
            logger.info(f"Mode transition: {old_mode.value} -> {new_mode.value}")
            
            # Handle interruptions during speaking
            if old_mode == AssistantMode.SPEAKING and new_mode == AssistantMode.RECORDING:
                logger.info("🔇 Interrupting speech for new input")
                self.interrupt_flag.set()
            
            if new_mode in [AssistantMode.PROCESSING, AssistantMode.LOCAL_PROCESSING]:
                self.processing_start_time = time.time()
                self.processing_request = True  # Block new requests
            elif old_mode in [AssistantMode.PROCESSING, AssistantMode.LOCAL_PROCESSING]:
                self.processing_start_time = None
                self.processing_request = False  # Allow new requests
    
    def get_mode(self) -> AssistantMode:
        """Thread-safe mode getter"""
        with self.mode_lock:
            return self.mode
    
    def is_processing(self) -> bool:
        """Check if currently processing a request"""
        with self.mode_lock:
            return self.processing_request
    
    def is_stuck(self) -> bool:
        """Check if we've been processing too long"""
        if self.processing_start_time and self.get_mode() in [AssistantMode.PROCESSING, AssistantMode.LOCAL_PROCESSING]:
            return (time.time() - self.processing_start_time) > config.processing_timeout
        return False
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("Conversation history reset")

# Global state instance
state = AssistantState()

# Model Router for Local/Cloud Decisions
class ModelRouter:
    """Routes queries between local and cloud models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.local_available = self._check_local_model()
        
    def _check_local_model(self) -> bool:
        """Check if local model is available"""
        if not OLLAMA_AVAILABLE:
            logger.info("Ollama package not available")
            return False
            
        try:
            models_response = ollama.list()
            if 'models' in models_response and models_response['models']:
                available_models = [model.get('name', '') for model in models_response['models']]
                logger.info(f"Available models from Ollama: {available_models}")
                
                # Check if we have the expected model name
                if self.config.local_model in available_models:
                    logger.info(f"Found expected model: {self.config.local_model}")
                    return True
                
                # Handle 'unnamed' models or other available models
                if available_models:
                    if available_models[0] == 'unnamed' or available_models[0] == '':
                        # Try to use the expected model name anyway (common with phi3:mini)
                        logger.info(f"Found unnamed model, will try using expected name '{self.config.local_model}'")
                        try:
                            # Test if we can actually generate with the expected name
                            test_response = ollama.generate(
                                model=self.config.local_model,
                                prompt='test',
                                options={'num_predict': 1}
                            )
                            logger.info(f"✅ Local model '{self.config.local_model}' works despite showing as unnamed")
                            return True
                        except Exception as e:
                            logger.warning(f"Expected model name '{self.config.local_model}' doesn't work: {e}")
                            return False
                    else:
                        # Use the first available model as fallback
                        logger.info(f"Expected model '{self.config.local_model}' not found, using '{available_models[0]}'")
                        self.config.local_model = available_models[0]
                        return True
                        
            logger.warning("No models found in Ollama")
            return False
            
        except Exception as e:
            logger.warning(f"Local model check failed: {e}")
            return False
    
    def should_use_local(self, user_text: str) -> bool:
        """Determine if query should use local model first"""
        if not self.config.use_local_first or not self.local_available:
            return False
            
        text_lower = user_text.lower()
        
        # Check for tool-related keywords
        keyword_matches = sum(1 for keyword in self.config.tool_call_keywords 
                             if keyword in text_lower)
        
        # Strong indicators for local processing
        local_indicators = [
            keyword_matches >= 1,  # Contains tool keywords
            len(user_text.split()) <= 15,  # Short, likely action-oriented
            any(phrase in text_lower for phrase in [
                "open", "launch", "start", "create", "delete", "run",
                "play", "stop", "close", "save", "make"
            ])
        ]
        
        return any(local_indicators)

# Initialize router
router = ModelRouter(config)

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
# Audio recording with mode-aware listening (unchanged)
################################################################################

class ContinuousAudioRecorder:
    """Manages continuous audio recording with mode awareness"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.recording = False
        self.stream = None
        self._lock = threading.Lock()
        self.currently_recording_speech = False  # NEW: Track if we're actively recording speech
        
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
                blocksize=int(self.sample_rate * 0.03),
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
        """Audio stream callback - only queue audio when appropriate"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Only queue audio if we're recording and not in certain blocking states
        if self.recording:
            current_mode = state.get_mode()
            # Allow queuing during LISTENING and RECORDING modes, block during PROCESSING/SPEAKING
            if current_mode in [AssistantMode.LISTENING, AssistantMode.RECORDING, AssistantMode.STUCK_CHECK]:
                self.audio_queue.put(bytes(indata))
    
    def clear_audio_queue(self):
        """Clear any pending audio data"""
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass
    
    def get_audio_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy of audio data"""
        return np.sqrt(np.mean(audio_data.astype(float)**2))
    
    async def record_until_silence(self, check_stuck_phrase: bool = False) -> Optional[io.BytesIO]:
        """Record audio until silence is detected, optionally checking for stuck phrase"""
        current_mode = state.get_mode()
        
        # Don't record if we're processing a request (unless checking for stuck phrase)
        if not check_stuck_phrase:
            if current_mode not in [AssistantMode.LISTENING, AssistantMode.STUCK_CHECK]:
                return None
            if state.is_processing():
                logger.debug("Audio recording blocked - currently processing a request")
                return None
        
        # Clear any pending audio data before starting
        self.clear_audio_queue()
        
        try:
            logger.info(f"Listening for {'stuck phrase' if check_stuck_phrase else 'speech'} in mode: {current_mode.value}")
            
            frames = []
            silence_frames = 0
            speech_frames = 0
            speech_started = False
            required_silence_frames = int(config.silence_duration * self.sample_rate / (self.sample_rate * 0.03))
            min_speech_frames = int(config.min_speech_duration * self.sample_rate / (self.sample_rate * 0.03))
            
            if check_stuck_phrase:
                required_silence_frames = int(0.5 * self.sample_rate / (self.sample_rate * 0.03))
                min_speech_frames = int(0.3 * self.sample_rate / (self.sample_rate * 0.03))
            
            noise_samples = []
            noise_floor = config.silence_threshold
            peak_energy = 0
            
            while self.recording and not state.interrupt_flag.is_set():
                try:
                    # Check if we should stop recording (but allow RECORDING mode)
                    if not check_stuck_phrase:
                        current_mode = state.get_mode()
                        if current_mode not in [AssistantMode.LISTENING, AssistantMode.RECORDING, AssistantMode.STUCK_CHECK]:
                            logger.debug(f"Mode changed to {current_mode.value}, stopping recording")
                            return None
                    
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                    energy = self.get_audio_energy(audio_data)
                    
                    if energy > peak_energy:
                        peak_energy = energy
                    
                    if not speech_started and len(noise_samples) < 50:
                        noise_samples.append(energy)
                        if len(noise_samples) >= 20:
                            noise_floor = np.percentile(noise_samples, 95) * config.energy_threshold_multiplier
                            noise_floor = max(noise_floor, config.silence_threshold)
                            noise_floor = min(noise_floor, config.max_energy_threshold)
                    
                    if config.enable_vad and state.vad and VAD_AVAILABLE:
                        frame_duration = 30
                        required_samples = int(self.sample_rate * frame_duration / 1000)
                        
                        if len(audio_data) == required_samples:
                            is_speech = state.vad.is_speech(audio_chunk, self.sample_rate)
                            is_speech = is_speech and energy > noise_floor
                        else:
                            is_speech = energy > noise_floor
                    else:
                        is_speech = energy > noise_floor
                    
                    if is_speech:
                        if not speech_started:
                            if energy > noise_floor * 1.5:
                                logger.debug(f"Speech detected (energy: {energy:.4f}, threshold: {noise_floor:.4f})")
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
                            if speech_frames >= min_speech_frames:
                                if peak_energy > noise_floor * 2:
                                    logger.debug(f"Silence detected after {speech_frames} speech frames (peak energy: {peak_energy:.4f})")
                                    break
                                else:
                                    logger.debug(f"Ignoring low-energy speech (peak: {peak_energy:.4f})")
                                    frames = []
                                    speech_frames = 0
                                    speech_started = False
                                    silence_frames = 0
                                    peak_energy = 0
                            else:
                                logger.debug(f"Ignoring short noise burst ({speech_frames} frames)")
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
            
            if peak_energy < noise_floor * 1.5:
                logger.debug(f"Rejecting low-energy audio (peak: {peak_energy:.4f})")
                return None
            
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
            
        finally:
            # Clear any remaining audio data
            self.clear_audio_queue()

# Global audio recorder
audio_recorder = ContinuousAudioRecorder(config.sample_rate)

################################################################################
# Local Model Processing
################################################################################

async def process_with_local_model(user_text: str, tools: List[Dict], mcp_client) -> tuple[bool, str, list]:
    """
    Process query with local model
    Returns: (success, response, tool_calls)
    """
    if not OLLAMA_AVAILABLE or not router.local_available:
        return False, "", []
    
    state.set_mode(AssistantMode.LOCAL_PROCESSING)
    
    try:
        # Create detailed tool information for the local model
        tool_descriptions = []
        for tool in tools:
            tool_spec = tool["function"]
            name = tool_spec["name"]
            description = tool_spec.get("description", "")
            
            # Extract parameter information
            params = tool_spec.get("parameters", {}).get("properties", {})
            param_info = []
            for param_name, param_spec in params.items():
                param_type = param_spec.get("type", "string")
                param_desc = param_spec.get("description", "")
                param_info.append(f'"{param_name}" ({param_type}): {param_desc}')
            
            tool_descriptions.append(f"{name}: {description}\nParameters: {', '.join(param_info) if param_info else 'none'}")
        
        local_system_prompt = f"""You are a helpful assistant that performs system operations using tools.

Available tools and their exact parameters:
{chr(10).join(tool_descriptions)}

IMPORTANT: Use the EXACT parameter names shown above.

For user requests, respond in this exact format:

If you can handle it with a tool:
TOOL: tool_name
ARGS: {{"exact_param_name": "value"}}
RESPONSE: Brief confirmation message

If the request is too complex or you can't handle it:
ESCALATE: Brief reason

Examples:
User: "Open notepad"
TOOL: launch_app
ARGS: {{"app_name": "notepad"}}
RESPONSE: Opening Notepad for you.

User: "What's the weather like?"
ESCALATE: Weather queries require internet access

Keep responses concise and action-oriented. Always use the exact parameter names from the tool definitions above."""
        
        # Prepare the prompt
        prompt = f"{local_system_prompt}\n\nUser: {user_text}\nAssistant:"
        
        logger.debug(f"Local model processing: {user_text}")
        
        # Generate response with local model
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.generate(
                    model=config.local_model,
                    prompt=prompt,
                    options={
                        'num_predict': config.local_max_tokens,
                        'temperature': 0.1,
                        'top_p': 0.9
                    }
                )
            ),
            timeout=config.local_timeout
        )
        
        response_text = response['response'].strip()
        logger.debug(f"Local model response: {response_text}")
        
        # Check if model wants to escalate
        if response_text.startswith("ESCALATE"):
            reason = response_text.split(":", 1)[1].strip() if ":" in response_text else "Complex query"
            logger.info(f"🔄 Local model escalated: {reason}")
            return False, "", []
        
        # Parse tool calls from local model response
        if "TOOL:" in response_text:
            lines = response_text.split('\n')
            tool_name = None
            tool_args = {}
            response_message = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("TOOL:"):
                    tool_name = line.split(":", 1)[1].strip()
                elif line.startswith("ARGS:"):
                    try:
                        args_str = line.split(":", 1)[1].strip()
                        tool_args = json.loads(args_str)
                        logger.debug(f"Parsed tool args: {tool_args}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool args: {args_str} - Error: {e}")
                        tool_args = {}
                elif line.startswith("RESPONSE:"):
                    response_message = line.split(":", 1)[1].strip()
            
            # Validate tool exists
            tool_names = [tool["function"]["name"] for tool in tools]
            if tool_name and tool_name in tool_names:
                # Validate parameters against tool schema
                tool_spec = next((tool["function"] for tool in tools if tool["function"]["name"] == tool_name), None)
                if tool_spec:
                    required_params = tool_spec.get("parameters", {}).get("required", [])
                    provided_params = set(tool_args.keys())
                    required_params_set = set(required_params)
                    
                    if not required_params_set.issubset(provided_params):
                        missing = required_params_set - provided_params
                        logger.warning(f"❌ Local model missing required parameters for {tool_name}: {missing}")
                        logger.warning(f"   Required: {required_params}")
                        logger.warning(f"   Provided: {list(provided_params)}")
                        return False, "", []
                
                # Execute the tool
                try:
                    logger.info(f"🔧 Local model executing tool: {tool_name} with args: {tool_args}")
                    tool_result = await call_tool_with_timeout(mcp_client, {
                        "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                        "id": str(uuid.uuid4())
                    })
                    
                    final_response = response_message or f"Executed {tool_name} successfully."
                    if "error" not in tool_result.lower() and "failed" not in tool_result.lower():
                        logger.info(f"✅ Local model successfully executed {tool_name}")
                        return True, final_response, [{"name": tool_name, "result": tool_result}]
                    else:
                        logger.warning(f"❌ Local tool execution failed: {tool_result}")
                        return False, "", []
                        
                except Exception as e:
                    logger.warning(f"❌ Local tool execution error: {e}")
                    return False, "", []
            else:
                logger.warning(f"❌ Local model suggested invalid tool: {tool_name}")
                return False, "", []
        
        # If no tool call but response looks good, return it
        if response_text and len(response_text.split()) > 2 and not response_text.startswith("ESCALATE"):
            logger.info("✅ Local model provided direct response")
            return True, response_text, []
        
        logger.info("🔄 Local model response insufficient, escalating")
        return False, "", []
        
    except asyncio.TimeoutError:
        logger.warning(f"⏱️  Local model timeout after {config.local_timeout}s")
        return False, "", []
    except Exception as e:
        logger.warning(f"❌ Local model error: {e}")
        return False, "", []
    finally:
        # Always ensure we're not stuck in local processing mode
        if state.get_mode() == AssistantMode.LOCAL_PROCESSING:
            state.set_mode(AssistantMode.PROCESSING)

################################################################################
# Stuck detection task (unchanged)
################################################################################

async def stuck_detection_task():
    """Background task to check if assistant is stuck and listen for wake phrase"""
    while state.running:
        try:
            await asyncio.sleep(config.stuck_check_interval)
            
            if state.is_stuck():
                logger.warning(f"Assistant appears stuck (processing for {time.time() - state.processing_start_time:.1f}s)")
                state.set_mode(AssistantMode.STUCK_CHECK)
                
                audio_buffer = await audio_recorder.record_until_silence(check_stuck_phrase=True)
                
                if audio_buffer:
                    try:
                        response = state.openai_client.audio.transcriptions.create(
                            model=config.stt_model,
                            file=audio_buffer,
                            language="en"
                        )
                        text = response.text.strip().lower().replace(",", "").replace("?", "")
                        
                        logger.info(f"Stuck check transcription: {text}")
                        
                        if any(word in config.stuck_phrase.split() for word in text.split()):
                            logger.info("Wake phrase detected! Resetting to listening mode")
                            state.interrupt_flag.set()
                            state.set_mode(AssistantMode.LISTENING)
                            asyncio.create_task(speak_text("I'm back! Sorry about that. How can I help you?"))
                    except Exception as e:
                        logger.error(f"Error in stuck phrase detection: {e}")
                
        except Exception as e:
            logger.error(f"Error in stuck detection task: {e}")
            await asyncio.sleep(1)

################################################################################
# MCP Client management (unchanged)
################################################################################

@asynccontextmanager
async def get_mcp_client():
    """Context manager for MCP client with subprocess management"""
    server_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mcp_os", "server.py")
    )
    
    client = None
    try:
        if not state.mcp_process or state.mcp_process.poll() is not None:
            logger.info(f"Starting MCP server: {server_path}")
            state.mcp_process = subprocess.Popen(
                [sys.executable, server_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
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
    
    if (use_cache and 
        state.tools_cache and 
        current_time - state.tools_cache_time < 60):
        return state.tools_cache
    
    tools = []
    try:
        for tool in await mcp_client.list_tools():
            try:
                if hasattr(tool, 'openai_schema'):
                    spec = tool.openai_schema()
                else:
                    spec = {
                        "name": tool.name,
                        "description": getattr(tool, "description", ""),
                        "parameters": getattr(tool, "inputSchema", {
                            "type": "object", 
                            "properties": {}
                        })
                    }
                
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
        if state.tools_cache:
            logger.info("Using cached tools due to listing failure")
            return state.tools_cache
        raise
    
    state.tools_cache = tools
    state.tools_cache_time = current_time
    logger.info(f"Loaded {len(tools)} tools")
    
    return tools

################################################################################
# Tool execution with timeout and error handling (unchanged)
################################################################################

async def call_tool_with_timeout(mcp_client: Client, call) -> str:
    """Execute a tool call with timeout and error handling"""
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
        result = await asyncio.wait_for(execute_tool(), timeout=config.tool_timeout)
        logger.info(f"Tool {name} completed successfully")
        return result
    except asyncio.TimeoutError:
        error_msg = f"Tool {name} timed out after {config.tool_timeout}s"
        logger.error(error_msg)
        return error_msg

################################################################################
# STT with retry logic (unchanged)
################################################################################

async def transcribe_audio(audio_buffer: io.BytesIO, check_stuck_phrase: bool = False) -> Optional[str]:
    """Transcribe audio with retry logic and validation"""
    async def do_transcribe():
        try:
            response = state.openai_client.audio.transcriptions.create(
                model=config.stt_model,
                file=audio_buffer,
                language="en"
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"STT error: {e}")
            raise
    
    try:
        text = await retry_with_backoff(do_transcribe())
        
        if check_stuck_phrase:
            return text
        
        if not text or len(text.strip()) == 0:
            logger.info("Empty transcription, ignoring")
            return None
        
        noise_patterns = [
            "Thank you.", "Thanks for watching!", "Bye!", "Thanks.", 
            "Thank you for watching.", "Please subscribe.", "Bye-bye",
            "Hello?", "Yeah.", "Uh-huh.", "Mm-hmm.", "Okay.", "Alright.",
            "[Music]", "[Applause]", "[Laughter]", "[MUSIC]", "[music]",
            "♪", "♫", "♪♪", "♬", "♩",
            "음악", "音楽", "嗯", "啊", "呃", "哦", "是", "的",
            "ありがとう", "こんにちは", "さようなら",
            "Merci", "Bonjour", "Au revoir", "Gracias", "Hola",
            "you", "the", "a", ".", ",", "!", "?", "-", "...",
            "Shh", "Shh.", "Shhh", "Hmm", "Hmm.", "Hm",
            "background noise", "inaudible", "[inaudible]",
            "playing", "music playing", "video playing",
            "Transcribed by", "Subtitles by", "Captions by"
        ]
        
        if text.strip().lower() in [p.lower() for p in noise_patterns]:
            logger.info(f"Ignoring noise transcription: {text}")
            return None
        
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars < len(text) * 0.5:
            logger.info(f"Ignoring non-text transcription: {text}")
            return None
        
        words = [w for w in text.split() if w.strip() and any(c.isalpha() for c in w)]
        
        # More lenient word count check - allow 2+ words if they seem meaningful
        if len(words) < 2:
            logger.info(f"Transcription too short ({len(words)} words): {text}")
            return None
        elif len(words) < config.min_confidence_length:
            # For 2-word phrases, check if they contain meaningful content
            meaningful_words = {"hello", "hi", "hey", "open", "launch", "start", "create", "play", "stop", "close", "help", "thanks", "please", "yes", "no"}
            if any(word.lower() in meaningful_words for word in words):
                logger.info(f"Accepting short but meaningful transcription ({len(words)} words): {text}")
            else:
                logger.info(f"Transcription too short ({len(words)} words): {text}")
                return None
        
        if len(set(text.replace(" ", ""))) < 3:
            logger.info(f"Ignoring repetitive transcription: {text}")
            return None
        
        non_ascii_chars = sum(1 for c in text if ord(c) > 127)
        if non_ascii_chars > len(text) * 0.3:
            logger.info(f"Ignoring non-English transcription: {text}")
            return None
        
        common_english_words = {"the", "is", "are", "and", "or", "but", "in", "on", "at", 
                               "to", "for", "of", "with", "as", "by", "that", "this",
                               "what", "how", "when", "where", "why", "who", "which",
                               "can", "will", "would", "should", "could", "have", "has", 
                               "launch", "open", "start", "play", "run", "create", "delete", "edit", "save"}
        
        words_lower = [w.lower() for w in words]
        english_word_count = sum(1 for w in words_lower if w in common_english_words)
        
        if len(words) < 10 and english_word_count == 0:
            logger.info(f"No common English words found in short phrase: {text}")
            return None
        
        logger.info(f"Valid transcription: {text[:100]}...")
        return text
        
    except Exception as e:
        logger.error(f"Failed to transcribe after retries: {e}")
        return None

################################################################################
# TTS with interruption support (unchanged)
################################################################################

async def speak_text(text: str) -> bool:
    """Convert text to speech and play it with interruption support"""
    if not text or text.isspace():
        text = "Okay."
    
    state.set_mode(AssistantMode.SPEAKING)
    state.interrupt_flag.clear()
    
    try:
        logger.info(f"Speaking: {text[:50]}...")
        
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
        
        if state.interrupt_flag.is_set():
            logger.info("Speech interrupted before playback")
            state.set_mode(AssistantMode.LISTENING)
            return False
        
        audio_data = audio_response.read()
        success = False
        
        try:
            def play_with_sounddevice():
                audio_buffer = io.BytesIO(audio_data)
                data, sample_rate = sf.read(audio_buffer, dtype="float32")
                
                try:
                    sd.default.device = None
                    sd.play(data, sample_rate)
                    sd.wait()
                    return True
                except sd.PortAudioError:
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
        
        if not success and PYGAME_AVAILABLE:
            try:
                def play_with_pygame():
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(audio_data)
                        tmp_path = tmp_file.name
                    
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    
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
        state.set_mode(AssistantMode.LISTENING)
        return True
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        state.set_mode(AssistantMode.LISTENING)
        return False

################################################################################
# Error handling and retry utilities (unchanged)
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
# Shutdown utilities (unchanged)
################################################################################

async def shutdown_system():
    """Properly shut down both server and client"""
    logger.info("Initiating system shutdown...")
    
    audio_recorder.stop()
    
    if state.mcp_client:
        try:
            await state.mcp_client.close()
        except Exception as e:
            logger.error(f"Error closing MCP client: {e}")
    
    if state.mcp_process:
        try:
            logger.info("Terminating MCP server process...")
            state.mcp_process.terminate()
            
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
# Enhanced conversation loop with local model integration
################################################################################

async def enhanced_conversation_loop():
    """Enhanced conversation loop with local model routing"""
    logger.info("Starting enhanced conversation loop with local model support")
    
    state.reset_conversation()
    
    if not state.openai_client:
        try:
            state.openai_client = OpenAI(api_key=config.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    audio_recorder.start()
    stuck_task = asyncio.create_task(stuck_detection_task())
    
    async with get_mcp_client() as mcp_client:
        state.mcp_client = mcp_client
        tools = await get_tools(mcp_client)
        
        # Enhanced greeting based on available models
        if router.local_available:
            await speak_text(f"Hello! I'm ready with local {config.local_model} and cloud processing.")
        else:
            await speak_text("Hello! I'm ready with cloud processing. Local model unavailable.")
        
        while state.running:
            try:
                # Only listen when in listening mode AND not processing another request
                if state.get_mode() != AssistantMode.LISTENING or state.is_processing():
                    await asyncio.sleep(0.1)
                    continue
                
                # Record and transcribe
                audio_buffer = await audio_recorder.record_until_silence()
                if not audio_buffer:
                    continue
                
                # IMMEDIATE: Set processing flag before transcription to block overlapping requests
                if state.is_processing():
                    logger.debug("🚫 Another request started during recording, aborting")
                    continue
                
                # Lock this request immediately
                state.set_mode(AssistantMode.RECORDING)
                state.processing_request = True  # Immediately block new requests
                logger.info("🔒 Request locked - blocking new audio processing")
                
                try:
                    user_text = await transcribe_audio(audio_buffer)
                    if not user_text:
                        continue
                    
                    logger.info(f"🎤 User said: {user_text}")
                    
                    # Handle special commands
                    lower_text = user_text.lower().strip()
                    if lower_text in {"reset chat", "new chat", "clear history"}:
                        state.reset_conversation()
                        await speak_text("Starting a new conversation.")
                        continue
                    
                    if any(phrase in lower_text for phrase in ["exit", "quit", "goodbye", "shut down", "shutdown"]):
                        await speak_text("Goodbye! Shutting down the system...")
                        await shutdown_system()
                        break
                    
                    # Process the request
                    logger.info("🔒 Starting request processing")
                    
                    # Determine processing route
                    use_local = router.should_use_local(user_text)
                    local_success = False
                    assistant_response = ""
                    
                    # Add user message to history first
                    state.conversation_history.append({"role": "user", "content": user_text})
                    
                    # Try local model first if applicable
                    if use_local:
                        logger.info("🤖 Trying local model first...")
                        local_success, assistant_response, tool_calls = await process_with_local_model(
                            user_text, tools, mcp_client
                        )
                        
                        if local_success:
                            logger.info("✅ Local model handled request successfully - SKIPPING cloud model")
                            state.conversation_history.append({"role": "assistant", "content": assistant_response})
                            # Skip cloud processing entirely
                        else:
                            logger.info("❌ Local model failed/escalated, falling back to cloud...")
                            # Continue to cloud processing below
                    
                    # Only use cloud model if local didn't succeed
                    if not local_success:
                        logger.info("☁️  Using cloud model...")
                        state.set_mode(AssistantMode.PROCESSING)
                        
                        completion = state.openai_client.chat.completions.create(
                            model=config.chat_model,
                            messages=state.conversation_history,
                            tools=tools,
                            tool_choice="auto"
                        )
                        
                        choice = completion.choices[0]
                        message = choice.message
                        
                        if choice.finish_reason == "tool_calls" and message.tool_calls:
                            # Handle cloud tool calls
                            state.conversation_history.append({
                                "role": "assistant",
                                "content": message.content,
                                "tool_calls": message.tool_calls
                            })
                            
                            for tool_call in message.tool_calls:
                                if state.interrupt_flag.is_set() or state.get_mode() == AssistantMode.STUCK_CHECK:
                                    logger.info("Processing interrupted")
                                    break
                                    
                                tool_result = await call_tool_with_timeout(mcp_client, tool_call)
                                state.conversation_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result
                                })
                            
                            # Get follow-up response if not interrupted
                            if not state.interrupt_flag.is_set() and state.get_mode() != AssistantMode.STUCK_CHECK:
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
                    
                    # Speak the response if not interrupted
                    if not state.interrupt_flag.is_set() and state.get_mode() != AssistantMode.STUCK_CHECK:
                        await speak_text(assistant_response)
                    else:
                        logger.info("Skipping speech due to interruption")
                    
                except Exception as e:
                    logger.error(f"Error in processing: {e}")
                    assistant_response = "I encountered an error processing your request. Please try again."
                    await speak_text(assistant_response)
                
                finally:
                    # Always ensure we return to listening mode and clear processing flag
                    state.set_mode(AssistantMode.LISTENING)
                    logger.info("🔓 Request processing complete - accepting new requests")
                    
            except Exception as e:
                logger.error(f"Error in enhanced conversation loop: {e}")
                state.set_mode(AssistantMode.ERROR)
                await asyncio.sleep(config.reconnect_delay)
                state.set_mode(AssistantMode.LISTENING)
                continue
    
    stuck_task.cancel()
    try:
        await stuck_task
    except asyncio.CancelledError:
        pass

################################################################################
# Main application with restart capability
################################################################################

async def run_forever():
    """Run the assistant with automatic restart on failure"""
    restart_count = 0
    
    while state.running:
        try:
            logger.info(f"Starting enhanced voice assistant (restart #{restart_count})")
            await enhanced_conversation_loop()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            await shutdown_system()
            break
            
        except Exception as e:
            restart_count += 1
            logger.exception(f"Fatal error in conversation loop (restart #{restart_count}): {e}")
            
            if restart_count > 10:
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
    """Main entry point with enhanced local model support"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Enhanced Voice Assistant starting up...")
    logger.info(f"Configuration:")
    logger.info(f"  STT: {config.stt_model}")
    logger.info(f"  Cloud Chat: {config.chat_model}")
    logger.info(f"  Local Model: {config.local_model} ({'Available' if router.local_available else 'Unavailable'})")
    logger.info(f"  TTS: {config.tts_model}")
    logger.info(f"  Local First: {config.use_local_first}")
    logger.info(f"  Processing timeout: {config.processing_timeout}s")
    logger.info(f"  Wake phrase: '{config.stuck_phrase}'")
    
    if router.local_available:
        logger.info("🚀 Hybrid mode: Local model for system operations, cloud for complex queries")
    else:
        logger.info("☁️  Cloud-only mode: All processing via OpenAI")
    
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        logger.info("\nShutdown complete")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    
    logger.info("Enhanced Voice Assistant shutdown complete")

if __name__ == "__main__":
    main()