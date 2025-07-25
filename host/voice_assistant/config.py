# voice_assistant/config.py - PERFORMANCE OPTIMIZED
"""
Configuration management for the voice assistant with performance optimizations
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the voice assistant"""
    # === CORE SETTINGS ===
    use_local_first: bool
    
    # === API KEYS ===
    openai_api_key: str
    
    # === MODEL CONFIGURATION ===
    # Transcription
    stt_model: str
    
    # Chat Models
    local_chat_model: str
    frontier_chat_model: str
    local_chat_timeout: float
    
    # TTS
    tts_model: str
    tts_voice: str
    
    # === LOCAL MODEL CONFIGURATION ===
    whisper_model_size: str
    ollama_host: str
    pyttsx3_rate: int
    
    # === AUDIO CONFIGURATION ===
    sample_rate: int
    min_audio_size: int
    silence_threshold: float
    silence_duration: float
    min_speech_duration: float
    energy_threshold_multiplier: float
    max_energy_threshold: float
    
    # === PROCESSING CONFIGURATION ===
    tool_timeout: float
    processing_timeout: float
    stuck_phrase: str
    stuck_check_interval: float
    
    # === SYSTEM CONFIGURATION ===
    reconnect_delay: float
    max_retries: int
    retry_delay: float
    
    # === LOGGING CONFIGURATION ===
    log_level: str
    log_file: str
    
    # === VAD CONFIGURATION ===
    enable_vad: bool
    vad_aggressiveness: int
    
    # === VALIDATION CONFIGURATION ===
    min_confidence_length: int
    english_confidence_threshold: float
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables with performance optimizations"""
        return cls(
            # === CORE SETTINGS ===
            use_local_first=os.getenv("USE_LOCAL_FIRST", "true").lower() in ["true", "1", "yes", "on"],
            
            # === API KEYS ===
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            
            # === MODEL CONFIGURATION ===
            # Transcription
            stt_model=os.getenv("STT_MODEL", "gpt-4o-transcribe"),
            
            # Chat Models
            local_chat_model=os.getenv("LOCAL_CHAT_MODEL", "llama3.1:8b-instruct-q4_0"),
            frontier_chat_model=os.getenv("FRONTIER_CHAT_MODEL", "gpt-4o"),
            # OPTIMIZED: Reduced timeout from 30 to 12 seconds for better UX
            local_chat_timeout=float(os.getenv("LOCAL_CHAT_TIMEOUT", "12")),
            
            # TTS
            tts_model=os.getenv("TTS_MODEL", "tts-1"),
            tts_voice=os.getenv("TTS_VOICE", "nova"),
            
            # === LOCAL MODEL CONFIGURATION ===
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            pyttsx3_rate=int(os.getenv("PYTTSX3_RATE", "150")),
            
            # === AUDIO CONFIGURATION ===
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            min_audio_size=int(os.getenv("MIN_AUDIO_SIZE", "10000")),
            silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.03")),
            silence_duration=float(os.getenv("SILENCE_DURATION", "1.5")),
            min_speech_duration=float(os.getenv("MIN_SPEECH_DURATION", "0.8")),
            energy_threshold_multiplier=float(os.getenv("ENERGY_THRESHOLD_MULTIPLIER", "2.0")),
            max_energy_threshold=float(os.getenv("MAX_ENERGY_THRESHOLD", "0.5")),
            
            # === PROCESSING CONFIGURATION ===
            tool_timeout=float(os.getenv("TOOL_TIMEOUT", "30.0")),
            # OPTIMIZED: Reduced from 60 to 45 seconds
            processing_timeout=float(os.getenv("PROCESSING_TIMEOUT", "45.0")),
            stuck_phrase=os.getenv("STUCK_PHRASE", "hello abraxas are you stuck").lower().replace(",", "").replace("?", ""),
            stuck_check_interval=float(os.getenv("STUCK_CHECK_INTERVAL", "5.0")),
            
            # === SYSTEM CONFIGURATION ===
            reconnect_delay=float(os.getenv("RECONNECT_DELAY", "2.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
            
            # === LOGGING CONFIGURATION ===
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "voice_assistant.log"),
            
            # === VAD CONFIGURATION ===
            enable_vad=os.getenv("ENABLE_VAD", "true").lower() == "true",
            vad_aggressiveness=int(os.getenv("VAD_AGGRESSIVENESS", "3")),
            
            # === VALIDATION CONFIGURATION ===
            min_confidence_length=int(os.getenv("MIN_CONFIDENCE_LENGTH", "3")),
            english_confidence_threshold=float(os.getenv("ENGLISH_CONFIDENCE_THRESHOLD", "0.7")),
        )

def setup_logging(config: Config):
    """Configure logging with rotation and proper formatting"""
    # Check if we should force console output (when launched via launcher)
    force_console = os.getenv("CHATOS_CONSOLE_OUTPUT") == "1"
    
    # Create handlers with UTF-8 encoding
    file_handler = logging.FileHandler(config.log_file, encoding='utf-8')
    
    handlers = [file_handler]
    
    # Always add console handler, but configure it properly
    console_handler = logging.StreamHandler(sys.stdout)
    if sys.platform == "win32":
        # Set console to UTF-8 mode on Windows
        import locale
        import codecs
        if sys.stdout.encoding != 'utf-8':
            try:
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            except AttributeError:
                pass  # Already wrapped or not available
        if sys.stderr.encoding != 'utf-8':
            try:
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
            except AttributeError:
                pass  # Already wrapped or not available
    
    handlers.append(console_handler)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if force_console:
        # Simpler format for console output via launcher
        log_format = '%(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True  # Reconfigure even if already configured
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)