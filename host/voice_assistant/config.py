# voice_assistant/config.py
"""
Configuration management for the voice assistant
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
    # Model provider configuration
    transcription_provider: str = "openai"  # "openai", "local_whisper", "hybrid"
    transcription_fallback: Optional[str] = None
    chat_provider: str = "openai"  # "openai", "ollama", "hybrid"
    chat_fallback: Optional[str] = None
    tts_provider: str = "openai"  # "openai", "pyttsx3", "hybrid"
    tts_fallback: Optional[str] = None

    use_local_first: bool = True  # Use local models first if available
    
    # OpenAI configuration


    openai_api_key: str = ""
    stt_model: str = "gpt-4o-transcribe"
    chat_model: str = "gpt-4o"
    tts_model: str = "tts-1"
    tts_voice: str = "nova"
    
    # Local model configuration
    whisper_model_size: str = "base"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    pyttsx3_voice_id: Optional[str] = None
    pyttsx3_rate: int = 150
    
    # Hybrid configuration
    use_primary_for_tools: bool = True  # Always use OpenAI for tool calls
    
    # Audio configuration
    sample_rate: int = 16000
    min_audio_size: int = 10000
    silence_threshold: float = 0.03
    silence_duration: float = 1.5
    min_speech_duration: float = 0.8
    energy_threshold_multiplier: float = 2.0
    max_energy_threshold: float = 0.5
    
    # Processing configuration
    tool_timeout: float = 30.0
    processing_timeout: float = 60.0  # Maximum time for full processing cycle
    stuck_phrase: str = "hello abraxas are you stuck"  # Wake phrase (normalized)
    stuck_check_interval: float = 5.0  # How often to check for stuck state
    
    # Retry and connection configuration
    reconnect_delay: float = 2.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "voice_assistant.log"
    
    # VAD configuration
    enable_vad: bool = True
    vad_aggressiveness: int = 3
    
    # Validation configuration
    min_confidence_length: int = 2
    english_confidence_threshold: float = 0.7
    
    # Compatibility fields (for backward compatibility)
    min_phrase_length: float = 0.5
    recording_timeout: float = 10.0
    chunk_duration: float = 0.1
    wake_phrase_timeout: float = 5.0

    local_chat_model: str      = "llama3.1:8b-instruct-q4_0"
    frontier_chat_model: str   = "gpt-4o"
    local_chat_timeout: float  = 30.0
    ollama_host: str           = "http://localhost:11434"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls(
            # Model provider configuration
            transcription_provider=os.getenv("TRANSCRIPTION_PROVIDER", "openai"),
            transcription_fallback=os.getenv("TRANSCRIPTION_FALLBACK"),
            chat_provider=os.getenv("CHAT_PROVIDER", "openai"),
            chat_fallback=os.getenv("CHAT_FALLBACK"),
            tts_provider=os.getenv("TTS_PROVIDER", "openai"),
            tts_fallback=os.getenv("TTS_FALLBACK"),

            use_local_first = os.getenv("USE_LOCAL_FIRST", "true").lower() in ["true", "1", "yes", "on"],
            
            # OpenAI configuration
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            stt_model=os.getenv("STT_MODEL", "gpt-4o-transcribe"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o"),
            tts_model=os.getenv("TTS_MODEL", "tts-1"),
            tts_voice=os.getenv("TTS_VOICE", "nova"),
            
            # Local model configurationF
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
            pyttsx3_voice_id=os.getenv("PYTTSX3_VOICE_ID"),
            pyttsx3_rate=int(os.getenv("PYTTSX3_RATE", "150")),

            # Ollama configuration -- todo: fix these 
            local_chat_model   = os.getenv("LOCAL_CHAT_MODEL", "llama3.1:8b-instruct-q4_0"),
            frontier_chat_model = os.getenv("FRONTIER_CHAT_MODEL", "gpt-4o"),
            local_chat_timeout = int(os.getenv("LOCAL_CHAT_TIMEOUT", "30")),
            ollama_host        = os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            
            # Hybrid configuration
            use_primary_for_tools=os.getenv("USE_PRIMARY_FOR_TOOLS", "true").lower() == "true",
            
            # Audio configuration
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            min_audio_size=int(os.getenv("MIN_AUDIO_SIZE", "10000")),
            silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.03")),
            silence_duration=float(os.getenv("SILENCE_DURATION", "1.5")),
            min_speech_duration=float(os.getenv("MIN_SPEECH_DURATION", "0.8")),
            energy_threshold_multiplier=float(os.getenv("ENERGY_THRESHOLD_MULTIPLIER", "2.0")),
            max_energy_threshold=float(os.getenv("MAX_ENERGY_THRESHOLD", "0.5")),
            
            # Processing configuration
            tool_timeout=float(os.getenv("TOOL_TIMEOUT", "30.0")),
            processing_timeout=float(os.getenv("PROCESSING_TIMEOUT", "60.0")),
            stuck_phrase=os.getenv("STUCK_PHRASE", "hello abraxas are you stuck").lower().replace(",", "").replace("?", ""),
            stuck_check_interval=float(os.getenv("STUCK_CHECK_INTERVAL", "5.0")),
            
            # Retry and connection configuration
            reconnect_delay=float(os.getenv("RECONNECT_DELAY", "2.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
            
            # Logging configuration
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "voice_assistant.log"),
            
            # VAD configuration
            enable_vad=os.getenv("ENABLE_VAD", "true").lower() == "true",
            vad_aggressiveness=int(os.getenv("VAD_AGGRESSIVENESS", "3")),
            
            # Validation configuration
            min_confidence_length=int(os.getenv("MIN_CONFIDENCE_LENGTH", "3")),
            english_confidence_threshold=float(os.getenv("ENGLISH_CONFIDENCE_THRESHOLD", "0.7")),
            
            # Compatibility fields
            min_phrase_length=float(os.getenv("MIN_PHRASE_LENGTH", "0.5")),
            recording_timeout=float(os.getenv("RECORDING_TIMEOUT", "10.0")),
            chunk_duration=float(os.getenv("CHUNK_DURATION", "0.1")),
            wake_phrase_timeout=float(os.getenv("WAKE_PHRASE_TIMEOUT", "5.0"))
        )

def setup_logging(config: Config):
    """Configure logging with rotation and proper formatting"""
    # Create handlers with UTF-8 encoding
    file_handler = logging.FileHandler(config.log_file, encoding='utf-8')
    
    # For console output on Windows, use UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    if sys.platform == "win32":
        # Set console to UTF-8 mode on Windows
        import locale
        import codecs
        if sys.stdout.encoding != 'utf-8':
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)