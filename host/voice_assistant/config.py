# voice_assistant/config.py
"""
Configuration management for the voice assistant
"""

import os
import sys
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    processing_timeout: float = 60.0  # Maximum time for full processing cycle
    stuck_phrase: str = "hello abraxas are you stuck"  # Wake phrase (normalized)
    stuck_check_interval: float = 5.0  # How often to check for stuck state
    reconnect_delay: float = 2.0
    max_retries: int = 3
    log_level: str = "INFO"
    log_file: str = "voice_assistant.log"
    enable_vad: bool = True
    vad_aggressiveness: int = 3
    silence_threshold: float = 0.03
    silence_duration: float = 1.5
    min_speech_duration: float = 0.8
    energy_threshold_multiplier: float = 2.0
    min_confidence_length: int = 2
    max_energy_threshold: float = 0.5
    english_confidence_threshold: float = 0.7
    
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