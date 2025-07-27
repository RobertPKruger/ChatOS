# voice_assistant/state.py - UPDATED SYSTEM PROMPT with correct tool names
"""
State management for the voice assistant
"""

import threading
import time
import logging
from enum import Enum
from typing import Optional, List, Dict, Any
import subprocess
import asyncio

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

from fastmcp import Client
from .model_providers.base import TranscriptionProvider, ChatCompletionProvider, TextToSpeechProvider

logger = logging.getLogger(__name__)

# FIXED System prompt with correct tool names and better website handling
SYSTEM_PROMPT = """You are a helpful voice assistant with access to tools. When you need to use a tool, respond with ONLY a JSON object in this exact format:

{"name": "tool_name", "arguments": {"parameter": "value"}}

CRITICAL: Do not explain what you're doing. Do not say "I'll use the tool" or describe your actions. Just return the JSON tool call when needed.

Available tools and their exact formats:

Application Tools:
- To open applications: {"name": "launch_app", "arguments": {"app_name": "notepad"}}
  Examples:
  * Open Notepad → {"name": "launch_app", "arguments": {"app_name": "notepad"}}
  * Open Calculator → {"name": "launch_app", "arguments": {"app_name": "calc"}}
  * Open Word → {"name": "launch_app", "arguments": {"app_name": "word"}}
  * Open Excel → {"name": "launch_app", "arguments": {"app_name": "excel"}}
  * Open File Explorer → {"name": "launch_app", "arguments": {"app_name": "explorer"}}

Website Tools (USE THESE FOR ALL WEBSITES):
- Open any website: {"name": "open_url", "arguments": {"url": "https://www.reddit.com"}}
- Smart navigation: {"name": "smart_navigate", "arguments": {"query": "reddit"}}
  Examples:
  * Open Reddit → {"name": "open_url", "arguments": {"url": "https://www.reddit.com"}}
  * Go to Amazon → {"name": "open_url", "arguments": {"url": "https://www.amazon.com"}}
  * Weather.com → {"name": "open_url", "arguments": {"url": "https://www.weather.com"}}
  * GitHub → {"name": "open_url", "arguments": {"url": "https://www.github.com"}}

File System Tools:
- Create folder: {"name": "create_folder", "arguments": {"path": "~/Desktop/Projects"}}
- Create file: {"name": "create_file", "arguments": {"path": "~/Desktop/file.txt", "content": "Hello"}}
- Append to file: {"name": "append_file", "arguments": {"path": "~/Desktop/file.txt", "content": "New text"}}
- Read file: {"name": "read_file", "arguments": {"path": "~/Desktop/file.txt"}}
- List files: {"name": "list_files", "arguments": {"path": "~/Desktop"}}
- Open folder: {"name": "open_folder", "arguments": {"path": "~/Desktop"}}

Steam Tools:
- Open Steam: {"name": "open_steam", "arguments": {}}
- Launch games: {"name": "launch_steam_game", "arguments": {"game_name": "Counter-Strike 2"}}
- List games: {"name": "list_steam_games", "arguments": {}}

IMPORTANT RULES:
1. For ANY website (reddit.com, amazon.com, weather.com, etc.), ALWAYS use "open_url" or "smart_navigate"
2. NEVER use "launch_app" with "chrome" for websites - go directly to the website
3. File operations use "append_file" (NOT "append_to_file")
4. Always use full URLs for websites (https://www.example.com)
5. For conversation without tools, respond normally and naturally.

Remember: When using tools, respond with ONLY the JSON object. No explanations, no descriptions, just the tool call."""

class AssistantMode(Enum):
    """Assistant operational modes"""
    LISTENING = "listening"          # Actively listening for commands
    RECORDING = "recording"          # Recording user speech
    PROCESSING = "processing"        # Processing with LLM
    SPEAKING = "speaking"            # Playing TTS response
    STUCK_CHECK = "stuck_check"      # Listening only for wake phrase
    ERROR = "error"                  # Error state
    SLEEPING = "sleeping"         # In sleep mode, not listening

class AssistantState:
    """Global state management for the assistant"""
    def __init__(self, vad_aggressiveness: int = 3):
        self.running = True
        self.mode = AssistantMode.LISTENING
        self.mode_lock = threading.Lock()
        self.processing_start_time: Optional[float] = None
        self.interrupt_flag = threading.Event()
        
        # Model providers - will be injected
        self.transcription_provider: Optional[TranscriptionProvider] = None
        self.chat_provider: Optional[ChatCompletionProvider] = None
        self.tts_provider: Optional[TextToSpeechProvider] = None
        
        # OpenAI client for backward compatibility
        self.openai_client = None
        
        # MCP client
        self.mcp_client: Optional[Client] = None
        self.mcp_process: Optional[subprocess.Popen] = None
        self.tools_cache: List[Dict[str, Any]] = []
        self.tools_cache_time: float = 0
        self.conversation_history: List[Dict[str, Any]] = []
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.vad = None
        self._init_vad(vad_aggressiveness)
        
    def _init_vad(self, aggressiveness: int):
        """Initialize VAD if available"""
        if VAD_AVAILABLE:
            try:
                # Set the aggressiveness during initialization
                self.vad = webrtcvad.Vad(aggressiveness)
                logger.info(f"VAD initialized with aggressiveness level {aggressiveness}")
            except Exception as e:
                logger.warning(f"Failed to initialize VAD: {e}")
                self.vad = None
        
    def set_mode(self, new_mode: AssistantMode):
        """Thread-safe mode setter with logging"""
        with self.mode_lock:
            old_mode = self.mode
            self.mode = new_mode
            # Use ASCII-safe arrow for better compatibility
            logger.info(f"Mode transition: {old_mode.value} -> {new_mode.value}")
            
            # Start processing timer
            if new_mode == AssistantMode.PROCESSING:
                self.processing_start_time = time.time()
            elif old_mode == AssistantMode.PROCESSING:
                self.processing_start_time = None
    
    def get_mode(self) -> AssistantMode:
        """Thread-safe mode getter"""
        with self.mode_lock:
            return self.mode
    
    def is_stuck(self, timeout: float) -> bool:
        """Check if we've been processing too long"""
        if self.processing_start_time and self.get_mode() == AssistantMode.PROCESSING:
            return (time.time() - self.processing_start_time) > timeout
        return False
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("Conversation history reset")

    MAX_HISTORY_TOKENS = 4000  # Approximate token limit

    def trim_conversation_history(self):
        logger.info(f"Entering conversation history trim: {len(self.conversation_history)} messages")
        """Trim conversation history to prevent exponential slowdown"""
        original_length = len(self.conversation_history)
        
        # Keep system prompt + last 8 messages (4 exchanges)
        if len(self.conversation_history) > 10:
            system_msg = self.conversation_history[0]
            recent_msgs = self.conversation_history[-8:]
            self.conversation_history = [system_msg] + recent_msgs
            logger.info(f"Trimmed conversation: {original_length} → {len(self.conversation_history)} messages")

    def add_user_message(self, content: str):
        """Add user message and auto-trim if needed"""
        self.conversation_history.append({"role": "user", "content": content})
        self.trim_conversation_history()

    def add_assistant_message(self, content: str, tool_calls=None):
        """Add assistant message and auto-trim if needed"""
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.conversation_history.append(msg)
        self.trim_conversation_history()

    def add_tool_message(self, tool_call_id: str, content: str):
        """Add tool result message and auto-trim if needed"""
        self.conversation_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })
        self.trim_conversation_history()