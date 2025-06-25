# voice_assistant/state.py
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

from openai import OpenAI
from fastmcp import Client

logger = logging.getLogger(__name__)

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

class AssistantMode(Enum):
    """Assistant operational modes"""
    LISTENING = "listening"          # Actively listening for commands
    RECORDING = "recording"          # Recording user speech
    PROCESSING = "processing"        # Processing with LLM
    SPEAKING = "speaking"            # Playing TTS response
    STUCK_CHECK = "stuck_check"      # Listening only for wake phrase
    ERROR = "error"                  # Error state

class AssistantState:
    """Global state management for the assistant"""
    def __init__(self, vad_aggressiveness: int = 3):
        self.running = True
        self.mode = AssistantMode.LISTENING
        self.mode_lock = threading.Lock()
        self.processing_start_time: Optional[float] = None
        self.interrupt_flag = threading.Event()
        self.openai_client: Optional[OpenAI] = None
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