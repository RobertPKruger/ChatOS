# voice_assistant/state.py - ENHANCED SYSTEM PROMPT FOR CONSISTENT TOOL CALLING
"""
State management for the voice assistant with improved system prompt
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


SYSTEM_PROMPT = """You are a voice assistant. When someone asks you to DO something, use a JSON tool. When they ask a question, give a text answer.

🚨 **RULE: If you see these words, you MUST use a tool:**
- "open" = tool
- "go to" = tool  
- "navigate" = tool
- "launch" = tool
- "start" = tool
- "create" = tool

🚨 **NEVER say "I've completed" or "I've opened" without using a tool first!**

=== STEP-BY-STEP PROCESS ===

1. Read the user request
2. Look for action words: "open", "go to", "navigate", "launch", "start", "create"
3. If you find action words → Use a tool
4. If no action words → Give text response

=== TOOLS ===

**Applications:** {"name": "launch_app", "arguments": {"app_name": "notepad"}}
**Websites:** {"name": "open_url", "arguments": {"url": "https://site.com"}}
**Steam Games:** {"name": "launch_steam_game", "arguments": {"game_name": "game"}}
**Files:** {"name": "create_file", "arguments": {"path": "file.txt", "content": ""}}

=== EXAMPLES ===

User: "Please open Nugget News"
Think: Contains "open" → Use tool → Website
Response: {"name": "open_url", "arguments": {"url": "https://www.nuggetnews.com"}}

User: "Go to nuggetnews.com"  
Think: Contains "go to" → Use tool → Website
Response: {"name": "open_url", "arguments": {"url": "https://nuggetnews.com"}}

User: "Open Magic the Gathering Arena on Steam"
Think: Contains "open" → Use tool → Steam game
Response: {"name": "launch_steam_game", "arguments": {"game_name": "Magic the Gathering Arena"}}

User: "Please open notepad"
Think: Contains "open" → Use tool → Application
Response: {"name": "launch_app", "arguments": {"app_name": "notepad"}}

User: "What is 2+2?"
Think: No action words → Text response
Response: "2+2 equals 4."

User: "Hello"
Think: No action words → Text response  
Response: "Hello! How can I help?"

=== REMEMBER ===
- Action words = JSON tool (always)
- Questions = Text response
- If someone asks you to open/go to/navigate to ANYTHING, use a tool
- Never fake completion with text"""

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
            # Ensure tool_calls are JSON-serializable
            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                if hasattr(tool_calls[0], '__dict__'):  # It's an object, not a dict
                    # Convert to dict format
                    serialized_tool_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            serialized_tool_calls.append(tc)
                        else:
                            serialized_tc = {
                                'id': getattr(tc, 'id', ''),
                                'type': getattr(tc, 'type', 'function'),
                                'function': {
                                    'name': tc.function.name if hasattr(tc, 'function') else '',
                                    'arguments': tc.function.arguments if hasattr(tc, 'function') else '{}'
                                }
                            }
                            serialized_tool_calls.append(serialized_tc)
                    msg["tool_calls"] = serialized_tool_calls
                else:
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