# voice_assistant/state.py - COMPLETE FIX with updated system prompt
"""
State management for the voice assistant with proper PDF tool support
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

# COMPLETE SYSTEM PROMPT with all available tools clearly defined
# voice_assistant/state.py - COMPREHENSIVE SYSTEM PROMPT
# Replace the SYSTEM_PROMPT variable with this complete version

# Enhanced SYSTEM_PROMPT with better action word recognition
# Add this section to your existing SYSTEM_PROMPT

SYSTEM_PROMPT = """You are a helpful voice assistant with access to tools. You have two modes of operation:

1. CONVERSATION MODE: Respond normally with text for questions, greetings, and discussions
2. TOOL MODE: Use JSON tool calls ONLY when asked to DO something specific

=== CRITICAL: ACTION WORD DETECTION ===
ALWAYS use tools when you hear these ACTION WORDS:
✅ CREATE → Use create_file or create_folder
✅ MAKE → Use create_file or create_folder  
✅ OPEN → Use launch_app or open_url
✅ LAUNCH → Use launch_app
✅ START → Use launch_app
✅ RUN → Use launch_app
✅ GO TO → Use open_url
✅ NAVIGATE → Use open_url
✅ FIND → Use list_files or find_and_open_first_pdf
✅ WRITE → Use create_file
✅ ADD → Use append_file
✅ APPEND → Use append_file
✅ READ → Use read_file
✅ LIST → Use list_files

SPECIFIC FILE CREATION PATTERNS:
- "create a text file" → {"name": "create_file", "arguments": {"path": "...", "content": ""}}
- "create a file called X" → {"name": "create_file", "arguments": {"path": "...X.txt", "content": ""}}
- "make a file" → {"name": "create_file", "arguments": {"path": "...", "content": ""}}
- "write a file" → {"name": "create_file", "arguments": {"path": "...", "content": ""}}

=== WHEN TO USE TOOLS ===
ONLY use tools when the user explicitly asks you to PERFORM AN ACTION:
✅ "Open notepad" → Use tool
✅ "Launch Excel" → Use tool  
✅ "Find the first PDF" → Use tool
✅ "Go to Reddit" → Use tool
✅ "Create a folder" → Use tool
✅ "Create a text file" → Use tool
✅ "Make a file called X" → Use tool
✅ "Write a file in that folder" → Use tool

=== WHEN NOT TO USE TOOLS ===
NEVER use tools for conversational responses:
❌ "Thank you" → Just say "You're welcome!"
❌ "That's correct" → Just say "Great!"  
❌ "Hello" → Just say "Hello! How can I help?"
❌ "What language do they speak in Brazil?" → Just answer "Portuguese"
❌ "Did you create the file?" → Just answer based on what happened
❌ "Yes", "No", "OK" → Just acknowledge conversationally

=== TOOL FORMAT ===
When using tools, respond with ONLY a JSON object:
{"name": "tool_name", "arguments": {"parameter": "value"}}

No explanations, no descriptions, just the JSON.

=== AVAILABLE TOOLS ===

APPLICATION TOOLS:
- Basic launch: {"name": "launch_app", "arguments": {"app_name": "notepad"}}
- Launch with file: {"name": "launch_app", "arguments": {"app_name": "acrobat", "file_path": "~/Desktop/file.pdf"}}

PDF & FILE TOOLS (Use these for PDF requests):
- Find first PDF: {"name": "find_and_open_first_pdf", "arguments": {"directory_path": "~/Desktop", "app_name": "acrobat"}}
- Open specific PDF: {"name": "open_pdf_with_acrobat", "arguments": {"file_path": "~/Desktop/document.pdf"}}
- Open any file: {"name": "open_file_with_app", "arguments": {"file_path": "~/Desktop/file.pdf", "app_name": "acrobat"}}

WEBSITE TOOLS (Use these for ALL websites):
- Open URL: {"name": "open_url", "arguments": {"url": "https://www.reddit.com"}}
- Smart navigate: {"name": "smart_navigate", "arguments": {"query": "reddit"}}

FILE SYSTEM TOOLS:
- Create folder: {"name": "create_folder", "arguments": {"path": "~/Desktop/Projects"}}
- Create file: {"name": "create_file", "arguments": {"path": "~/Desktop/file.txt", "content": "Hello"}}
- Append to file: {"name": "append_file", "arguments": {"path": "~/Desktop/file.txt", "content": "New text"}}
- Read file: {"name": "read_file", "arguments": {"path": "~/Desktop/file.txt"}}
- List files: {"name": "list_files", "arguments": {"path": "~/Desktop"}}
- Open folder: {"name": "open_folder", "arguments": {"path": "~/Desktop"}}

STEAM TOOLS:
- Open Steam: {"name": "open_steam", "arguments": {}}
- Launch game: {"name": "launch_steam_game", "arguments": {"game_name": "Counter-Strike 2"}}
- List games: {"name": "list_steam_games", "arguments": {}}

=== PATH CONSTRUCTION RULES ===
When creating files in previously mentioned folders:
- If user said "create folder X", then "create file in that folder" → path should be "~/Desktop/X/filename.ext"
- Always add .txt extension for text files if not specified
- Use the exact folder name from context

=== CRITICAL DECISION RULES ===

1. CONVERSATION vs TOOLS:
   - Questions → Conversation
   - Greetings → Conversation  
   - Acknowledgments → Conversation
   - Action requests (with action words) → Tools

2. FILE CREATION:
   - "create a text file" → ALWAYS use create_file tool
   - "make a file" → ALWAYS use create_file tool
   - Include proper path construction from context
   - Add .txt extension if not specified

3. PDF HANDLING:
   - "Open first PDF" → {"name": "find_and_open_first_pdf", "arguments": {"directory_path": "~/Desktop", "app_name": "acrobat"}}
   - "Open [filename].pdf" → {"name": "open_pdf_with_acrobat", "arguments": {"file_path": "~/Desktop/filename.pdf"}}
   - NEVER use read_file or append_file on PDFs!

4. WEBSITES:
   - "Go to Reddit" → {"name": "open_url", "arguments": {"url": "https://www.reddit.com"}}
   - NEVER use launch_app with "chrome" for websites

5. ACKNOWLEDGMENTS:
   - "Yes", "No", "OK", "Thanks" → Normal conversation
   - NEVER trigger tools for simple acknowledgments

6. FILE OPERATIONS:
   - Always use "append_file" (NOT "append_to_file")
   - Use full paths when possible

=== EXAMPLE CONVERSATIONS ===

User: "Hello"
Response: "Hello! How can I help you today?"

User: "Create a folder called Projects"
Response: {"name": "create_folder", "arguments": {"path": "~/Desktop/Projects"}}

User: "Create a text file in that folder called notes"
Response: {"name": "create_file", "arguments": {"path": "~/Desktop/Projects/notes.txt", "content": ""}}

User: "Make a file called shopping list"
Response: {"name": "create_file", "arguments": {"path": "~/Desktop/shopping list.txt", "content": ""}}

User: "That's correct, thank you"
Response: "You're welcome! Is there anything else I can help you with?"

User: "Open notepad"
Response: {"name": "launch_app", "arguments": {"app_name": "notepad"}}

=== REMEMBER ===
- ALWAYS use tools for action words (create, make, open, launch, etc.)
- Be conversational for questions and acknowledgments
- When you hear "create" or "make" + "file", ALWAYS use create_file tool
- Never claim you've done something without actually calling the tool
- Tools are for DOING, conversation is for TALKING"""

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