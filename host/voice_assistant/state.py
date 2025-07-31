# voice_assistant/state.py - ENHANCED WITH DICTATION MODE
"""
State management for the voice assistant with dictation mode support
"""

import threading
import time
import logging
from enum import Enum
from typing import Optional, List, Dict, Any
import subprocess
import asyncio
import os
from pathlib import Path

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

from fastmcp import Client
from .model_providers.base import TranscriptionProvider, ChatCompletionProvider, TextToSpeechProvider

logger = logging.getLogger(__name__)

# Enhanced system prompt with dictation mode instructions
SYSTEM_PROMPT = """You are a helpful voice assistant with access to tools. You MUST use tools for action requests.

=== CRITICAL: ALWAYS USE TOOLS FOR ACTIONS ===

When users ask you to DO something, you MUST call the appropriate tool. NEVER just say "I've completed that task" without actually calling a tool.

ACTION WORDS THAT REQUIRE TOOLS:
✅ "go to" + website → MUST use open_url tool
✅ "open" + app → MUST use launch_app tool  
✅ "current stock price" → MUST use get_current_stock_price tool
✅ "current weather" → MUST use get_weather tool
✅ "search the web" → MUST use web_search tool
✅ "latest news" → MUST use search_news tool
✅ "create file" → MUST use create_file tool

=== DICTATION MODE SUPPORT ===

The system supports a special "Dictation Mode" with these behaviors:
- User can enter dictation mode by saying phrases like "take dictation", "start dictation", "please take dictation"
- In dictation mode, ALL user speech is captured as text (no tool calls or responses)
- The ONLY command recognized in dictation mode is "end dictation" or "stop dictation"
- When dictation ends, the system saves the captured text to a file

=== WEBSITE/URL REQUESTS ===
For ANY website request, you MUST use the open_url tool:

User: "Please go to nuggetnews.com"
CORRECT: {"name": "open_url", "arguments": {"url": "https://nuggetnews.com"}}
WRONG: Just saying "I've completed that task"

User: "Go to Amazon" 
CORRECT: {"name": "open_url", "arguments": {"url": "https://www.amazon.com"}}

User: "Please go to Reddit"
CORRECT: {"name": "open_url", "arguments": {"url": "https://www.reddit.com"}}

=== CURRENT INFORMATION REQUESTS ===
For current/real-time info, you MUST use web search tools:

User: "Tell me the current stock price of Nvidia"
CORRECT: {"name": "get_current_stock_price", "arguments": {"symbol": "NVDA"}}
WRONG: Just saying "I've completed that task"

User: "What's the current weather in New York?"
CORRECT: {"name": "get_weather", "arguments": {"location": "New York"}}

User: "Search the web for latest AI news"
CORRECT: {"name": "search_news", "arguments": {"query": "AI artificial intelligence"}}

=== AVAILABLE TOOLS ===

WEB & CURRENT INFO TOOLS:
- open_url: Open any website
- get_current_stock_price: Get real-time stock prices  
- web_search: Search web for current information
- search_news: Get latest news
- get_weather: Get current weather
- search_definition: Define terms

APPLICATION TOOLS:
- launch_app: Open applications like Excel, Word, Chrome
- open_file_with_app: Open files with specific apps

**Applications:** {"name": "launch_app", "arguments": {"app_name": "notepad"}}
**Websites:** {"name": "open_url", "arguments": {"url": "https://site.com"}}
**Steam Games:** {"name": "launch_steam_game", "arguments": {"game_name": "game"}}

FILE SYSTEM TOOLS:
- create_file: Create new files
- create_folder: Create directories  
- list_files: List directory contents
- read_file: Read file contents
**Files:** {"name": "create_file", "arguments": {"path": "file.txt", "content": ""}}

=== RESPONSE FORMAT ===
When users ask you to DO something, respond with ONLY the tool call JSON:
{"name": "tool_name", "arguments": {"parameter": "value"}}

Do NOT add explanations like "I'll help you with that" before the tool call.
Do NOT say "I've completed that task" without actually calling a tool.

=== CONVERSATION MODE ===
Only use conversation mode for:
- Greetings: "Hello" → "Hello! How can I help you?"
- Questions about general knowledge: "What is Python?" → Explain programming language
- Thank you/acknowledgments: "Thank you" → "You're welcome!"

=== CRITICAL RULES ===
1. If user says "go to [website]" → ALWAYS call open_url tool
2. If user asks for "current [anything]" → ALWAYS call appropriate web search tool
3. If user says "open [app]" → ALWAYS call launch_app tool
4. NEVER say "I've completed that task" without calling a tool
5. NEVER give generic responses to action requests

=== EXAMPLES ===

❌ WRONG:
User: "Please go to nuggetnews.com"
Assistant: "I've completed that task for you."

✅ CORRECT:
User: "Please go to nuggetnews.com"  
Assistant: {"name": "open_url", "arguments": {"url": "https://nuggetnews.com"}}

❌ WRONG:
User: "Tell me the current stock price of Nvidia"
Assistant: "I've completed that task for you."

✅ CORRECT:
User: "Tell me the current stock price of Nvidia"
Assistant: {"name": "get_current_stock_price", "arguments": {"symbol": "NVDA"}}

Remember: You have tools for a reason. USE THEM when users ask you to do something!

=== FOLLOW-UP RESPONSES ===
CRITICAL: After executing tools, you MUST provide a text summary, NOT more tool calls.

When you've just executed a tool:
✅ CORRECT: "The current stock price of Nvidia is $543.21, up 2.5% from yesterday."
✅ CORRECT: "I've opened nuggetnews.com in your browser."
✅ CORRECT: "The weather in Central Oregon is 72°F with sunny skies."

❌ WRONG: Calling more tools after you just executed tools
❌ WRONG: Responding with JSON when the user needs spoken information

=== TOOL RESULT INTERPRETATION ===
When tools return data, summarize it conversationally:

Stock Price Results → "The current stock price of [symbol] is [price]"
Weather Results → "The weather in [location] is [temperature] with [conditions]"
Website Opening → "I've opened [website] for you"
App Launch → "I've launched [app]"

Remember: Users want to HEAR the results, not just know that tools were executed!"""


class AssistantMode(Enum):
    """Assistant operational modes"""
    LISTENING = "listening"          # Actively listening for commands
    RECORDING = "recording"          # Recording user speech
    PROCESSING = "processing"        # Processing with LLM
    SPEAKING = "speaking"            # Playing TTS response
    STUCK_CHECK = "stuck_check"      # Listening only for wake phrase
    ERROR = "error"                  # Error state
    SLEEPING = "sleeping"            # In sleep mode, not listening
    DICTATION = "dictation"          # In dictation mode, capturing text

class AssistantState:
    """Global state management for the assistant with dictation support"""
    
    def __init__(self, vad_aggressiveness: int = 3):
        self.running = True
        self.mode = AssistantMode.LISTENING
        self.mode_lock = threading.Lock()
        self.processing_start_time: Optional[float] = None
        self.interrupt_flag = threading.Event()
        
        # Dictation mode state
        self.dictation_buffer: List[str] = []
        self.dictation_start_time: Optional[float] = None
        
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
            logger.info(f"Mode transition: {old_mode.value} -> {new_mode.value}")
            
            # Start processing timer
            if new_mode == AssistantMode.PROCESSING:
                self.processing_start_time = time.time()
            elif old_mode == AssistantMode.PROCESSING:
                self.processing_start_time = None
                
            # Handle dictation mode transitions
            if new_mode == AssistantMode.DICTATION:
                self.start_dictation()
            elif old_mode == AssistantMode.DICTATION and new_mode != AssistantMode.DICTATION:
                # Dictation ended - don't auto-save here, let conversation handler do it
                pass
    
    def get_mode(self) -> AssistantMode:
        """Thread-safe mode getter"""
        with self.mode_lock:
            return self.mode
    
    def is_stuck(self, timeout: float) -> bool:
        """Check if we've been processing too long"""
        if self.processing_start_time and self.get_mode() == AssistantMode.PROCESSING:
            return (time.time() - self.processing_start_time) > timeout
        return False
    
    def is_dictation_mode(self) -> bool:
        """Check if currently in dictation mode"""
        return self.get_mode() == AssistantMode.DICTATION
    
    def start_dictation(self):
        """Start dictation mode"""
        self.dictation_buffer = []
        self.dictation_start_time = time.time()
        logger.info("Dictation mode started")
    
    def add_dictation_text(self, text: str):
        """Add text to dictation buffer"""
        if self.is_dictation_mode():
            self.dictation_buffer.append(text)
            logger.info(f"Added to dictation: {text}")
            # Also print to console for real-time feedback
            print(f"📝 Dictation: {text}")
    
    def get_dictation_content(self) -> str:
        """Get the complete dictation content"""
        return "\n".join(self.dictation_buffer)
    
    def clear_dictation_buffer(self):
        """Clear the dictation buffer"""
        self.dictation_buffer = []
        self.dictation_start_time = None
    
    def get_dictation_stats(self) -> Dict[str, Any]:
        """Get dictation session statistics"""
        if not self.dictation_start_time:
            return {}
        
        duration = time.time() - self.dictation_start_time
        word_count = len(self.get_dictation_content().split()) if self.dictation_buffer else 0
        
        return {
            "duration_seconds": duration,
            "duration_formatted": f"{int(duration//60)}m {int(duration%60)}s",
            "word_count": word_count,
            "line_count": len(self.dictation_buffer),
            "character_count": len(self.get_dictation_content())
        }
    
    def save_dictation_to_file(self, filename: str, directory: Optional[str] = None) -> str:
        """Save dictation content to a file"""
        try:
            # Determine save directory
            if directory is None:
                # Use environment variable or default to Desktop
                directory = os.getenv("CHATOS_DICTATION_DIR", "~/Desktop")
            
            # Expand user path
            directory = os.path.expanduser(directory)
            directory_path = Path(directory)
            
            # Create directory if it doesn't exist
            directory_path.mkdir(parents=True, exist_ok=True)
            
            # Ensure filename has .txt extension
            if not filename.lower().endswith('.txt'):
                filename += '.txt'
            
            # Create full file path
            file_path = directory_path / filename
            
            # Get dictation content and stats
            content = self.get_dictation_content()
            stats = self.get_dictation_stats()
            
            # Create file header with metadata
            header = f"""Dictation Session - {time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {stats.get('duration_formatted', 'Unknown')}
Words: {stats.get('word_count', 0)}
Lines: {stats.get('line_count', 0)}
Characters: {stats.get('character_count', 0)}

--- Begin Dictation ---

"""
            
            footer = f"""

--- End Dictation ---
Saved: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(header + content + footer)
            
            logger.info(f"Dictation saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            error_msg = f"Failed to save dictation: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("Conversation history reset")

    MAX_HISTORY_TOKENS = 4000  # Approximate token limit

    def trim_conversation_history(self):
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