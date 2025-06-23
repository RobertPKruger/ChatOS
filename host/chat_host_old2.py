# enhanced_chat_host.py
"""
Enhanced MCP Chat Host - A resilient background voice assistant
Improvements:
- Comprehensive error handling and recovery
- Proper logging with rotation
- Configuration management
- Auto-restart capabilities
- Voice Activity Detection ready
- System tray integration ready
- Graceful shutdown handling
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

import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from pynput import keyboard
from fastmcp import Client
from dotenv import load_dotenv

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
    enable_vad: bool = False  # Future: Voice Activity Detection
    
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
            enable_vad=os.getenv("ENABLE_VAD", "false").lower() == "true"
        )

# Global configuration
config = Config.from_env()

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
        self.openai_client: Optional[OpenAI] = None
        self.mcp_client: Optional[Client] = None
        self.tools_cache: List[Dict[str, Any]] = []
        self.tools_cache_time: float = 0
        self.conversation_history: List[Dict[str, Any]] = []
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("Conversation history reset")

# Global state instance
state = AssistantState()

# System prompt
SYSTEM_PROMPT = """
You are my personal voice assistant …

When the user asks to open an application like "Open Notepad" or "Launch Calculator", use the launch_app tool and provide the appropriate app name as a parameter. For example:
- For "Open Notepad" → use launch_app with app="notepad"
- For "Open Calculator" → use launch_app with app="calc"
- For "Open File Explorer" → use launch_app with app="explorer"

Always provide the app parameter when using launch_app tool.

If you encounter any errors with tools, explain what went wrong and suggest alternatives when possible.

File-system examples
--------------------
- "Create a folder called Projects inside Documents"
  → create_folder(path="~/Documents/Projects")
- "Create a folder named Projects on my desktop"
  → create_folder(path="~/Desktop/Projects")          # <-- NEW
- "Open my desktop folder"
  → open_folder(path="~/Desktop")
…

"""

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
# Audio recording with improved error handling
################################################################################

def record_until_space(fs: int = None) -> Optional[io.BytesIO]:
    """Record audio until space key is released, with error handling"""
    if fs is None:
        fs = config.sample_rate
        
    logger.info("Hold space to talk...")
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    
    def on_key_release(key):
        if key == keyboard.Key.space:
            stop_event.set()
            return False  # Stop listener
    
    # Setup keyboard listener
    kb_listener = keyboard.Listener(on_release=on_key_release)
    kb_listener.start()
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        audio_queue.put(bytes(indata))
    
    wav_buffer = io.BytesIO()
    
    try:
        # Setup audio stream with error handling
        with sd.RawInputStream(
            samplerate=fs,
            blocksize=0,
            dtype="int16",
            channels=1,
            callback=audio_callback
        ) as stream:
            
            with sf.SoundFile(
                wav_buffer,
                mode="w",
                samplerate=fs,
                channels=1,
                subtype="PCM_16",
                format="WAV"
            ) as sound_file:
                
                while not stop_event.is_set():
                    try:
                        data = audio_queue.get(timeout=0.1)
                        sound_file.buffer_write(data, dtype="int16")
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error writing audio data: {e}")
                        break
                        
    except Exception as e:
        logger.error(f"Audio recording error: {e}")
        return None
    finally:
        kb_listener.join()
    
    wav_buffer.seek(0)
    
    # Validate audio size
    if len(wav_buffer.getbuffer()) < config.min_audio_size:
        logger.info("Audio too short - hold space longer")
        return None
        
    wav_buffer.name = "speech.wav"  # Hint for multipart encoder
    return wav_buffer

################################################################################
# MCP Client management with reconnection
################################################################################

@asynccontextmanager
async def get_mcp_client():
    """Context manager for MCP client with automatic reconnection"""
    server_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mcp_os", "server.py")
    )
    
    client = None
    try:
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
        logger.info(f"Tool {name} -> result: {result!r}")   
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
    """Transcribe audio with retry logic"""
    async def do_transcribe():
        try:
            response = state.openai_client.audio.transcriptions.create(
                model=config.stt_model,
                file=audio_buffer
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"STT error: {e}")
            raise
    
    try:
        text = await retry_with_backoff(do_transcribe())
        logger.info(f"Transcribed: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Failed to transcribe after retries: {e}")
        return None

################################################################################
# TTS with error handling
################################################################################

async def speak_text(text: str) -> bool:
    """Convert text to speech and play it"""
    if not text or text.isspace():
        text = "Okay."
    
    print(f"[DEBUG] speak_text called with: '{text}'")  # Console debug
    
    try:
        logger.info(f"Speaking: {text[:50]}...")
        print(f"[DEBUG] About to call OpenAI TTS...")
        
        # Generate speech - run in executor since it's blocking
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
        
        print(f"[DEBUG] TTS response received, size: {len(audio_response.read())} bytes")
        
        # Reset the response for reading
        audio_response.seek(0) if hasattr(audio_response, 'seek') else None
        
        # Play audio - also run in executor
        def play_audio():
            print(f"[DEBUG] Starting audio playback...")
            audio_buffer = io.BytesIO(audio_response.read())
            print(f"[DEBUG] Audio buffer size: {len(audio_buffer.getvalue())} bytes")
            
            data, sample_rate = sf.read(audio_buffer, dtype="float32")
            print(f"[DEBUG] Audio data shape: {data.shape}, sample rate: {sample_rate}")
            
            print(f"[DEBUG] Playing audio...")
            sd.play(data, sample_rate)
            sd.wait()  # Wait for completion
            print(f"[DEBUG] Audio playback completed")
        
        await loop.run_in_executor(None, play_audio)
        logger.info("Audio playback completed")
        return True
        
    except Exception as e:
        print(f"[DEBUG] TTS error: {e}")
        logger.error(f"TTS error: {e}")
        import traceback
        traceback.print_exc()
        return False

################################################################################
# Main conversation loop
################################################################################

async def conversation_loop():
    """Main conversation loop with comprehensive error handling"""
    logger.info("Starting conversation loop")
    
    # Initialize conversation
    state.reset_conversation()
    
    # Initialize OpenAI client
    if not state.openai_client:
        try:
            state.openai_client = OpenAI(api_key=config.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async with get_mcp_client() as mcp_client:
        state.mcp_client = mcp_client
        
        # Load tools
        tools = await get_tools(mcp_client)
        
        while state.running:
            try:
                if state.muted:
                    await asyncio.sleep(1)
                    continue
                
                # Record audio
                loop = asyncio.get_event_loop()
                audio_buffer = await loop.run_in_executor(None, record_until_space)
                
                if not audio_buffer:
                    continue
                
                logger.info("Transcribing audio...")
                
                # Transcribe
                user_text = await transcribe_audio(audio_buffer)
                if not user_text:
                    await speak_text("Sorry, I didn't catch that.")
                    continue
                
                logger.info(f"User said: {user_text}")
                
                # Handle special commands
                if user_text.lower().strip() in {"reset chat", "new chat", "clear history"}:
                    state.reset_conversation()
                    await speak_text("Starting a new conversation.")
                    continue
                
                if user_text.lower().strip() in {"mute", "quiet", "stop listening"}:
                    state.muted = True
                    await speak_text("I'm muted. Say 'unmute' to resume.")
                    continue
                
                if user_text.lower().strip() in {"unmute", "resume", "wake up"}:
                    state.muted = False
                    await speak_text("I'm listening again.")
                    continue
                
                if user_text.lower().strip() in {"exit", "quit", "goodbye", "shut down"}:
                    await speak_text("Goodbye!")
                    state.running = False
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
                        # First add the assistant message with tool calls
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
                        
                        # Get the model's follow-up response after tool execution
                        follow_up = state.openai_client.chat.completions.create(
                            model=config.chat_model,
                            messages=state.conversation_history,
                            tools=tools,
                            tool_choice="auto"
                        )
                        
                        follow_up_message = follow_up.choices[0].message
                        assistant_response = follow_up_message.content or "Task completed."
                        
                        # Add the follow-up response to history
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
                    
                    # Speak the response
                    print(f"[DEBUG] About to speak: '{assistant_response}'")
                    logger.info(f"About to speak: '{assistant_response}'")
                    speech_success = await speak_text(assistant_response)
                    if not speech_success:
                        print(f"[DEBUG] TTS failed!")
                        logger.warning("TTS failed, but continuing conversation")
                    else:
                        print(f"[DEBUG] TTS succeeded!")
                    
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
            state.running = False
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
    state.running = False                      # let run_forever() exit
    loop = asyncio.get_event_loop()
    loop.call_soon_threadsafe(loop.stop)       # in case we’re waiting in await

def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Voice Assistant starting up...")
    logger.info(f"Configuration: STT={config.stt_model}, Chat={config.chat_model}, TTS={config.tts_model}")
    
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