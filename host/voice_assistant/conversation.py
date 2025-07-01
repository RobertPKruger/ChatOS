# voice_assistant/conversation.py
"""
Main conversation loop and management

The LLM’s text output touches the speech layer in exactly one place:

process_user_input()

Builds assistant_response (the string that comes back from the Mistral / OpenAI chat call).

Returns that string to the caller.

conversation_loop()


assistant_response = await self.process_user_input(user_text, mcp_client, tools)

# ───► this is the only point where the text response is handed off to TTS
if assistant_response and not self.state.interrupt_flag.is_set() \
   and self.state.get_mode() != AssistantMode.STUCK_CHECK:
    await speak_text(assistant_response, self.state, self.config)
speak_text() (imported from voice_assistant/speech.py) is your text-to-speech gateway. Whatever engine you configure there—Coqui TTS “Nova”, a cloud TTS, etc.—will synthesize assistant_response and play or stream it.

The same helper is called for system messages (greetings, error notices, stuck-mode wake-ups), but the only place the LLM’s inference is voiced is in that await speak_text(assistant_response, …) block.

So to get the Mistral 7B output to come back in Nova’s voice:

Wire the Coqui TTS Nova model inside speech.py’s speak_text() (or whichever function it delegates to).

Make sure self.config fields (e.g., tts_model, tts_voice, audio device) point at Nova.

Nothing else in the conversation manager needs to change—the handoff already happens through that single call.
"""

import asyncio
import logging
from typing import Optional
from openai import OpenAI

from .state import AssistantState, AssistantMode
from .audio import ContinuousAudioRecorder
from .speech import transcribe_audio, speak_text
from .mcp_client import get_mcp_client, get_tools, call_tool_with_timeout, shutdown_mcp_server
from .config import Config

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages the main conversation loop and interactions"""
    
    def __init__(self, config: Config, state: AssistantState, audio_recorder: ContinuousAudioRecorder):
        self.config = config
        self.state = state
        self.audio_recorder = audio_recorder
        self.stuck_task: Optional[asyncio.Task] = None
        
    async def stuck_detection_task(self):
        """Background task to check if assistant is stuck and listen for wake phrase"""
        while self.state.running:
            try:
                # Check every few seconds
                await asyncio.sleep(self.config.stuck_check_interval)
                
                # Only check if we're in processing mode and stuck
                if self.state.is_stuck(self.config.processing_timeout):
                    logger.warning(f"Assistant appears stuck (processing for {self.config.processing_timeout}s)")
                    self.state.set_mode(AssistantMode.STUCK_CHECK)
                    
                    # Try to detect the wake phrase
                    audio_buffer = await self.audio_recorder.record_until_silence(
                        self.state, self.config, check_stuck_phrase=True
                    )
                    
                    if audio_buffer:
                        # Quick transcription check for wake phrase
                        try:
                            text = await transcribe_audio(
                                audio_buffer, self.state, self.config, check_stuck_phrase=True
                            )
                            
                            if text:
                                text_normalized = text.strip().lower().replace(",", "").replace("?", "")
                                logger.info(f"Stuck check transcription: {text_normalized}")
                                
                                # Check if it matches our wake phrase (fuzzy match)
                                wake_words = set(self.config.stuck_phrase.split())
                                detected_words = set(text_normalized.split())
                                if len(wake_words.intersection(detected_words)) >= 2:  # At least 2 matching words
                                    logger.info("Wake phrase detected! Resetting to listening mode")
                                    self.state.interrupt_flag.set()
                                    self.state.set_mode(AssistantMode.LISTENING)
                                    # Speak acknowledgment
                                    asyncio.create_task(
                                        speak_text("I'm back! Sorry about that. How can I help you?", 
                                                 self.state, self.config)
                                    )
                        except Exception as e:
                            logger.error(f"Error in stuck phrase detection: {e}")
                    
            except Exception as e:
                logger.error(f"Error in stuck detection task: {e}")
                await asyncio.sleep(1)
    
    async def handle_special_commands(self, text: str) -> bool:
        """Handle special commands and return True if handled"""
        lower_text = text.lower().strip()
        
        if lower_text in {"reset chat", "new chat", "clear history"}:
            self.state.reset_conversation()
            await speak_text("Starting a new conversation.", self.state, self.config)
            return True
        
        if any(phrase in lower_text for phrase in ["exit", "quit", "goodbye", "shut down", "shutdown"]):
            await speak_text("Goodbye! Shutting down the system...", self.state, self.config)
            await self.shutdown_system()
            return True
            
        return False
    
    async def process_user_input(self, user_text: str, mcp_client, tools):
        """Process user input and generate response"""
        # Add user message to history
        self.state.conversation_history.append({"role": "user", "content": user_text})
        
        try:
            completion = self.state.openai_client.chat.completions.create(
                model=self.config.chat_model,
                messages=self.state.conversation_history,
                tools=tools,
                tool_choice="auto"
            )
            
            choice = completion.choices[0]
            message = choice.message
            
            assistant_response = ""
            
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                # Handle tool calls
                self.state.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    # Check if we should interrupt
                    if self.state.interrupt_flag.is_set() or self.state.get_mode() == AssistantMode.STUCK_CHECK:
                        logger.info("Processing interrupted")
                        self.state.set_mode(AssistantMode.LISTENING)
                        return None
                        
                    tool_result = await call_tool_with_timeout(
                        mcp_client, tool_call, self.config.tool_timeout
                    )
                    
                    # Add tool result to history
                    self.state.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # Get the model's follow-up response if not interrupted
                if not self.state.interrupt_flag.is_set() and self.state.get_mode() != AssistantMode.STUCK_CHECK:
                    follow_up = self.state.openai_client.chat.completions.create(
                        model=self.config.chat_model,
                        messages=self.state.conversation_history,
                        tools=tools,
                        tool_choice="auto"
                    )
                    
                    follow_up_message = follow_up.choices[0].message
                    assistant_response = follow_up_message.content or "Task completed."
                    
                    self.state.conversation_history.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
            else:
                # Regular text response
                assistant_response = message.content or "I'm not sure how to respond to that."
                self.state.conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return "I encountered an error processing your request. Please try again."
    
    async def conversation_loop(self):
        """Main conversation loop with improved state management"""
        logger.info("Starting conversation loop with mode-aware listening")
        
        # Initialize conversation
        self.state.reset_conversation()
        
        # Initialize OpenAI client
        if not self.state.openai_client:
            try:
                self.state.openai_client = OpenAI(api_key=self.config.openai_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
        
        # Start continuous audio recording
        self.audio_recorder.start()
        
        # Start stuck detection task
        self.stuck_task = asyncio.create_task(self.stuck_detection_task())
        
        async with get_mcp_client(self.state) as mcp_client:
            self.state.mcp_client = mcp_client
            
            # Load tools
            tools = await get_tools(mcp_client, self.state)
            
            # Initial greeting
            await speak_text("Hello! I'm listening.", self.state, self.config)
            
            while self.state.running:
                try:
                    # Only record in listening mode
                    if self.state.get_mode() != AssistantMode.LISTENING:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Record audio until silence
                    audio_buffer = await self.audio_recorder.record_until_silence(
                        self.state, self.config
                    )
                    
                    if not audio_buffer:
                        continue
                    
                    # Switch to recording mode
                    self.state.set_mode(AssistantMode.RECORDING)
                    
                    # Transcribe
                    user_text = await transcribe_audio(audio_buffer, self.state, self.config)
                    if not user_text:
                        self.state.set_mode(AssistantMode.LISTENING)
                        continue
                    
                    logger.info(f"User said: {user_text}")
                    
                    # Switch to processing mode
                    self.state.set_mode(AssistantMode.PROCESSING)
                    
                    # Handle special commands
                    if await self.handle_special_commands(user_text):
                        if not self.state.running:  # Exit was called
                            break
                        continue
                    
                    # Process user input
                    assistant_response = await self.process_user_input(user_text, mcp_client, tools)
                    
                    # Speak the response if not interrupted
                    if assistant_response and not self.state.interrupt_flag.is_set() and \
                       self.state.get_mode() != AssistantMode.STUCK_CHECK:
                        await speak_text(assistant_response, self.state, self.config)
                    else:
                        logger.info("Skipping speech due to interruption or no response")
                        self.state.set_mode(AssistantMode.LISTENING)
                        
                except Exception as e:
                    logger.error(f"Error in conversation loop: {e}")
                    self.state.set_mode(AssistantMode.ERROR)
                    await asyncio.sleep(self.config.reconnect_delay)
                    self.state.set_mode(AssistantMode.LISTENING)
                    continue
        
        # Cancel stuck detection task
        if self.stuck_task:
            self.stuck_task.cancel()
            try:
                await self.stuck_task
            except asyncio.CancelledError:
                pass
    
    async def shutdown_system(self):
        """Properly shut down both server and client"""
        logger.info("Initiating system shutdown...")
        
        # Stop audio recording
        self.audio_recorder.stop()
        
        # Shutdown MCP
        await shutdown_mcp_server(self.state)
        
        self.state.running = False
        logger.info("System shutdown complete")