# voice_assistant/conversation.py - FIXED WITH SIMPLE READABLE RESPONSE APPROACH
"""
Main conversation loop with improved text extraction and error handling
"""

import asyncio
import logging
from typing import Optional
from openai import OpenAI

from .state import AssistantState, AssistantMode
from .audio import ContinuousAudioRecorder
from .speech import transcribe_audio, speak_text
from .mcp_client import get_mcp_client, get_tools_cached, call_tool_with_timeout, shutdown_mcp_server
from .config import Config

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages the main conversation loop and interactions"""
    
    def __init__(self, config: Config, state: AssistantState, audio_recorder: ContinuousAudioRecorder):
        self.config = config
        self.state = state
        self.audio_recorder = audio_recorder
        self.stuck_task: Optional[asyncio.Task] = None

    def _is_response_readable(self, text: str) -> bool:
        """Check if a response is clean and readable for speech synthesis"""
        if not text or not isinstance(text, str):
            return False
        
        # Check for common unreadable patterns
        unreadable_patterns = [
            '[TextContent(',
            'TextContent(',
            'annotations=',
            'type=',
            "text='",
            'text="',
            ')]',
            '[{',
            '}]',
            '<class ',
            '<',
            '__dict__',
            'object at 0x'
        ]
        
        for pattern in unreadable_patterns:
            if pattern in text:
                return False
        
        # Check if it has too many special characters
        special_char_count = sum(1 for c in text if c in "[]{}()<>\"'=")
        if special_char_count > len(text) * 0.2:  # More than 20% special chars
            return False
        
        return True

    def _create_generic_response(self, tool_name: str, user_request: str) -> str:
        """Create a generic response based on the tool and user request"""
        # Extract key information from user request
        user_lower = user_request.lower()
        
        # Application launches
        if tool_name == "launch_app" or "open" in user_lower or "launch" in user_lower:
            # Try to extract app name from user request
            app_words = ["notepad", "calculator", "word", "excel", "steam", "gcai", "chrome", "explorer"]
            for app in app_words:
                if app in user_lower:
                    return f"I've opened {app} for you."
            return "I've opened the application for you."
        
        # Steam operations
        elif tool_name == "launch_steam_game":
            if "magic" in user_lower or "gathering" in user_lower:
                return "I've launched Magic: The Gathering Arena in Steam."
            elif "game" in user_lower:
                return "I've launched the game in Steam."
            return "I've launched the Steam game for you."
        
        elif tool_name == "list_steam_games":
            return "I've displayed your Steam games list."
        
        elif tool_name == "open_steam":
            return "I've opened Steam for you."
        
        # File operations
        elif tool_name == "create_folder":
            return "I've created the folder for you."
        
        elif tool_name == "open_folder":
            return "I've opened the folder in Explorer."
        
        elif tool_name == "list_files":
            return "I've listed the files for you."
        
        elif tool_name == "search_files":
            return "I've completed the file search."
        
        elif tool_name == "read_file":
            return "I've read the file contents."
        
        elif tool_name == "create_file":
            return "I've created the file for you."
        
        # Default response
        else:
            return "I've completed that task for you."

    def _clean_response_for_speech(self, response: str, tool_name: str = None, user_request: str = None) -> str:
        """Clean any response to ensure it's readable for speech synthesis"""
        # First try to extract clean text
        if not isinstance(response, str):
            response = self._extract_text_from_any_format(response)
        
        # If it's already clean and readable, return it
        if self._is_response_readable(response):
            return response
        
        # If not readable, create a generic response
        if tool_name and user_request:
            return self._create_generic_response(tool_name, user_request)
        
        # Fallback generic responses based on content hints
        response_lower = str(response).lower()
        
        if "launched" in response_lower or "opened" in response_lower:
            return "I've opened the application for you."
        elif "created" in response_lower:
            return "I've created that for you."
        elif "list" in response_lower or "found" in response_lower:
            return "I've completed the search and displayed the results."
        elif "error" in response_lower or "failed" in response_lower:
            return "I encountered an error with that request. Please try again."
        else:
            return "I've completed that task for you."

    def _extract_text_from_any_format(self, data):
        """Extract text from any data format - simplified version"""
        if data is None:
            return ""
        
        # Already a string
        if isinstance(data, str):
            return data.strip()
        
        # Handle TextContent objects (from MCP)
        if hasattr(data, 'text'):
            return str(data.text).strip()
        
        # Handle list of items
        if isinstance(data, list):
            if len(data) == 0:
                return ""
            
            text_parts = []
            for item in data:
                if hasattr(item, 'text'):
                    text_parts.append(str(item.text).strip())
                elif isinstance(item, str):
                    text_parts.append(item.strip())
                elif isinstance(item, dict) and 'text' in item:
                    text_parts.append(str(item['text']).strip())
                else:
                    # Convert to string as last resort
                    text_parts.append(str(item).strip())
            
            return " ".join(text_parts)
        
        # Dictionary
        if isinstance(data, dict):
            for key in ['text', 'content', 'message', 'result', 'data']:
                if key in data:
                    return str(data[key]).strip()
        
        # Last resort - convert to string
        return str(data).strip()
        
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

        if "use local model" in lower_text or "enable local" in lower_text:
            if not self.config.use_local_first:
                await speak_text("Switching to local-first mode. This will take effect on restart.", self.state, self.config)
                # Note: Would need restart to actually change providers
            else:
                await speak_text("Local-first mode is already enabled.", self.state, self.config)
            return True
            
        if "use cloud only" in lower_text or "disable local" in lower_text:
            if self.config.use_local_first:
                await speak_text("Cloud-only mode would take effect on restart.", self.state, self.config)
                # Note: Would need restart to actually change providers
            else:
                await speak_text("Already using cloud-only mode.", self.state, self.config)
            return True
        
        # Sleep commands
        if any(phrase in lower_text for phrase in ["go to sleep", "sleep mode", "sleep now", "go sleep", "sleep"]):
            await speak_text("Going to sleep. Say 'wake up' or 'hello' to wake me.", self.state, self.config)
            self.state.set_mode(AssistantMode.SLEEPING)
            logger.info("System entering sleep mode")
            return True
        
        # Wake commands (only processed when sleeping)
        if self.state.get_mode() == AssistantMode.SLEEPING:
            if any(phrase in lower_text for phrase in ["wake up", "wake", "hello", "hey", "wake me up"]):
                self.state.set_mode(AssistantMode.LISTENING)
                await speak_text("I'm awake! How can I help you?", self.state, self.config)
                logger.info("System waking up from sleep mode")
                return True
            else:
                # In sleep mode, ignore all other commands
                logger.debug(f"Ignoring command while sleeping: {text}")
                return True
        
        # Existing commands (only when not sleeping)
        if lower_text in {"reset chat", "new chat", "clear history"}:
            self.state.reset_conversation()
            await speak_text("Starting a new conversation.", self.state, self.config)
            return True
        
        if any(phrase in lower_text for phrase in ["exit", "quit", "goodbye", "shut down", "shutdown"]):
            await speak_text("Goodbye! Shutting down the system...", self.state, self.config)
            await self.shutdown_system()
            return True
            
        return False
    
    def _should_use_tools_parameter(self, provider) -> bool:
        """Determine if the provider supports OpenAI-style tools parameter"""
        # Only use tools parameter for OpenAI-compatible providers
        # Local models (Ollama) should use text-based tool calling
        provider_class_name = provider.__class__.__name__
        
        if hasattr(provider, 'primary'):
            # This is a FailoverChatProvider - check the primary provider
            primary_class_name = provider.primary.__class__.__name__
            return primary_class_name in ["OpenAIChatProvider", "OpenAIChatCompletionProvider"]
        
        return provider_class_name in ["OpenAIChatProvider", "OpenAIChatCompletionProvider"]
    
    def _detect_poor_tool_response(self, response_content: str, has_tool_results: bool = False) -> bool:
        """Detect if a response is too generic after tool execution"""
        if not has_tool_results:
            return False
        
        poor_phrases = [
            "task completed",
            "done",
            "finished", 
            "completed successfully",
            "all set",
            "ready",
            "task complete",
            "operation completed"
        ]
        
        response_lower = response_content.lower().strip()
        
        # Check if response is very short and contains generic phrases
        if len(response_lower) < 100:  # Short response
            for phrase in poor_phrases:
                if phrase in response_lower:
                    # If the response is mostly just the generic phrase, it's poor
                    if len(response_lower) < len(phrase) + 30:
                        return True
        
        return False
    
    def _improve_tool_response(self, response_content: str, tool_results: list) -> str:
        """Improve a generic response by incorporating actual tool results"""
        if not tool_results:
            return response_content
        
        # Get the most recent tool result
        last_result = tool_results[-1] if tool_results else ""
        
        # First check if the response is already clean
        if self._is_response_readable(response_content):
            return response_content
        
        # If not, create a generic response
        # This is safer than trying to parse complex objects
        return self._clean_response_for_speech(last_result)
    
    async def process_user_input(self, user_text: str, mcp_client, tools):
        """Process user input and generate response"""
        # Store the original user request for context
        original_user_text = user_text
        
        # Add user message to history
        self.state.add_user_message(user_text)
        
        try:
            # Determine if we should pass tools parameter based on the provider
            use_tools = self._should_use_tools_parameter(self.state.chat_provider)
            
            if use_tools:
                # OpenAI-compatible provider - use tools parameter
                completion = self.state.chat_provider.complete(
                    messages=self.state.conversation_history,
                    tools=tools,
                    tool_choice="auto"
                )
            else:
                # Local model - no tools parameter (will use text parsing)
                completion = self.state.chat_provider.complete(
                    messages=self.state.conversation_history
                )

            choice  = completion.choices[0]
            message = choice.message
            
            assistant_response = ""
            last_tool_name = None  # Track the last tool used
            
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                # Handle tool calls
                self.state.add_assistant_message(message.content, message.tool_calls)
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    # Track tool name for generic responses
                    last_tool_name = tool_call.function.name
                    
                    # Check if we should interrupt
                    if self.state.interrupt_flag.is_set() or self.state.get_mode() == AssistantMode.STUCK_CHECK:
                        logger.info("Processing interrupted")
                        self.state.set_mode(AssistantMode.LISTENING)
                        return None
                    
                    try:
                        tool_result = await call_tool_with_timeout(
                            mcp_client, tool_call, self.config.tool_timeout
                        )
                        
                        # Extract clean text from result
                        clean_result = self._extract_text_from_any_format(tool_result)
                        
                        # Log for debugging
                        logger.debug(f"Tool {tool_call.function.name} raw result type: {type(tool_result)}")
                        logger.debug(f"Tool {tool_call.function.name} clean result: {clean_result[:100]}...")
                        
                        # Add tool result to history with clean text
                        self.state.add_tool_message(tool_call.id, clean_result)
                        
                    except Exception as tool_error:
                        # Tool failed - trigger backup immediately for better reliability
                        logger.warning(f"Tool {tool_call.function.name} failed: {tool_error}")
                        
                        # Remove the assistant message with tool_calls that we just added
                        if (self.state.conversation_history and 
                            self.state.conversation_history[-1].get("role") == "assistant" and
                            "tool_calls" in self.state.conversation_history[-1]):
                            self.state.conversation_history.pop()
                        
                        # Use the backup provider directly
                        logger.info("Tool failed - using backup provider for this request")
                        
                        try:
                            # Force backup to handle the original request
                            from .model_providers.openai_chat import OpenAIChatProvider
                            backup_provider = OpenAIChatProvider(
                                api_key=self.config.openai_api_key,
                                model=self.config.frontier_chat_model
                            )
                            
                            # Clean the conversation history for OpenAI
                            clean_history = self._clean_conversation_history()
                            
                            fallback_completion = backup_provider.complete(
                                messages=clean_history,
                                tools=tools,
                                tool_choice="auto"
                            )
                            
                            fallback_choice = fallback_completion.choices[0]
                            fallback_message = fallback_choice.message
                            
                            # Handle OpenAI's response
                            if fallback_choice.finish_reason == "tool_calls" and fallback_message.tool_calls:
                                self.state.conversation_history.append({
                                    "role": "assistant",
                                    "content": fallback_message.content,
                                    "tool_calls": [
                                        {
                                            'id': tc.id,
                                            'type': tc.type,
                                            'function': {
                                                'name': tc.function.name,
                                                'arguments': tc.function.arguments
                                            }
                                        } for tc in fallback_message.tool_calls
                                    ]
                                })
                                
                                # Execute OpenAI's tool calls
                                for backup_tool_call in fallback_message.tool_calls:
                                    try:
                                        backup_result = await call_tool_with_timeout(
                                            mcp_client, backup_tool_call, self.config.tool_timeout
                                        )
                                        
                                        clean_backup_result = self._extract_text_from_any_format(backup_result)
                                        
                                        self.state.conversation_history.append({
                                            "role": "tool",
                                            "tool_call_id": backup_tool_call.id,
                                            "content": clean_backup_result
                                        })
                                    except Exception as backup_tool_error:
                                        logger.error(f"Backup tool also failed: {backup_tool_error}")
                                        return "I'm having trouble with that request. Please try being more specific."
                                
                                # Get OpenAI's follow-up
                                final_completion = backup_provider.complete(
                                    messages=self.state.conversation_history,
                                    tools=tools,
                                    tool_choice="auto"
                                )
                                
                                final_response = final_completion.choices[0].message.content or "Task completed."
                                final_response = self._extract_text_from_any_format(final_response)
                                self.state.conversation_history.append({
                                    "role": "assistant",
                                    "content": final_response
                                })
                                
                                logger.info("Turn answered by: backup (tool failure fallback)")
                                return final_response
                            else:
                                # OpenAI gave direct response
                                response = fallback_message.content or "I can help you with that."
                                response = self._extract_text_from_any_format(response)
                                self.state.conversation_history.append({
                                    "role": "assistant",
                                    "content": response
                                })
                                
                                logger.info("Turn answered by: backup (tool failure fallback)")
                                return response
                                
                        except Exception as fallback_error:
                            logger.error(f"Backup provider also failed: {fallback_error}")
                            return "I'm having trouble processing your request. Please try again."
                
                # Get the model's follow-up response if not interrupted and all tools succeeded
                if not self.state.interrupt_flag.is_set() and self.state.get_mode() != AssistantMode.STUCK_CHECK:
                    # Track tool results for potential improvement
                    tool_results = []
                    for msg in self.state.conversation_history:
                        if msg.get("role") == "tool":
                            content = msg.get("content", "")
                            clean_content = self._extract_text_from_any_format(content)
                            tool_results.append(clean_content)
                    
                    # For follow-up, use the same tools logic
                    if use_tools:
                        follow_up = self.state.chat_provider.complete(
                            messages=self.state.conversation_history,
                            tools=tools,
                            tool_choice="auto"
                        )
                    else:
                        follow_up = self.state.chat_provider.complete(
                            messages=self.state.conversation_history
                        )
                    
                    follow_up_choice = follow_up.choices[0]
                    follow_up_message = follow_up_choice.message
                    
                    # CRITICAL FIX: Check if the local model returned ANOTHER tool call instead of text
                    if follow_up_choice.finish_reason == "tool_calls" and follow_up_message.tool_calls:
                        logger.warning("Local model returned tool calls in follow-up instead of text response!")
                        
                        # Use clean generic response
                        assistant_response = self._clean_response_for_speech(
                            tool_results[-1] if tool_results else "Task completed",
                            tool_name=last_tool_name,
                            user_request=original_user_text
                        )
                        
                        self.state.conversation_history.append({
                            "role": "assistant", 
                            "content": assistant_response
                        })
                    else:
                        # Normal text response from the model
                        assistant_response = follow_up_message.content or "Task completed."
                        assistant_response = self._extract_text_from_any_format(assistant_response)
                        
                        # Check if the response is too generic and improve it
                        if self._detect_poor_tool_response(assistant_response, bool(tool_results)):
                            logger.info("Detected poor tool response, improving...")
                            improved_response = self._improve_tool_response(assistant_response, tool_results)
                            if improved_response != assistant_response:
                                assistant_response = improved_response
                                logger.info("Improved generic response with actual tool results")
                        
                        self.state.conversation_history.append({
                            "role": "assistant", 
                            "content": assistant_response
                        })
            else:
                # Direct response without tool calls
                assistant_response = message.content or "I'm not sure how to respond to that."
                assistant_response = self._extract_text_from_any_format(assistant_response)
                self.state.conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
            
            return assistant_response
            
        except Exception as e:
            # Handle all errors at this level
            logger.error(f"Error in chat completion: {e}")
            return "I encountered an error processing your request. Please try again."

    def _clean_conversation_history(self):
        """Clean conversation history for OpenAI compatibility"""
        clean_history = []
        for msg in self.state.conversation_history:
            if isinstance(msg, dict):
                clean_msg = dict(msg)
                if 'tool_calls' in clean_msg and clean_msg['tool_calls']:
                    clean_tool_calls = []
                    for tc in clean_msg['tool_calls']:
                        if isinstance(tc, dict):
                            clean_tool_calls.append(tc)
                        else:
                            clean_tc = {
                                'id': getattr(tc, 'id', ''),
                                'type': getattr(tc, 'type', 'function'),
                                'function': {
                                    'name': getattr(tc.function, 'name', '') if hasattr(tc, 'function') else '',
                                    'arguments': getattr(tc.function, 'arguments', '{}') if hasattr(tc, 'function') else '{}'
                                }
                            }
                            clean_tool_calls.append(clean_tc)
                    clean_msg['tool_calls'] = clean_tool_calls
                clean_history.append(clean_msg)
            else:
                clean_history.append(msg)
        
        return clean_history

    async def conversation_loop(self):
        """Main conversation loop with sleep/wake functionality"""
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
            tools = await get_tools_cached(mcp_client, self.state)
            
            # Initial greeting
            await speak_text("Hello! I'm listening. You can say 'go to sleep' to put me in sleep mode.", self.state, self.config)
            
            while self.state.running:
                try:
                    current_mode = self.state.get_mode()
                    
                    # Handle sleep mode - keep listening but only process wake commands
                    if current_mode == AssistantMode.SLEEPING:
                        logger.debug("In sleep mode, listening for wake commands only")
                        
                        # IMPORTANT: Stay in LISTENING mode for audio system, but track sleep state separately
                        # The audio system needs to think we're listening to keep capturing audio
                        if self.state.get_mode() == AssistantMode.SLEEPING:
                            # Temporarily set to LISTENING for audio capture
                            self.state.set_mode(AssistantMode.LISTENING)
                        
                        # Record audio (this will work now that we're in LISTENING mode)
                        audio_buffer = await self.audio_recorder.record_until_silence(
                            self.state, self.config
                        )
                        
                        # Immediately set back to SLEEPING
                        self.state.set_mode(AssistantMode.SLEEPING)
                        
                        if not audio_buffer:
                            await asyncio.sleep(0.5)
                            continue
                        
                        # Process for wake commands only
                        try:
                            # Use transcribe_audio with bypass validation
                            user_text = await transcribe_audio(audio_buffer, self.state, self.config, check_stuck_phrase=True)
                            
                            if user_text:
                                logger.info(f"Sleep mode heard: '{user_text}'")
                                
                                # Check for wake commands
                                lower_text = user_text.lower().strip()
                                wake_words = ["wake", "hello", "hey", "up", "wake up"]
                                
                                if any(word in lower_text for word in wake_words):
                                    logger.info("Wake command detected! Waking up...")
                                    self.state.set_mode(AssistantMode.LISTENING)
                                    await speak_text("I'm awake! How can I help you?", self.state, self.config)
                                    continue
                                else:
                                    logger.debug(f"Not a wake command, staying asleep: '{user_text}'")
                            else:
                                logger.debug("No valid transcription in sleep mode")
                                
                        except Exception as transcribe_error:
                            logger.debug(f"Transcription error in sleep mode: {transcribe_error}")
                        
                        # Stay in sleep mode
                        continue
                    
                    # Normal operation - only process when in listening mode
                    elif current_mode != AssistantMode.LISTENING:
                        logger.debug(f"Not listening, current mode: {current_mode}")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Record audio until silence (normal mode)
                    audio_buffer = await self.audio_recorder.record_until_silence(
                        self.state, self.config
                    )
                    
                    if not audio_buffer:
                        continue
                    
                    # Switch to recording mode
                    self.state.set_mode(AssistantMode.RECORDING)
                    
                    # Transcribe with normal validation
                    user_text = await transcribe_audio(audio_buffer, self.state, self.config)
                    if not user_text:
                        self.state.set_mode(AssistantMode.LISTENING)
                        continue
                    
                    logger.info(f"User said: {user_text}")
                    
                    # Handle special commands first (including sleep)
                    if await self.handle_special_commands(user_text):
                        if not self.state.running:  # Exit was called
                            break
                        # If we just went to sleep, continue to sleep mode handling
                        continue
                    
                    # Switch to processing mode for normal commands
                    self.state.set_mode(AssistantMode.PROCESSING)
                    
                    # Process user input
                    assistant_response = await self.process_user_input(user_text, mcp_client, tools)

                    provider_used = getattr(self.state.chat_provider, "last_provider", "unknown")
                    logger.info(f"Turn answered by: {provider_used}")
                    
                    # CRITICAL FIX: Ensure response is clean before speaking
                    if assistant_response:
                        # Use the cleaner helper to ensure readable response
                        clean_response = self._clean_response_for_speech(assistant_response)
                        
                        logger.debug(f"Original response: {assistant_response[:100] if isinstance(assistant_response, str) else str(assistant_response)[:100]}...")
                        logger.debug(f"Clean response: {clean_response}")
                        
                        # Speak the response if not interrupted
                        if not self.state.interrupt_flag.is_set() and \
                           self.state.get_mode() != AssistantMode.STUCK_CHECK:
                            await speak_text(clean_response, self.state, self.config)
                        else:
                            logger.info("Skipping speech due to interruption")
                            self.state.set_mode(AssistantMode.LISTENING)
                    else:
                        logger.info("No response to speak")
                        self.state.set_mode(AssistantMode.LISTENING)
                        
                except Exception as e:
                    # Handle conversation-level errors
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