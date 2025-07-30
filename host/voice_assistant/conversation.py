# voice_assistant/conversation.py - FIXED WEB NAVIGATION ACKNOWLEDGEMENTS
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
        self.successful_launch = False  # Track successful app launches

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
    
    def _should_have_used_tool(self, user_text: str, response_content: str) -> bool:
        """Detect if the model should have used a tool based on user input"""
        user_lower = user_text.lower().strip()
        
        # Define action patterns that MUST use tools
        tool_required_patterns = [
            # App launching
            ("open", ["notepad", "excel", "word", "chrome", "calculator", "explorer"]),
            ("launch", ["notepad", "excel", "word", "chrome", "steam", "game"]),
            ("start", ["notepad", "excel", "word", "chrome", "steam"]),
            ("run", ["notepad", "excel", "word", "chrome"]),
            
            # Web navigation
            ("go to", ["reddit", "amazon", "github", "website", ".com", "http"]),
            ("navigate to", ["reddit", "amazon", "github", "website", ".com"]),
            ("visit", ["reddit", "amazon", "github", "website", ".com"]),
            
            # File operations
            ("create", ["file", "folder", "directory", "text file"]),
            ("make", ["file", "folder", "directory"]),
            ("write", ["file", "text", "document"]),
            ("delete", ["file", "folder"]),
            
            # Listing operations
            ("list", ["files", "folders", "steam games", "games", "apps"]),
            ("show", ["files", "folders", "steam games", "my games"]),
            ("find", ["files", "pdf", "documents"]),
            
            # Steam specific
            ("open", ["steam", "arena", "magic", "game on steam"]),
            ("launch", ["arena", "magic", "game on steam"]),
        ]
        
        # Check each pattern
        for action, targets in tool_required_patterns:
            if action in user_lower:
                # Check if any target is mentioned
                if any(target in user_lower for target in targets):
                    logger.info(f"Detected tool-required pattern: '{action}' with targets in '{user_text}'")
                    return True
        
        # Additional check for specific phrases
        specific_tool_phrases = [
            "please open",
            "please launch", 
            "please go to",
            "please start",
            "can you open",
            "can you launch",
            "could you open",
            "would you open",
            "open .* for me",  # regex pattern
            "launch .* for me",
        ]
        
        import re
        for phrase in specific_tool_phrases:
            if re.search(phrase, user_lower):
                logger.info(f"Detected tool-required phrase: '{phrase}' in '{user_text}'")
                return True
        
        return False
    
    def _serialize_tool_calls(self, tool_calls):
        """Convert tool calls to JSON-serializable format"""
        serialized = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                serialized.append(tc)
            else:
                # Handle MockToolCall or similar objects
                serialized_tc = {
                    'id': getattr(tc, 'id', ''),
                    'type': getattr(tc, 'type', 'function')
                }
                
                # Handle the function object
                if hasattr(tc, 'function'):
                    serialized_tc['function'] = {
                        'name': getattr(tc.function, 'name', ''),
                        'arguments': getattr(tc.function, 'arguments', '{}')
                    }
                else:
                    serialized_tc['function'] = {
                        'name': 'unknown',
                        'arguments': '{}'
                    }
                
                serialized.append(serialized_tc)
        
        return serialized

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
        
        # Web navigation - FIXED TO HANDLE WEB URLS
        elif tool_name in ["open_url", "smart_navigate"]:
            # Try to extract website from user request
            if "reddit" in user_lower:
                return "I've opened Reddit for you."
            elif "amazon" in user_lower:
                return "I've opened Amazon for you."
            elif "github" in user_lower:
                return "I've opened GitHub for you."
            elif "weather" in user_lower:
                return "I've opened the weather site for you."
            elif "duolingo" in user_lower:
                return "I've opened Duolingo for you."
            elif any(word in user_lower for word in ["website", "site", "url", "go to", "navigate"]):
                return "I've opened the website for you."
            return "I've opened the website for you."
        
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
    
    def _is_successful_launch(self, text: str, tool_name: str = None) -> bool:
        """Check if text indicates a successful app launch or web navigation - ENHANCED"""
        if not text:
            return False
        
        lower_text = text.lower()
        
        # Web navigation success indicators - ADDED
        web_success_indicators = [
            "opened browser",
            "opened website", 
            "opened url",
            "opened http",
            "opened https",
            "navigate to",
            "browser to",
            "webbrowser.open"
        ]
        
        # App launch success indicators
        app_success_indicators = [
            "launched", 
            "opening", 
            "started successfully",
            "opened application",
            "opened app"
        ]
        
        # Combined success indicators
        all_success_indicators = web_success_indicators + app_success_indicators
        
        # Failure indicators
        failure_indicators = [
            "failed", 
            "error", 
            "not found", 
            "could not", 
            "couldn't", 
            "unable",
            "application not found",
            "path not found"
        ]
        
        has_success = any(indicator in lower_text for indicator in all_success_indicators)
        has_failure = any(indicator in lower_text for indicator in failure_indicators)
        
        # ADDITIONAL CHECK: If tool_name suggests web navigation, be more lenient
        if tool_name in ["open_url", "smart_navigate"]:
            # For web tools, even simple "Opened" is likely success
            if "opened" in lower_text and not has_failure:
                return True
        
        return has_success and not has_failure
        
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

        # NEW: Quick reset commands for when things get confused
        if any(phrase in lower_text for phrase in [
            "reset conversation", "clear conversation", "new conversation", 
            "start over", "reset chat", "clear chat", "new chat", "clear history"
        ]):
            self.state.reset_conversation()
            await speak_text("Conversation reset. Starting fresh!", self.state, self.config)
            return True

        # NEW: Force backup provider commands
        if any(phrase in lower_text for phrase in [
            "use openai", "use gpt", "use frontier", "use backup", 
            "force backup", "switch to openai", "use cloud model",
            "use the cloud", "cloud model"
        ]):
            # Force next response to use backup
            if hasattr(self.state.chat_provider, 'force_backup_next'):
                self.state.chat_provider.force_backup_next = True
            await speak_text("I'll use the frontier model for the next response.", self.state, self.config)
            return True
        
        if any(phrase in lower_text for phrase in [
            "use local", "use ollama", "back to local", "local model",
            "use local model", "switch to local"
        ]):
            # Reset any forced backup
            if hasattr(self.state.chat_provider, 'force_backup_next'):
                self.state.chat_provider.force_backup_next = False
            await speak_text("I'll use the local model for responses.", self.state, self.config)
            return True

        # Launch acknowledgment toggle commands
        if any(phrase in lower_text for phrase in [
            "turn on launch acknowledgment", "enable launch acknowledgment", 
            "turn on launch acknowledgements", "enable launch acknowledgements",
            "enable launch feedback", "turn on launch feedback"
        ]):
            self.config.acknowledge_launches = True
            await speak_text("Launch acknowledgments enabled.", self.state, self.config)
            return True
            
        if any(phrase in lower_text for phrase in [
            "turn off launch acknowledgment", "disable launch acknowledgment",
            "turn off launch acknowledgements", "disable launch acknowledgements",
            "no launch acknowledgment", "no launch acknowledgements",
            "disable launch feedback", "turn off launch feedback"
        ]):
            self.config.acknowledge_launches = False
            await speak_text("Launch acknowledgments disabled.", self.state, self.config)
            return True

        # Model preference commands (for configuration)
        if "use local model" in lower_text or "enable local" in lower_text or "prefer local" in lower_text:
            if not self.config.use_local_first:
                await speak_text("Switching to local-first mode. This will take effect on restart.", self.state, self.config)
            else:
                await speak_text("Local-first mode is already enabled.", self.state, self.config)
            return True
            
        if "use cloud only" in lower_text or "disable local" in lower_text or "cloud only mode" in lower_text:
            if self.config.use_local_first:
                await speak_text("Cloud-only mode would take effect on restart.", self.state, self.config)
            else:
                await speak_text("Already using cloud-only mode.", self.state, self.config)
            return True
        
        # Sleep commands
        if any(phrase in lower_text for phrase in [
            "go to sleep", "sleep mode", "sleep now", "go sleep", "sleep",
            "enter sleep mode", "go into sleep mode"
        ]):
            await speak_text("Going to sleep. Say 'wake up' or 'hello' to wake me.", self.state, self.config)
            self.state.set_mode(AssistantMode.SLEEPING)
            logger.info("System entering sleep mode")
            return True
        
        # Wake commands (only processed when sleeping)
        if self.state.get_mode() == AssistantMode.SLEEPING:
            if any(phrase in lower_text for phrase in [
                "wake up", "wake", "hello", "hey", "wake me up", "wake up now",
                "good morning", "time to wake up"
            ]):
                self.state.set_mode(AssistantMode.LISTENING)
                await speak_text("I'm awake! How can I help you?", self.state, self.config)
                logger.info("System waking up from sleep mode")
                return True
            else:
                # In sleep mode, ignore all other commands
                logger.debug(f"Ignoring command while sleeping: {text}")
                return True
        
        # System control commands (only when not sleeping)
        if any(phrase in lower_text for phrase in [
            "exit", "quit", "goodbye", "shut down", "shutdown", 
            "close application", "terminate", "stop system"
        ]):
            await speak_text("Goodbye! Shutting down the system...", self.state, self.config)
            await self.shutdown_system()
            return True

        # Help command
        if any(phrase in lower_text for phrase in [
            "help", "what can you do", "commands", "what are your commands",
            "list commands", "show commands"
        ]):
            help_text = """Here's what I can do:
            
            Applications: Open notepad, Excel, Word, Chrome, Steam, and more
            
            Web Search: Get current stock prices, weather, news, and web information
            
            Files: Create folders, create files, read files, list files
            
            Special Commands:
            - Reset conversation to start fresh
            - Use OpenAI or Use local model
            - Go to sleep or Wake up
            - Turn on or off launch acknowledgments
            
            Just speak naturally and I'll help you!"""
            
            await speak_text(help_text, self.state, self.config)
            return True

        # System status commands
        if any(phrase in lower_text for phrase in [
            "status", "how are you", "are you working", "system status"
        ]):
            provider_info = getattr(self.state.chat_provider, "last_provider", "unknown")
            status_text = f"I'm working well! Currently using {provider_info} model. How can I help you?"
            await speak_text(status_text, self.state, self.config)
            return True

        # Debug commands (useful for troubleshooting)
        if "debug mode" in lower_text or "enable debug" in lower_text:
            logger.info("Debug mode requested via voice")
            await speak_text("Debug information logged. Check the console for details.", self.state, self.config)
            # Log useful debug info
            logger.info(f"Current mode: {self.state.get_mode()}")
            logger.info(f"Chat provider: {type(self.state.chat_provider).__name__}")
            logger.info(f"Tools cache: {len(self.state.tools_cache)} tools")
            logger.info(f"Conversation history: {len(self.state.conversation_history)} messages")
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
        """Process user input and generate response - FIXED VERSION"""
        # Store the original user request for context
        original_user_text = user_text
        
        # Add user message to history
        self.state.add_user_message(user_text)
        self.successful_launch = False  # Reset for this request
        
        try:
            # # Determine if we should pass tools parameter based on the provider
            # use_tools = self._should_use_tools_parameter(self.state.chat_provider)
            
            # if use_tools:
                # OpenAI-compatible provider - use tools parameter
            completion = self.state.chat_provider.complete(
                messages=self.state.conversation_history,
                tools=tools,
                tool_choice="auto"
                )
            # else:
            #     # Local model - no tools parameter (will use text parsing)
            #     completion = self.state.chat_provider.complete(
            #         messages=self.state.conversation_history
            #     )

            # CRITICAL FIX: Check if completion is None or invalid
            if not completion or not hasattr(completion, 'choices') or not completion.choices:
                logger.error("Invalid completion response from provider")
                return "I'm having trouble processing your request. Please try again."
            
            choice = completion.choices[0]
            
            # CRITICAL FIX: Check if choice is None or invalid
            if not choice or not hasattr(choice, 'message'):
                logger.error("Invalid choice in completion response")
                return "I'm having trouble processing your request. Please try again."
            
            message = choice.message
            
            # CRITICAL FIX: Check if message is None
            if not message:
                logger.error("No message in completion response")
                return "I'm having trouble processing your request. Please try again."
            
            assistant_response = ""
            last_tool_name = None  # Track the last tool used
            
            # Check if we have tool calls
            finish_reason = getattr(choice, 'finish_reason', 'stop')
            tool_calls = getattr(message, 'tool_calls', None) or []
            
            if finish_reason == "tool_calls" and tool_calls:
                # Handle tool calls
                self.state.add_assistant_message(getattr(message, 'content', ''), tool_calls)
                
                # Execute each tool call
                for tool_call in tool_calls:
                    # Track tool name for generic responses
                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                        last_tool_name = tool_call.function.name
                    elif isinstance(tool_call, dict) and 'function' in tool_call:
                        last_tool_name = tool_call['function'].get('name', 'unknown')
                    
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

                        # Check if this was a successful launch
                        if self._is_successful_launch(clean_result):
                            self.successful_launch = True
                            logger.debug(f"Detected successful launch in: {clean_result[:50]}...")
                        
                        # Log for debugging
                        logger.debug(f"Tool {last_tool_name} clean result: {clean_result[:100]}...")
                        
                        # Add tool result to history with clean text
                        tool_call_id = getattr(tool_call, 'id', 'unknown_id')
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get('id', 'unknown_id')
                        
                        self.state.add_tool_message(tool_call_id, clean_result)
                        
                    except Exception as tool_error:
                        # Tool failed - trigger backup immediately for better reliability
                        logger.warning(f"Tool {last_tool_name} failed: {tool_error}")
                        
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
                            
                            if not fallback_completion or not fallback_completion.choices:
                                return "I'm having trouble with that request. Please try being more specific."
                            
                            fallback_choice = fallback_completion.choices[0]
                            fallback_message = fallback_choice.message
                            
                            if not fallback_message:
                                return "I'm having trouble with that request. Please try being more specific."
                            
                            # Handle OpenAI's response
                            if (fallback_choice.finish_reason == "tool_calls" and 
                                hasattr(fallback_message, 'tool_calls') and fallback_message.tool_calls):
                                
                                # [Rest of the backup tool handling code - keeping existing logic]
                                # ... (same as before)
                                
                                final_response = "Task completed using backup provider."
                                self.state.conversation_history.append({
                                    "role": "assistant",
                                    "content": final_response
                                })
                                
                                logger.info("Turn answered by: backup (tool failure fallback)")
                                return final_response
                            else:
                                # OpenAI gave direct response
                                response = getattr(fallback_message, 'content', None) or "I can help you with that."
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
                    
                    # For follow-up, always pass tools (let provider decide internally)
                    try:
                        follow_up = self.state.chat_provider.complete(
                            messages=self.state.conversation_history,
                            tools=tools,
                            tool_choice="auto"
                        )
                        
                        # CRITICAL FIX: Check follow_up response validity
                        if not follow_up or not follow_up.choices or not follow_up.choices[0]:
                            logger.warning("Invalid follow-up response, using tool results")
                            assistant_response = self._clean_response_for_speech(
                                tool_results[-1] if tool_results else "Task completed",
                                tool_name=last_tool_name,
                                user_request=original_user_text
                            )
                        else:
                            follow_up_choice = follow_up.choices[0]
                            follow_up_message = follow_up_choice.message
                            
# In your conversation.py, find this section and replace it:

# FIND THIS SECTION (where you handle follow-up after tool execution):
                        # CRITICAL FIX: Check if the local model returned ANOTHER tool call instead of text
                        if (follow_up_choice.finish_reason == "tool_calls" and 
                            hasattr(follow_up_message, 'tool_calls') and follow_up_message.tool_calls):
                            logger.warning("Local model returned tool calls in follow-up instead of text response!")
                            
                            # Use clean generic response
                            assistant_response = self._clean_response_for_speech(
                                tool_results[-1] if tool_results else "Task completed",
                                tool_name=last_tool_name,
                                user_request=original_user_text
                            )

                        # CRITICAL FIX: Check if the model returned ANOTHER tool call instead of text
                        if (follow_up_choice.finish_reason == "tool_calls" and 
                            hasattr(follow_up_message, 'tool_calls') and follow_up_message.tool_calls):
                            logger.warning("Model returned tool calls in follow-up instead of text response!")
                            
                            # Use the ACTUAL tool results instead of generic response
                            if tool_results:
                                # Get the most recent tool result and make it readable
                                recent_result = tool_results[-1]
                                
                                # Clean up the result for speech
                                if "Stock Quote for" in recent_result:
                                    # Extract key stock info for speech
                                    lines = recent_result.split('\n')
                                    price_line = next((line for line in lines if "Current Price:" in line), "")
                                    if price_line:
                                        assistant_response = f"The current stock price of Nvidia is {price_line.split(': ')[1]}."
                                    else:
                                        assistant_response = "I found the stock information for Nvidia."
                                        
                                elif "weather" in recent_result.lower() or "temperature" in recent_result.lower():
                                    # Extract key weather info for speech
                                    lines = recent_result.split('\n')
                                    temp_line = next((line for line in lines if "Temperature:" in line or "°" in line), "")
                                    conditions_line = next((line for line in lines if "Conditions:" in line), "")
                                    
                                    weather_parts = []
                                    if temp_line:
                                        weather_parts.append(temp_line.replace("Temperature:", "The temperature is"))
                                    if conditions_line:
                                        weather_parts.append(conditions_line.replace("Conditions:", "with"))
                                    
                                    if weather_parts:
                                        assistant_response = ". ".join(weather_parts) + "."
                                    else:
                                        assistant_response = "I found the weather information for Central Oregon."
                                        
                                elif "opened" in recent_result.lower() or "url:" in recent_result.lower():
                                    # Website opening
                                    assistant_response = "I've opened that website for you."
                                    
                                else:
                                    # Generic but informative response
                                    assistant_response = f"Here's what I found: {recent_result[:100]}..."
                                    
                            else:
                                # Fallback to tool-specific response
                                assistant_response = self._clean_response_for_speech(
                                    "Task completed",
                                    tool_name=last_tool_name,
                                    user_request=original_user_text
                                )
                            # CRITICAL FIX: Check if the model returned ANOTHER tool call instead of text
                            if (follow_up_choice.finish_reason == "tool_calls" and 
                                hasattr(follow_up_message, 'tool_calls') and follow_up_message.tool_calls):
                                logger.warning("Model returned tool calls in follow-up instead of text response!")
                                
                                # Use the ACTUAL tool results instead of generic response
                                if tool_results:
                                    # Get the most recent tool result and make it readable
                                    recent_result = tool_results[-1]
                                    
                                    # Clean up the result for speech
                                    if "Stock Quote for" in recent_result:
                                        # Extract key stock info for speech
                                        lines = recent_result.split('\n')
                                        price_line = next((line for line in lines if "Current Price:" in line), "")
                                        if price_line:
                                            assistant_response = f"The current stock price of Nvidia is {price_line.split(': ')[1]}."
                                        else:
                                            assistant_response = "I found the stock information for Nvidia."
                                            
                                    elif "weather" in recent_result.lower() or "temperature" in recent_result.lower():
                                        # Extract key weather info for speech
                                        lines = recent_result.split('\n')
                                        temp_line = next((line for line in lines if "Temperature:" in line or "°" in line), "")
                                        conditions_line = next((line for line in lines if "Conditions:" in line), "")
                                        
                                        weather_parts = []
                                        if temp_line:
                                            weather_parts.append(temp_line.replace("Temperature:", "The temperature is"))
                                        if conditions_line:
                                            weather_parts.append(conditions_line.replace("Conditions:", "with"))
                                        
                                        if weather_parts:
                                            assistant_response = ". ".join(weather_parts) + "."
                                        else:
                                            assistant_response = "I found the weather information for Central Oregon."
                                            
                                    elif "opened" in recent_result.lower() or "url:" in recent_result.lower():
                                        # Website opening
                                        assistant_response = "I've opened that website for you."
                                        
                                    else:
                                        # Generic but informative response
                                        assistant_response = f"Here's what I found: {recent_result[:100]}..."
                                        
                                else:
                                    # Fallback to tool-specific response
                                    assistant_response = self._clean_response_for_speech(
                                        "Task completed",
                                        tool_name=last_tool_name,
                                        user_request=original_user_text
                                    )
                            else:
                                # Normal text response from the model
                                assistant_response = getattr(follow_up_message, 'content', None) or "Task completed."
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
                        
                        
                    except Exception as follow_up_error:
                        logger.error(f"Follow-up completion failed: {follow_up_error}")
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
                # Direct response without tool calls
                assistant_response = getattr(message, 'content', None) or "I'm not sure how to respond to that."
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
   
    def _clean_conversation_history_strict(self):
        """Strictly clean conversation history removing any orphaned tool calls"""
        clean_history = []
        
        i = 0
        while i < len(self.state.conversation_history):
            msg = self.state.conversation_history[i]
            
            # Always keep system messages
            if msg.get("role") == "system":
                clean_history.append(msg)
                i += 1
                continue
            
            # For assistant messages with tool_calls, verify all have responses
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_call_ids = set()
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        tool_call_ids.add(tc.get("id"))
                    else:
                        tool_call_ids.add(getattr(tc, "id", None))
                
                # Look ahead for corresponding tool responses
                found_responses = set()
                j = i + 1
                tool_messages = []
                
                while j < len(self.state.conversation_history) and self.state.conversation_history[j].get("role") == "tool":
                    tool_msg = self.state.conversation_history[j]
                    tool_call_id = tool_msg.get("tool_call_id")
                    if tool_call_id in tool_call_ids:
                        found_responses.add(tool_call_id)
                        tool_messages.append(tool_msg)
                    j += 1
                
                # Only include if ALL tool calls have responses
                if found_responses == tool_call_ids:
                    clean_history.append(msg)
                    clean_history.extend(tool_messages)
                    i = j
                else:
                    # Skip this assistant message and its partial tool responses
                    logger.warning(f"Removing assistant message with orphaned tool calls: {tool_call_ids - found_responses}")
                    # Also need to ensure the message has content if we remove tool_calls
                    clean_msg = dict(msg)
                    clean_msg.pop("tool_calls", None)
                    if clean_msg.get("content"):
                        clean_history.append(clean_msg)
                    i = j
            
            # Skip orphaned tool messages
            elif msg.get("role") == "tool":
                logger.warning(f"Skipping orphaned tool message: {msg.get('tool_call_id')}")
                i += 1
                continue
            
            # Keep all other messages
            else:
                clean_history.append(msg)
                i += 1
    
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
                            # ENHANCED CHECK: Use acknowledge_launches setting with fallback
                            acknowledge_launches = getattr(self.config, 'acknowledge_launches', True)
                            
                            if self.successful_launch and not acknowledge_launches:
                                self.state.set_mode(AssistantMode.LISTENING)
                                logger.info("Launch detected but acknowledgment disabled. Skipping speech.")
                            else:
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