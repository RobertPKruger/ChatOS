# voice_assistant/conversation.py - CORRECTED VERSION
"""
Main conversation loop and management with proper tool handling for local vs backup models
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
        
        # Extract text content from various formats
        def extract_text(result):
            """Extract plain text from various result formats"""
            if result is None:
                return ""
            
            # If it's already a string, return it
            if isinstance(result, str):
                return result
            
            # If it's a TextContent object with .text attribute
            if hasattr(result, 'text'):
                return result.text
            
            # If it's a list, try to extract from first element
            if isinstance(result, list) and len(result) > 0:
                first_item = result[0]
                if hasattr(first_item, 'text'):
                    return first_item.text
                elif isinstance(first_item, str):
                    return first_item
                else:
                    return str(first_item)
            
            # If it's a dict, look for common text fields
            if isinstance(result, dict):
                for key in ['text', 'content', 'message', 'result']:
                    if key in result:
                        return str(result[key])
            
            # Fallback: convert to string
            return str(result)
        
        # Extract the actual text content
        result_text = extract_text(last_result)
        
        # Clean up any remaining formatting artifacts
        result_text = result_text.strip()
        
        # Try to extract meaningful content from tool result
        if result_text:
            # If it's a steam games list
            if "Steam Games" in result_text or "App ID:" in result_text:
                return f"Here are your Steam games:\n{result_text}"
            
            # If it's about launching something
            elif "Launched" in result_text or "opened" in result_text.lower():
                return result_text
            
            # If it's a list of files or directories
            elif "Directory" in result_text or "Files:" in result_text:
                return f"Here's what I found:\n{result_text}"
            
            # For any other tool result, present it clearly
            elif len(result_text) > 10:  # Non-trivial result
                return result_text  # Just return the clean text directly
        
        # If we can't improve it, return the original
        return response_content
    
    async def process_user_input(self, user_text: str, mcp_client, tools):
        """Process user input and generate response"""
        # Add user message to history
        self.state.conversation_history.append({"role": "user", "content": user_text})
        
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
                    
                    try:
                        tool_result = await call_tool_with_timeout(
                            mcp_client, tool_call, self.config.tool_timeout
                        )
                        
                        # Extract clean text from tool result before storing
                        if hasattr(tool_result, 'text'):
                            clean_result = tool_result.text
                        elif isinstance(tool_result, list) and len(tool_result) > 0 and hasattr(tool_result[0], 'text'):
                            clean_result = tool_result[0].text
                        elif isinstance(tool_result, str):
                            clean_result = tool_result
                        else:
                            clean_result = str(tool_result)
                        
                        # Add tool result to history with clean text
                        self.state.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": clean_result  # Store clean text instead of raw object
                        })
                        
                    except Exception as tool_error:
                        # Tool failed - clean up and trigger manual fallback
                        logger.warning(f"Tool {tool_call.function.name} failed: {tool_error}")
                        
                        # Remove the assistant message with tool_calls that we just added
                        if (self.state.conversation_history and 
                            self.state.conversation_history[-1].get("role") == "assistant" and
                            "tool_calls" in self.state.conversation_history[-1]):
                            self.state.conversation_history.pop()
                        
                        # Manually trigger OpenAI fallback for this request
                        logger.info("Tool failed - manually triggering OpenAI fallback")
                        
                        try:
                            # Force OpenAI to handle the original request
                            from .model_providers.openai_chat import OpenAIChatProvider
                            backup_provider = OpenAIChatProvider(
                                api_key=self.config.openai_api_key,
                                model=self.config.frontier_chat_model
                            )
                            
                            # Clean the conversation history for OpenAI (serialize tool calls)
                            clean_history = []
                            for msg in self.state.conversation_history:
                                if isinstance(msg, dict):
                                    clean_msg = dict(msg)
                                    if 'tool_calls' in clean_msg and clean_msg['tool_calls']:
                                        # Serialize tool calls properly
                                        clean_tool_calls = []
                                        for tc in clean_msg['tool_calls']:
                                            if isinstance(tc, dict):
                                                clean_tool_calls.append(tc)
                                            else:
                                                # Convert object to dict
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
                            
                            # Validate conversation history - remove orphaned tool_calls
                            validated_history = []
                            i = 0
                            while i < len(clean_history):
                                msg = clean_history[i]
                                
                                # If this is an assistant message with tool_calls
                                if (msg.get('role') == 'assistant' and 
                                    msg.get('tool_calls')):
                                    
                                    # Check if all tool_calls have corresponding tool responses
                                    tool_call_ids = [tc['id'] for tc in msg['tool_calls']]
                                    validated_msg = dict(msg)
                                    validated_history.append(validated_msg)
                                    
                                    # Look for corresponding tool responses
                                    j = i + 1
                                    found_responses = set()
                                    
                                    while j < len(clean_history) and clean_history[j].get('role') == 'tool':
                                        tool_response = clean_history[j]
                                        tool_call_id = tool_response.get('tool_call_id')
                                        if tool_call_id in tool_call_ids:
                                            validated_history.append(tool_response)
                                            found_responses.add(tool_call_id)
                                        j += 1
                                    
                                    # If some tool_calls don't have responses, remove them
                                    if len(found_responses) < len(tool_call_ids):
                                        logger.warning(f"Removing orphaned tool_calls: {set(tool_call_ids) - found_responses}")
                                        # Filter out tool_calls that don't have responses
                                        validated_msg['tool_calls'] = [
                                            tc for tc in validated_msg['tool_calls'] 
                                            if tc['id'] in found_responses
                                        ]
                                        
                                        # If no tool_calls remain, convert to regular response
                                        if not validated_msg['tool_calls']:
                                            validated_msg.pop('tool_calls', None)
                                            if not validated_msg.get('content'):
                                                validated_msg['content'] = "I'll help you with that."
                                    
                                    i = j  # Skip the tool responses we already processed
                                else:
                                    # Regular message
                                    validated_history.append(msg)
                                    i += 1
                            
                            fallback_completion = backup_provider.complete(
                                messages=validated_history,
                                tools=tools,
                                tool_choice="auto"
                            )
                            
                            fallback_choice = fallback_completion.choices[0]
                            fallback_message = fallback_choice.message
                            
                            # Handle OpenAI's response (might include tool calls)
                            if fallback_choice.finish_reason == "tool_calls" and fallback_message.tool_calls:
                                # Add OpenAI's assistant message
                                self.state.conversation_history.append({
                                    "role": "assistant",
                                    "content": fallback_message.content,
                                    "tool_calls": fallback_message.tool_calls
                                })
                                
                                # Execute OpenAI's tool calls
                                for backup_tool_call in fallback_message.tool_calls:
                                    try:
                                        backup_result = await call_tool_with_timeout(
                                            mcp_client, backup_tool_call, self.config.tool_timeout
                                        )
                                        
                                        self.state.conversation_history.append({
                                            "role": "tool",
                                            "tool_call_id": backup_tool_call.id,
                                            "content": backup_result
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
                                self.state.conversation_history.append({
                                    "role": "assistant",
                                    "content": final_response
                                })
                                
                                logger.info("Turn answered by: backup (manual fallback)")
                                return final_response
                            else:
                                # OpenAI gave direct response
                                response = fallback_message.content or "I can help you with that."
                                self.state.conversation_history.append({
                                    "role": "assistant",
                                    "content": response
                                })
                                
                                logger.info("Turn answered by: backup (manual fallback)")
                                return response
                                
                        except Exception as fallback_error:
                            logger.error(f"Manual fallback also failed: {fallback_error}")
                            return "I'm having trouble processing your request. Please try again."
                
                # Get the model's follow-up response if not interrupted and all tools succeeded
                if not self.state.interrupt_flag.is_set() and self.state.get_mode() != AssistantMode.STUCK_CHECK:
                    # Track tool results for potential improvement
                    tool_results = []
                    for msg in self.state.conversation_history:
                        if msg.get("role") == "tool":
                            tool_results.append(msg.get("content", ""))
                    
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
                    
                    follow_up_message = follow_up.choices[0].message
                    assistant_response = follow_up_message.content or "Task completed."
                    
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
                # Regular text response
                assistant_response = message.content or "I'm not sure how to respond to that."
                self.state.conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
            
            return assistant_response
            
        except Exception as e:
            # Handle all errors at this level - don't re-raise tool errors
            logger.error(f"Error in chat completion: {e}")
            return "I encountered an error processing your request. Please try again."

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
            tools = await get_tools(mcp_client, self.state)
            
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
                    
                    # Speak the response if not interrupted
                    if assistant_response and not self.state.interrupt_flag.is_set() and \
                       self.state.get_mode() != AssistantMode.STUCK_CHECK:
                        await speak_text(assistant_response, self.state, self.config)
                    else:
                        logger.info("Skipping speech due to interruption or no response")
                        self.state.set_mode(AssistantMode.LISTENING)
                        
                except Exception as e:
                    # Don't catch tool execution errors - let them bubble up to failover
                    if "Tool execution failed:" in str(e):
                        logger.info(f"Tool error bubbling up - this should not happen, failover should handle it")
                        self.state.set_mode(AssistantMode.ERROR)
                        await asyncio.sleep(self.config.reconnect_delay)
                        self.state.set_mode(AssistantMode.LISTENING)
                        continue
                    
                    # Handle other conversation-level errors
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