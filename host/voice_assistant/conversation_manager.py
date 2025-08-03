"""
ConversationManager coordinates the recorder, the LLM provider
and the helper objects – but delegates heavy lifting to helpers.
FIXED: Proper tool response handling to prevent OpenAI API errors
"""

from __future__ import annotations

import asyncio
import logging
import contextlib
from typing import Optional, List, Dict

from openai import OpenAI
from voice_assistant.parallel_tools import execute_tools_parallel

from voice_assistant.state import AssistantState, AssistantMode
from voice_assistant.audio import ContinuousAudioRecorder
from voice_assistant.speech import transcribe_audio, speak_text
from voice_assistant.mcp_client import (
    get_mcp_client,
    get_tools_cached,
    call_tool_with_timeout,
    shutdown_mcp_server,
)
from voice_assistant.config import Config
from voice_assistant.result_extractor import extract_meaningful_content
from voice_assistant.command_handler import CommandHandler
from voice_assistant.stuck_detector import StuckDetector

log = logging.getLogger(__name__)


class ConversationManager:
    """High-level conversation FSM with fixed tool response handling."""

    def __init__(
        self,
        config: Config,
        state: AssistantState,
        audio_recorder: ContinuousAudioRecorder,
    ) -> None:
        self.config   = config
        self.state    = state
        self.audio    = audio_recorder

        # helpers
        self.commands = CommandHandler(config, state, self)
        self.stuck    = StuckDetector(config, state, audio_recorder)

        # runtime flags
        self.successful_launch = False
        self._stuck_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ #
    # --------------------------  core logic  -------------------------- #
    # ------------------------------------------------------------------ #
    async def process_user_input(
        self, user_text: str, mcp_client, tools
    ) -> str | None:
        """
        FIXED: Enhanced process_user_input with proper tool response handling
        """
        self.state.add_user_message(user_text)
        self.successful_launch = False

        try:
            # 1) Get a completion from whichever provider is active
            completion = self.state.chat_provider.complete(
                messages=self.state.conversation_history,
                tools=tools,
                tool_choice="auto",
            )
            if not completion or not completion.choices:
                log.error("No completion from provider.")
                return "I'm having trouble processing your request. Please try again."

            choice = completion.choices[0]
            message = choice.message
            f_reason = getattr(choice, "finish_reason", "stop")
            tool_calls = getattr(message, "tool_calls", None) or []

            assistant_response: str | None = None

            # 2) Execute any requested tools - FIXED TOOL RESPONSE HANDLING
            if f_reason == "tool_calls" and tool_calls:
                log.info(f"🔧 Processing {len(tool_calls)} tool calls")
                
                # Add assistant message with tool calls to conversation FIRST
                self.state.add_assistant_message(message.content or "", tool_calls)

                # Execute tools in parallel
                log.info(f"⚡ Starting parallel execution of {len(tool_calls)} tools")
                tool_results = await execute_tools_parallel(mcp_client, tool_calls, self.config.tool_timeout)
                
                # CRITICAL FIX: Process results and add ALL tool responses to conversation history
                # This prevents OpenAI API errors about missing tool responses
                successful_results = []
                failed_results = []
                
                for result in tool_results:
                    if result["success"]:
                        # Mark launch success for UI feedback
                        result_text = str(result["result"])
                        if any(keyword in result_text.lower() for keyword in ("launched", "opened", "created")):
                            self.successful_launch = True
                        
                        # Add successful tool result to conversation history
                        self.state.add_tool_message(result["call_id"], result_text)
                        successful_results.append(result)
                        
                        log.info(f"✅ Tool {result['name']} succeeded: {result_text[:100]}...")
                    else:
                        # CRITICAL FIX: Always add tool responses to conversation, even failures
                        # This maintains conversation history consistency and prevents API errors
                        error_message = f"Tool execution failed: {result['error']}"
                        self.state.add_tool_message(result["call_id"], error_message)
                        failed_results.append(result)
                        log.warning(f"❌ Tool {result['name']} failed: {result['error']}")
                
                # Generate appropriate response based on results
                if successful_results:
                    # Focus on successful results for user response
                    if len(successful_results) == 1:
                        # Single successful tool
                        result = successful_results[0]
                        assistant_response = extract_meaningful_content(
                            result["result"], result["name"]
                        )
                    else:
                        # Multiple successful tools
                        tool_names = [r["name"] for r in successful_results]
                        last_result = successful_results[-1]
                        base_response = extract_meaningful_content(
                            last_result["result"], last_result["name"]
                        )
                        assistant_response = f"I've completed {len(successful_results)} tasks: {', '.join(tool_names)}. {base_response}"
                    
                    log.info(f"📝 Generated response from {len(successful_results)} successful tools")
                    
                elif failed_results:
                    # All tools failed
                    if len(failed_results) == 1:
                        # Single failed tool - provide specific error
                        result = failed_results[0]
                        assistant_response = f"I had trouble with {result['name']}: {result['error']}. Please try again."
                    else:
                        # Multiple failed tools
                        failed_tools = [r["name"] for r in failed_results]
                        assistant_response = f"I had trouble with the following tools: {', '.join(failed_tools)}. Please try again."
                    
                    log.warning(f"❌ All {len(failed_results)} tools failed")
                    
                else:
                    # No tool results at all - something went very wrong
                    assistant_response = "I had trouble executing your request. Please try again."
                    log.error("No tool results returned from parallel execution")
                
                # Add the final assistant response to conversation
                if assistant_response:
                    self.state.add_assistant_message(assistant_response)

            # 3) Pure LLM answer (no tools called)
            elif assistant_response is None:
                assistant_response = message.content or "I'm not sure how to respond to that."
                
                # CRITICAL FIX: Detect and prevent hallucinated tool completions
                suspicious_phrases = [
                    "i've completed that task",
                    "task completed",
                    "i've done that",
                    "i've created",
                    "i've opened",
                    "i've launched",
                    "file created",
                    "folder created",
                    "successfully created",
                    "successfully opened",
                    "successfully launched"
                ]
                
                response_lower = assistant_response.lower()
                is_hallucinated = any(phrase in response_lower for phrase in suspicious_phrases)
                
                if is_hallucinated:
                    log.warning(f"🚨 DETECTED HALLUCINATED RESPONSE: {assistant_response}")
                    # Replace with honest response
                    assistant_response = "I understand what you're asking for. Let me try to help you with that."
                
                # Clean up tool-related artifacts in text responses
                if "[Tools used:" in assistant_response or ("Called " in assistant_response and "{" in assistant_response):
                    assistant_response = "I understand your request. How can I help you further?"
                
                self.state.add_assistant_message(assistant_response)
                log.info(f"💬 Text-only response: {assistant_response[:100]}...")
                
            return assistant_response

        except Exception as e:
            log.error(f"Error in process_user_input: {e}")
            error_response = "I encountered an error processing your request. Please try again."
            self.state.add_assistant_message(error_response)
            return error_response

    # -----------------------  public event-loop  ----------------------- #
    async def conversation_loop(self) -> None:
        """Main outer loop – records audio, calls helpers, speaks back."""
        log.info("Starting conversation loop")
        self.state.reset_conversation()

        # OpenAI client (for STT / TTS fallbacks)
        self.state.openai_client = OpenAI(api_key=self.config.openai_api_key)

        # launch background stuck checker
        self._stuck_task = asyncio.create_task(self.stuck.run())

        self.audio.start()

        try:
            async with get_mcp_client(self.state) as mcp:
                tools = await get_tools_cached(mcp, self.state)
                await speak_text("Hello! I'm listening and ready to help.", self.state, self.config)

                while self.state.running:
                    try:
                        current_mode = self.state.get_mode()
                        
                        # Handle different modes
                        if current_mode == AssistantMode.SLEEPING:
                            # In sleep mode, only listen for wake commands
                            log.debug("In sleep mode, listening for wake commands...")
                            buffer = await self.audio.record_until_silence(self.state, self.config)
                            
                            if not buffer:
                                await asyncio.sleep(0.1)
                                continue

                            user_txt = await transcribe_audio(buffer, self.state, self.config)
                            if not user_txt:
                                continue

                            log.info(f"Sleep mode input: {user_txt}")
                            
                            # Check for wake commands
                            if await self.commands.handle(user_txt):
                                # Wake command was handled
                                continue
                            # If not a wake command, ignore and stay sleeping
                            log.debug("Not a wake command, staying asleep")
                            continue
                        
                        elif current_mode == AssistantMode.DICTATION:
                            # In dictation mode, capture all speech as text
                            log.debug("In dictation mode, capturing speech...")
                            buffer = await self.audio.record_until_silence(self.state, self.config)
                            
                            if not buffer:
                                await asyncio.sleep(0.1)
                                continue

                            user_txt = await transcribe_audio(buffer, self.state, self.config)
                            if not user_txt:
                                continue

                            log.info(f"Dictation input: {user_txt}")
                            
                            # Check for end dictation commands
                            if any(phrase in user_txt.lower() for phrase in ["end dictation", "stop dictation", "finish dictation"]):
                                # End dictation mode
                                try:
                                    # Generate filename based on current time
                                    import time
                                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                                    filename = f"Dictation_{timestamp}.txt"
                                    
                                    # Save dictation to file
                                    file_path = self.state.save_dictation_to_file(filename)
                                    stats = self.state.get_dictation_stats()
                                    
                                    # Clear dictation buffer
                                    self.state.clear_dictation_buffer()
                                    
                                    # Return to listening mode
                                    self.state.set_mode(AssistantMode.LISTENING)
                                    
                                    # Speak confirmation
                                    response = f"Dictation saved to {filename}. Captured {stats.get('word_count', 0)} words in {stats.get('duration_formatted', 'unknown time')}."
                                    await speak_text(response, self.state, self.config)
                                    
                                except Exception as e:
                                    log.error(f"Error saving dictation: {e}")
                                    self.state.set_mode(AssistantMode.LISTENING)
                                    await speak_text("Sorry, I had trouble saving the dictation.", self.state, self.config)
                                
                                continue
                            else:
                                # Add text to dictation buffer
                                self.state.add_dictation_text(user_txt)
                                continue
                        
                        elif current_mode != AssistantMode.LISTENING:
                            await asyncio.sleep(0.1)
                            continue

                        # Normal listening mode processing
                        buffer = await self.audio.record_until_silence(self.state, self.config)
                        if not buffer:
                            continue

                        self.state.set_mode(AssistantMode.RECORDING)
                        user_txt = await transcribe_audio(buffer, self.state, self.config)
                        if not user_txt:
                            self.state.set_mode(AssistantMode.LISTENING)
                            continue

                        # Check for dictation mode activation
                        if any(phrase in user_txt.lower() for phrase in ["take dictation", "start dictation", "please take dictation", "begin dictation"]):
                            self.state.set_mode(AssistantMode.DICTATION)
                            await speak_text("Starting dictation mode. Say 'end dictation' when you're finished.", self.state, self.config)
                            continue

                        # Handle other commands
                        if await self.commands.handle(user_txt):
                            continue

                        self.state.set_mode(AssistantMode.PROCESSING)
                        reply = await self.process_user_input(user_txt, mcp, tools)

                        if reply:
                            # Check for launch acknowledgment setting
                            should_speak = True
                            if (self.successful_launch and 
                                hasattr(self.config, 'acknowledge_launches') and 
                                self.config.acknowledge_launches == False):
                                should_speak = False  # Silent success for launches
                            
                            if should_speak:
                                await speak_text(reply, self.state, self.config)

                        self.state.set_mode(AssistantMode.LISTENING)

                    except Exception as e:
                        log.error(f"Loop error: {e}")
                        self.state.set_mode(AssistantMode.ERROR)
                        await asyncio.sleep(self.config.reconnect_delay)
                        self.state.set_mode(AssistantMode.LISTENING)

        except Exception as e:
            log.error(f"Critical error in conversation loop: {e}")
        finally:
            # cleanup
            if self._stuck_task:
                self._stuck_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._stuck_task
            
            self.audio.stop()
            
            try:
                await shutdown_mcp_server(self.state)
            except Exception as e:
                log.error(f"Error during MCP shutdown: {e}")

    async def shutdown_system(self) -> None:
        """Gracefully shutdown the conversation system"""
        log.info("Shutting down conversation system...")
        self.state.running = False
        
        # Stop audio first
        self.audio.stop()
        
        # Cancel stuck detector
        if self._stuck_task:
            self._stuck_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stuck_task
        
        # Shutdown MCP
        try:
            await shutdown_mcp_server(self.state)
        except Exception as e:
            log.error(f"Error during shutdown: {e}")
        
        log.info("Conversation system shutdown complete")