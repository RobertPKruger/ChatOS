"""
ConversationManager coordinates the recorder, the LLM provider
and the helper objects – but delegates heavy lifting to helpers.
"""

from __future__ import annotations

import asyncio
import logging
import contextlib
from typing import Optional, List, Dict

from openai import OpenAI

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
    """High-level conversation FSM."""

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
        Handles one user utterance, including tool execution and
        response derivation.
        """
        self.state.add_user_message(user_text)
        self.successful_launch = False

        try:
            # 1) get a completion from whichever provider is active
            completion = self.state.chat_provider.complete(
                messages=self.state.conversation_history,
                tools=tools,
                tool_choice="auto",
            )
            if not completion or not completion.choices:
                log.error("No completion from provider.")
                return "I'm having trouble processing your request. Please try again."

            choice   = completion.choices[0]
            message  = choice.message
            f_reason = getattr(choice, "finish_reason", "stop")
            tool_calls = getattr(message, "tool_calls", None) or []

            assistant_response: str | None = None
            tool_results: List[Dict[str, str]] = []

            # 2) Execute any requested tools
            if f_reason == "tool_calls" and tool_calls:
                self.state.add_assistant_message(message.content or "", tool_calls)

                for call in tool_calls:
                    # Handle both OpenAI tool calls and MockToolCall objects
                    try:
                        if hasattr(call, "function"):
                            # OpenAI tool call format
                            tool_name = call.function.name
                            call_id = getattr(call, "id", "unknown_id")
                        elif hasattr(call, "to_dict"):
                            # MockToolCall with to_dict method
                            call_dict = call.to_dict()
                            tool_name = call_dict["function"]["name"]
                            call_id = call_dict["id"]
                        elif isinstance(call, dict):
                            # Dictionary format
                            tool_name = call.get("function", {}).get("name", "unknown")
                            call_id = call.get("id", "unknown_id")
                        else:
                            # Fallback for other object types
                            tool_name = str(getattr(call, "name", "unknown"))
                            call_id = str(getattr(call, "id", "unknown_id"))
                        
                        log.info(f"Executing tool: {tool_name}")
                        
                        result = await call_tool_with_timeout(
                            mcp_client, call, self.config.tool_timeout
                        )
                        tool_results.append({"name": tool_name, "result": result})

                        # Mark launch success
                        if any(k in str(result).lower() for k in ("launched", "opened")):
                            self.successful_launch = True

                        # persist - use the extracted call_id
                        self.state.add_tool_message(call_id, str(result))

                    except Exception as e:
                        tool_name_safe = tool_name if 'tool_name' in locals() else 'unknown'
                        log.warning(f"Tool {tool_name_safe} failed: {e}")
                        return "I'm having trouble with that request. Please try again."

                # derive human-friendly answer from the last tool result
                if tool_results:
                    last = tool_results[-1]
                    assistant_response = extract_meaningful_content(
                        last["result"], last["name"]
                    )
                    self.state.conversation_history.append(
                        {"role": "assistant", "content": assistant_response}
                    )

            # 3) Pure LLM answer
            if assistant_response is None:
                assistant_response = message.content or "I'm not sure how to respond to that."
                if "[Tools used:" in assistant_response or ("Called " in assistant_response and "{" in assistant_response):
                    assistant_response = "I've completed that task for you."
                self.state.conversation_history.append(
                    {"role": "assistant", "content": assistant_response}
                )
            return assistant_response

        except Exception as e:
            log.error(f"Error in process_user_input: {e}")
            return "I encountered an error processing your request. Please try again."

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

                        if await self.commands.handle(user_txt):
                            continue

                        self.state.set_mode(AssistantMode.PROCESSING)
                        reply = await self.process_user_input(user_txt, mcp, tools)

                        if reply:
                            if self.successful_launch and getattr(self.config, 'acknowledge_launches', True) == False:
                                pass  # silent success
                            else:
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