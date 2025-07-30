# conversation.py - FIXED VERSION with proper response extraction
"""
Main conversation loop with FIXED response processing for tool results
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
        self.successful_launch = False

    def _extract_meaningful_content(self, tool_result: str, tool_name: str = None) -> str:
        """Extract meaningful content from tool results for speech"""
        if not tool_result:
            return "I couldn't get a result for that request."
        
        # Clean the result
        result = str(tool_result).strip()
        
        # STOCK PRICE EXTRACTION
        if tool_name == "get_current_stock_price" or "Stock Price for" in result:
            lines = result.split('\n')
            
            # Extract key information
            current_price = None
            previous_close = None
            change = None
            change_percent = None
            symbol = None
            
            for line in lines:
                line = line.strip()
                if "Current Price:" in line:
                    try:
                        current_price = line.split("Current Price:")[1].strip()
                    except:
                        pass
                elif "Previous Close:" in line:
                    try:
                        previous_close = line.split("Previous Close:")[1].strip()
                    except:
                        pass
                elif "Change:" in line:
                    try:
                        change_part = line.split("Change:")[1].strip()
                        if "(" in change_part:
                            change = change_part.split("(")[0].strip()
                            change_percent = change_part.split("(")[1].replace(")", "").strip()
                    except:
                        pass
                elif "Stock Price for" in line:
                    try:
                        symbol = line.split("Stock Price for")[1].split(":")[0].strip()
                    except:
                        pass
            
            # Build readable response
            if current_price and symbol:
                response = f"The current stock price of {symbol} is {current_price}"
                if change and change_percent:
                    response += f", {change} or {change_percent} from yesterday"
                return response + "."
            elif current_price:
                return f"The current stock price is {current_price}."
        
        # WEATHER EXTRACTION
        elif tool_name == "get_weather" or any(keyword in result.lower() for keyword in ["weather", "temperature", "°f", "°c"]):
            lines = result.split('\n')
            
            # Look for temperature and conditions
            temp_info = None
            conditions_info = None
            
            for line in lines:
                line = line.strip()
                if any(temp_word in line.lower() for temp_word in ["temperature", "°f", "°c", "degrees"]):
                    temp_info = line
                elif any(condition_word in line.lower() for condition_word in ["conditions", "sunny", "cloudy", "rain", "snow", "clear"]):
                    conditions_info = line
            
            # Build weather response
            if temp_info or conditions_info:
                response_parts = []
                if temp_info:
                    # Clean up temperature line
                    if "Temperature:" in temp_info:
                        temp_value = temp_info.split("Temperature:")[1].strip()
                        response_parts.append(f"The temperature is {temp_value}")
                    else:
                        response_parts.append(temp_info)
                
                if conditions_info:
                    if "Conditions:" in conditions_info:
                        cond_value = conditions_info.split("Conditions:")[1].strip()
                        response_parts.append(f"with {cond_value}")
                    else:
                        response_parts.append(conditions_info)
                
                if response_parts:
                    return " ".join(response_parts) + "."
        
        # WEB SEARCH EXTRACTION
        elif tool_name in ["web_search", "search_news"] or "Web search results" in result:
            lines = result.split('\n')
            
            # Extract the most relevant information
            key_info = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("Web search results"):
                    if line.startswith("Answer:"):
                        return line.replace("Answer:", "").strip()
                    elif line.startswith("Summary:"):
                        key_info.append(line.replace("Summary:", "").strip())
                    elif line.startswith("Related:") and len(key_info) < 2:
                        key_info.append(line.replace("Related:", "").strip())
            
            if key_info:
                return " ".join(key_info[:2])  # Limit to first 2 pieces of info
        
        # URL OPENING
        elif tool_name == "open_url" or "opened" in result.lower():
            if "nuggetnews.com" in result:
                return "I've opened Nugget News for you."
            elif "robertpkruger.com" in result:
                return "I've opened Robert P Kruger's website for you."
            elif "amazon.com" in result:
                return "I've opened Amazon for you."
            elif "reddit.com" in result:
                return "I've opened Reddit for you."
            elif "github.com" in result:
                return "I've opened GitHub for you."
            else:
                return "I've opened the website for you."
        
        # APP LAUNCHES
        elif tool_name == "launch_app" or tool_name == "launch_steam_game":
            if "launched" in result.lower() or "opened" in result.lower():
                return "I've launched the application for you."
            else:
                return "I've opened that for you."
        
        # GENERIC EXTRACTION - take first meaningful sentence
        sentences = result.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.startswith(('Web search', 'Tool result')):
                return sentence + "."
        
        # Fallback
        if len(result) > 200:
            return result[:150] + "..."
        return result

    def _should_use_tools_parameter(self, provider) -> bool:
        """Determine if the provider supports OpenAI-style tools parameter"""
        provider_class_name = provider.__class__.__name__
        
        if hasattr(provider, 'primary'):
            primary_class_name = provider.primary.__class__.__name__
            return primary_class_name in ["OpenAIChatProvider", "OpenAIChatCompletionProvider"]
        
        return provider_class_name in ["OpenAIChatProvider", "OpenAIChatCompletionProvider"]

    async def process_user_input(self, user_text: str, mcp_client, tools):
        """Process user input with FIXED response handling"""
        original_user_text = user_text
        
        # Add user message to history
        self.state.add_user_message(user_text)
        self.successful_launch = False
        
        try:
            # Get completion from provider
            completion = self.state.chat_provider.complete(
                messages=self.state.conversation_history,
                tools=tools,
                tool_choice="auto"
            )

            if not completion or not hasattr(completion, 'choices') or not completion.choices:
                logger.error("Invalid completion response from provider")
                return "I'm having trouble processing your request. Please try again."
            
            choice = completion.choices[0]
            
            if not choice or not hasattr(choice, 'message'):
                logger.error("Invalid choice in completion response")
                return "I'm having trouble processing your request. Please try again."
            
            message = choice.message
            
            if not message:
                logger.error("No message in completion response")
                return "I'm having trouble processing your request. Please try again."
            
            assistant_response = ""
            tool_results = []  # Track actual tool results
            
            # Check if we have tool calls
            finish_reason = getattr(choice, 'finish_reason', 'stop')
            tool_calls = getattr(message, 'tool_calls', None) or []
            
            if finish_reason == "tool_calls" and tool_calls:
                # Handle tool calls
                self.state.add_assistant_message(getattr(message, 'content', ''), tool_calls)
                
                # Execute each tool call
                for tool_call in tool_calls:
                    tool_name = None
                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                        tool_name = tool_call.function.name
                    elif isinstance(tool_call, dict) and 'function' in tool_call:
                        tool_name = tool_call['function'].get('name', 'unknown')
                    
                    # Check for interruption
                    if self.state.interrupt_flag.is_set() or self.state.get_mode() == AssistantMode.STUCK_CHECK:
                        logger.info("Processing interrupted")
                        self.state.set_mode(AssistantMode.LISTENING)
                        return None
                    
                    try:
                        tool_result = await call_tool_with_timeout(
                            mcp_client, tool_call, self.config.tool_timeout
                        )
                        
                        # Store the raw result
                        tool_results.append({
                            'name': tool_name,
                            'result': tool_result
                        })
                        
                        # Check if this was a successful launch
                        if "launched" in str(tool_result).lower() or "opened" in str(tool_result).lower():
                            self.successful_launch = True
                        
                        # Add to conversation history
                        tool_call_id = getattr(tool_call, 'id', 'unknown_id')
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get('id', 'unknown_id')
                        
                        self.state.add_tool_message(tool_call_id, str(tool_result))
                        
                    except Exception as tool_error:
                        logger.warning(f"Tool {tool_name} failed: {tool_error}")
                        # Handle tool failure with backup provider...
                        # [Keep existing error handling code]
                        return "I'm having trouble with that request. Please try again."
                
                # CRITICAL FIX: Instead of asking the model for follow-up, 
                # directly create a meaningful response from tool results
                if tool_results:
                    # Use the most recent/relevant tool result
                    primary_result = tool_results[-1]  # Last executed tool
                    
                    # Extract meaningful content for speech
                    assistant_response = self._extract_meaningful_content(
                        primary_result['result'], 
                        primary_result['name']
                    )
                    
                    logger.info(f"Extracted meaningful response: {assistant_response}")
                    
                    # Add the response to conversation history
                    self.state.conversation_history.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
                    
                else:
                    assistant_response = "I've completed that task for you."
                    self.state.conversation_history.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
            
            else:
                # Direct response without tool calls
                assistant_response = getattr(message, 'content', None) or "I'm not sure how to respond to that."
                self.state.conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return "I encountered an error processing your request. Please try again."

    # [Keep all other existing methods like handle_special_commands, etc.]
    
    async def handle_special_commands(self, text: str) -> bool:
        """Handle special commands and return True if handled"""
        lower_text = text.lower().strip()

        # Reset commands
        if any(phrase in lower_text for phrase in [
            "reset conversation", "clear conversation", "new conversation", 
            "start over", "reset chat", "clear chat", "new chat", "clear history"
        ]):
            self.state.reset_conversation()
            await speak_text("Conversation reset. Starting fresh!", self.state, self.config)
            return True

        # Force backup provider commands
        if any(phrase in lower_text for phrase in [
            "use openai", "use gpt", "use frontier", "use backup", 
            "force backup", "switch to openai", "use cloud model"
        ]):
            if hasattr(self.state.chat_provider, 'force_backup_next'):
                self.state.chat_provider.force_backup_next = True
            await speak_text("I'll use the frontier model for the next response.", self.state, self.config)
            return True
        
        # Launch acknowledgment toggle
        if any(phrase in lower_text for phrase in [
            "turn on launch acknowledgment", "enable launch acknowledgment"
        ]):
            self.config.acknowledge_launches = True
            await speak_text("Launch acknowledgments enabled.", self.state, self.config)
            return True
            
        if any(phrase in lower_text for phrase in [
            "turn off launch acknowledgment", "disable launch acknowledgment"
        ]):
            self.config.acknowledge_launches = False
            await speak_text("Launch acknowledgments disabled.", self.state, self.config)
            return True

        # Sleep commands
        if any(phrase in lower_text for phrase in [
            "go to sleep", "sleep mode", "sleep now"
        ]):
            await speak_text("Going to sleep. Say 'wake up' or 'hello' to wake me.", self.state, self.config)
            self.state.set_mode(AssistantMode.SLEEPING)
            return True
        
        # Wake commands (only when sleeping)
        if self.state.get_mode() == AssistantMode.SLEEPING:
            if any(phrase in lower_text for phrase in [
                "wake up", "wake", "hello", "hey"
            ]):
                self.state.set_mode(AssistantMode.LISTENING)
                await speak_text("I'm awake! How can I help you?", self.state, self.config)
                return True
            else:
                return True  # Ignore other commands when sleeping
        
        # System control
        if any(phrase in lower_text for phrase in [
            "exit", "quit", "goodbye", "shut down"
        ]):
            await speak_text("Goodbye! Shutting down the system...", self.state, self.config)
            await self.shutdown_system()
            return True

        # Help command
        if any(phrase in lower_text for phrase in [
            "help", "what can you do"
        ]):
            help_text = """I can help you with:
            
            Applications: Open apps like Excel, Word, Chrome, Steam
            Web: Get stock prices, weather, search the web, open websites
            Files: Create folders and files, read files, list files
            
            Special commands: Reset conversation, go to sleep, help
            
            Just speak naturally and I'll help you!"""
            
            await speak_text(help_text, self.state, self.config)
            return True

        return False

    async def stuck_detection_task(self):
        """Background task to check if assistant is stuck"""
        while self.state.running:
            try:
                await asyncio.sleep(self.config.stuck_check_interval)
                
                if self.state.is_stuck(self.config.processing_timeout):
                    logger.warning(f"Assistant appears stuck")
                    self.state.set_mode(AssistantMode.STUCK_CHECK)
                    
                    # Listen for wake phrase
                    audio_buffer = await self.audio_recorder.record_until_silence(
                        self.state, self.config, check_stuck_phrase=True
                    )
                    
                    if audio_buffer:
                        try:
                            text = await transcribe_audio(
                                audio_buffer, self.state, self.config, check_stuck_phrase=True
                            )
                            
                            if text:
                                text_normalized = text.strip().lower().replace(",", "").replace("?", "")
                                wake_words = set(self.config.stuck_phrase.split())
                                detected_words = set(text_normalized.split())
                                if len(wake_words.intersection(detected_words)) >= 2:
                                    logger.info("Wake phrase detected! Resetting")
                                    self.state.interrupt_flag.set()
                                    self.state.set_mode(AssistantMode.LISTENING)
                                    asyncio.create_task(
                                        speak_text("I'm back! How can I help you?", 
                                                 self.state, self.config)
                                    )
                        except Exception as e:
                            logger.error(f"Error in stuck phrase detection: {e}")
                    
            except Exception as e:
                logger.error(f"Error in stuck detection task: {e}")
                await asyncio.sleep(1)

    async def conversation_loop(self):
        """Main conversation loop"""
        logger.info("Starting conversation loop with enhanced response processing")
        
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
            await speak_text("Hello! I'm listening and ready to help.", self.state, self.config)
                       
            while self.state.running:
                try:
                    current_mode = self.state.get_mode()
                    
                    # Handle sleep mode
                    if current_mode == AssistantMode.SLEEPING:
                        # Temporarily listen for wake commands
                        self.state.set_mode(AssistantMode.LISTENING)
                        audio_buffer = await self.audio_recorder.record_until_silence(
                            self.state, self.config
                        )
                        self.state.set_mode(AssistantMode.SLEEPING)
                        
                        if audio_buffer:
                            try:
                                user_text = await transcribe_audio(audio_buffer, self.state, self.config, check_stuck_phrase=True)
                                if user_text and any(word in user_text.lower() for word in ["wake", "hello", "hey"]):
                                    self.state.set_mode(AssistantMode.LISTENING)
                                    await speak_text("I'm awake! How can I help you?", self.state, self.config)
                            except:
                                pass
                        continue
                    
                    # Only process when listening
                    elif current_mode != AssistantMode.LISTENING:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Record audio
                    audio_buffer = await self.audio_recorder.record_until_silence(
                        self.state, self.config
                    )
                    
                    if not audio_buffer:
                        continue
                    
                    self.state.set_mode(AssistantMode.RECORDING)
                    
                    # Transcribe
                    user_text = await transcribe_audio(audio_buffer, self.state, self.config)
                    if not user_text:
                        self.state.set_mode(AssistantMode.LISTENING)
                        continue
                    
                    logger.info(f"User said: {user_text}")
                    
                    # Handle special commands
                    if await self.handle_special_commands(user_text):
                        if not self.state.running:
                            break
                        continue
                    
                    # Process user input
                    self.state.set_mode(AssistantMode.PROCESSING)
                    assistant_response = await self.process_user_input(user_text, mcp_client, tools)

                    provider_used = getattr(self.state.chat_provider, "last_provider", "unknown")
                    logger.info(f"Turn answered by: {provider_used}")
                    
                    # Speak response
                    if assistant_response and not self.state.interrupt_flag.is_set():
                        acknowledge_launches = getattr(self.config, 'acknowledge_launches', True)
                        
                        if self.successful_launch and not acknowledge_launches:
                            self.state.set_mode(AssistantMode.LISTENING)
                            logger.info("Launch successful but acknowledgment disabled")
                        else:
                            await speak_text(assistant_response, self.state, self.config)
                    else:
                        self.state.set_mode(AssistantMode.LISTENING)
                        
                except Exception as e:
                    logger.error(f"Error in conversation loop: {e}")
                    self.state.set_mode(AssistantMode.ERROR)
                    await asyncio.sleep(self.config.reconnect_delay)
                    self.state.set_mode(AssistantMode.LISTENING)
                    continue
        
        # Cleanup
        if self.stuck_task:
            self.stuck_task.cancel()
            try:
                await self.stuck_task
            except asyncio.CancelledError:
                pass

    async def shutdown_system(self):
        """Properly shut down the system"""
        logger.info("Initiating system shutdown...")
        self.audio_recorder.stop()
        await shutdown_mcp_server(self.state)
        self.state.running = False
        logger.info("System shutdown complete")