# cli_chat_host.py
"""
Command-line interface for ChatOS
Uses the same backend as the voice interface but with text input/output
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from voice_assistant.config import Config, setup_logging
from voice_assistant.state import AssistantState, AssistantMode
from voice_assistant.conversation import ConversationManager
from voice_assistant.mcp_client import get_mcp_client, get_tools, shutdown_mcp_server
from voice_assistant.utils import signal_handler

logger = logging.getLogger(__name__)

class CLIInterface:
    """Command-line interface that reuses the conversation logic"""
    
    def __init__(self, config: Config, state: AssistantState):
        self.config = config
        self.state = state
        
    async def process_user_input(self, user_text: str, mcp_client, tools):
        """Process user input using the same logic as voice interface"""
        # Add user message to history
        self.state.conversation_history.append({"role": "user", "content": user_text})
        
        try:
            # Determine if we should pass tools parameter based on the provider
            conversation_manager = ConversationManager(self.config, self.state, None)
            use_tools = conversation_manager._should_use_tools_parameter(self.state.chat_provider)
            
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
                
                print(f"🔧 Executing {len(message.tool_calls)} tool(s)...")
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    try:
                        from voice_assistant.mcp_client import call_tool_with_timeout
                        tool_result = await call_tool_with_timeout(
                            mcp_client, tool_call, self.config.tool_timeout
                        )
                        
                        print(f"   ✅ {tool_call.function.name}: {tool_result[:100]}...")
                        
                        # Add tool result to history
                        self.state.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        
                    except Exception as tool_error:
                        print(f"   ❌ {tool_call.function.name} failed: {tool_error}")
                        
                        # Clean up and trigger manual fallback (same as voice interface)
                        if (self.state.conversation_history and 
                            self.state.conversation_history[-1].get("role") == "assistant" and
                            "tool_calls" in self.state.conversation_history[-1]):
                            self.state.conversation_history.pop()
                        
                        # Manually trigger OpenAI fallback
                        print("🔄 Tool failed - triggering OpenAI fallback...")
                        
                        try:
                            from voice_assistant.model_providers.openai_chat import OpenAIChatProvider
                            backup_provider = OpenAIChatProvider(
                                api_key=self.config.openai_api_key,
                                model=self.config.frontier_chat_model
                            )
                            
                            # Clean and validate conversation history (same logic as voice interface)
                            validated_history = self._clean_conversation_history()
                            
                            fallback_completion = backup_provider.complete(
                                messages=validated_history,
                                tools=tools,
                                tool_choice="auto"
                            )
                            
                            fallback_choice = fallback_completion.choices[0]
                            fallback_message = fallback_choice.message
                            
                            if fallback_choice.finish_reason == "tool_calls" and fallback_message.tool_calls:
                                # Handle OpenAI's tool calls
                                self.state.conversation_history.append({
                                    "role": "assistant",
                                    "content": fallback_message.content,
                                    "tool_calls": fallback_message.tool_calls
                                })
                                
                                print(f"🔧 OpenAI executing {len(fallback_message.tool_calls)} tool(s)...")
                                
                                for backup_tool_call in fallback_message.tool_calls:
                                    try:
                                        backup_result = await call_tool_with_timeout(
                                            mcp_client, backup_tool_call, self.config.tool_timeout
                                        )
                                        
                                        print(f"   ✅ {backup_tool_call.function.name}: {backup_result[:100]}...")
                                        
                                        self.state.conversation_history.append({
                                            "role": "tool",
                                            "tool_call_id": backup_tool_call.id,
                                            "content": backup_result
                                        })
                                    except Exception as backup_tool_error:
                                        print(f"   ❌ Backup tool also failed: {backup_tool_error}")
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
                                
                                print("🔄 Response from: OpenAI (fallback)")
                                return final_response
                            else:
                                # OpenAI gave direct response
                                response = fallback_message.content or "I can help you with that."
                                self.state.conversation_history.append({
                                    "role": "assistant",
                                    "content": response
                                })
                                
                                print("🔄 Response from: OpenAI (fallback)")
                                return response
                                
                        except Exception as fallback_error:
                            print(f"❌ Manual fallback also failed: {fallback_error}")
                            return "I'm having trouble processing your request. Please try again."
                
                # Get the model's follow-up response if not interrupted and all tools succeeded
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
            
            # Log which provider was used
            provider_used = getattr(self.state.chat_provider, "last_provider", "unknown")
            print(f"🔄 Response from: {provider_used}")
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return "I encountered an error processing your request. Please try again."
    
    def _clean_conversation_history(self):
        """Clean and validate conversation history for OpenAI (same as voice interface)"""
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
        
        return validated_history

def initialize_providers(config: Config, state: AssistantState):
    """Initialize model providers (same as voice interface)"""
    try:
        from voice_assistant.model_providers.factory import ModelProviderFactory
        from voice_assistant.model_providers.failover_chat import FailoverChatProvider
        
        logger.info("🔧 Initializing providers...")
        
        # Build the primary (via Ollama)
        primary_chat = ModelProviderFactory.create_chat_provider(
            provider_type="ollama",
            model=config.local_chat_model,
            host=config.ollama_host
        )

        # Build the backup (OpenAI)
        backup_chat = ModelProviderFactory.create_chat_provider(
            provider_type="openai",
            api_key=config.openai_api_key,
            model=config.frontier_chat_model
        )

        # Wrap them
        state.chat_provider = FailoverChatProvider(
            primary=primary_chat,
            backup=backup_chat,
            timeout=config.local_chat_timeout or 30
        )

        logger.info("✅ Chat provider: local-first with frontier fallback")
        
    except Exception as e:
        logger.error(f"Failed to initialize providers: {e}")
        raise

async def cli_main():
    """Main CLI loop"""
    config = Config.from_env()
    setup_logging(config)
    
    logger.info("🖥️  ChatOS CLI Mode Starting...")
    
    state = AssistantState(config.vad_aggressiveness)
    cli = CLIInterface(config, state)
    
    # Initialize providers
    initialize_providers(config, state)
    
    # Initialize conversation
    state.reset_conversation()
    
    print("\n" + "="*60)
    print("🖥️  ChatOS - Command Line Interface")
    print("="*60)
    print("💡 Type 'exit', 'quit', or 'goodbye' to quit")
    print("💡 Type 'reset' or 'clear' to start a new conversation")
    print("💡 Type 'help' for available commands")
    print("="*60 + "\n")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, state))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, state))
    
    async with get_mcp_client(state) as mcp_client:
        state.mcp_client = mcp_client
        
        # Load tools
        tools = await get_tools(mcp_client, state)
        print(f"🔧 Loaded {len(tools)} tools")
        
        print("🤖 Assistant: Hello! I'm ready to help. What can I do for you?")
        
        while state.running:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ["exit", "quit", "goodbye"]:
                    print("🤖 Assistant: Goodbye! Shutting down...")
                    break
                    
                if user_input.lower() in ["reset", "clear", "new chat"]:
                    state.reset_conversation()
                    print("🤖 Assistant: Starting a new conversation.")
                    continue
                
                if user_input.lower() == "help":
                    print("""
🔧 Available Commands:
  • exit, quit, goodbye - Quit the application
  • reset, clear, new chat - Start a new conversation
  • help - Show this help message
  
🎯 Example Requests:
  • Open notepad
  • Launch Word
  • What time is it?
  • Open Steam and launch Age of Mythology
  • List my Steam games
  • What's 2 + 2?
                    """)
                    continue
                
                # Process the user input
                print("🤔 Processing...")
                response = await cli.process_user_input(user_input, mcp_client, tools)
                
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🤖 Assistant: Goodbye! (Ctrl+C received)")
                break
            except Exception as e:
                logger.error(f"Error in CLI loop: {e}")
                print(f"❌ Error: {e}")
                continue
    
    # Cleanup
    await shutdown_mcp_server(state)
    print("🔧 Shutdown complete")

if __name__ == "__main__":
    asyncio.run(cli_main())