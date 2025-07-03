# debug_cli.py - Minimal version to find the exact issue
"""
Minimal CLI to isolate what's consuming stdin
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from voice_assistant.config import Config, setup_logging
from voice_assistant.state import AssistantState

async def debug_main():
    """Debug version - step by step initialization"""
    print("=== DEBUG CLI ===")
    print("Step 1: Basic Python input test")
    
    # Test 1: Raw Python input
    user_input = input("Test input 1: ").strip()
    print(f"Got: '{user_input}'")
    
    user_input = input("Test input 2: ").strip()
    print(f"Got: '{user_input}'")
    
    print("\nStep 2: Config initialization")
    config = Config.from_env()
    
    user_input = input("Test input 3 (after config): ").strip()
    print(f"Got: '{user_input}'")
    
    print("\nStep 3: Setup logging")
    setup_logging(config)
    
    user_input = input("Test input 4 (after logging): ").strip()
    print(f"Got: '{user_input}'")
    
    print("\nStep 4: AssistantState initialization")
    state = AssistantState(config.vad_aggressiveness)
    
    user_input = input("Test input 5 (after AssistantState): ").strip()
    print(f"Got: '{user_input}'")
    
    print("\nStep 5: MCP client")
    try:
        from voice_assistant.mcp_client import get_mcp_client, get_tools
        
        async with get_mcp_client(state) as mcp_client:
            user_input = input("Test input 6 (after MCP client): ").strip()
            print(f"Got: '{user_input}'")
            
            tools = await get_tools(mcp_client, state)
            
            user_input = input("Test input 7 (after tools): ").strip()
            print(f"Got: '{user_input}'")
            
    except Exception as e:
        print(f"MCP error: {e}")
    
    print("\nStep 6: Provider initialization")
    try:
        from voice_assistant.model_providers.factory import ModelProviderFactory
        from voice_assistant.model_providers.failover_chat import FailoverChatProvider
        
        user_input = input("Test input 8 (before providers): ").strip()
        print(f"Got: '{user_input}'")
        
        # Build the primary (via Ollama)
        primary_chat = ModelProviderFactory.create_chat_provider(
            provider_type="ollama",
            model=config.local_chat_model,
            host=config.ollama_host
        )
        
        user_input = input("Test input 9 (after primary provider): ").strip()
        print(f"Got: '{user_input}'")

        # Build the backup (OpenAI)
        backup_chat = ModelProviderFactory.create_chat_provider(
            provider_type="openai",
            api_key=config.openai_api_key,
            model=config.frontier_chat_model
        )
        
        user_input = input("Test input 10 (after backup provider): ").strip()
        print(f"Got: '{user_input}'")

        # Wrap them
        state.chat_provider = FailoverChatProvider(
            primary=primary_chat,
            backup=backup_chat,
            timeout=config.local_chat_timeout or 30
        )
        
        user_input = input("Test input 11 (after failover provider): ").strip()
        print(f"Got: '{user_input}'")
        
    except Exception as e:
        print(f"Provider error: {e}")
    
    print("\n=== FINAL TEST ===")
    for i in range(3):
        user_input = input(f"Final test {i+1}: ").strip()
        print(f"Got: '{user_input}'")

if __name__ == "__main__":
    asyncio.run(debug_main())