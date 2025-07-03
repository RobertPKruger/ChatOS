# steam_debug.py - Debug Steam games tool
"""
Quick debug script to test the Steam games tool directly
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from voice_assistant.config import Config, setup_logging
from voice_assistant.state import AssistantState
from voice_assistant.mcp_client import get_mcp_client, get_tools

async def debug_steam_games():
    """Debug the Steam games tool directly"""
    config = Config.from_env()
    setup_logging(config)
    state = AssistantState(config.vad_aggressiveness)
    
    print("🔍 Debugging Steam games tool...")
    
    async with get_mcp_client(state) as mcp_client:
        print("✅ MCP client connected")
        
        # Test the tool directly
        try:
            result = await mcp_client.call_tool(
                name="list_steam_games",
                arguments={}
            )
            
            print(f"🎮 Raw tool result:")
            print(f"Type: {type(result)}")
            print(f"Content: {result}")
            
            if hasattr(result, 'content'):
                for item in result.content:
                    print(f"  - {item}")
            
        except Exception as e:
            print(f"❌ Tool call failed: {e}")
            
        # Also test listing available tools
        try:
            tools_result = await mcp_client.list_tools()
            steam_tools = [tool for tool in tools_result.tools if 'steam' in tool.name.lower()]
            print(f"\n🔧 Available Steam tools:")
            for tool in steam_tools:
                print(f"  - {tool.name}: {tool.description}")
                
        except Exception as e:
            print(f"❌ Failed to list tools: {e}")

if __name__ == "__main__":
    asyncio.run(debug_steam_games())