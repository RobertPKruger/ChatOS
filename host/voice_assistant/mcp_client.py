# voice_assistant/mcp_client.py
"""
MCP (Model Context Protocol) client management
"""

import asyncio
import json
import logging
import os
import sys
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastmcp import Client

from .state import AssistantState

logger = logging.getLogger(__name__)

@asynccontextmanager
async def get_mcp_client(state: AssistantState):
    """Context manager for MCP client with subprocess management"""
    server_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "mcp_os", "server.py")
    )
    
    client = None
    try:
        # Start the MCP server as a subprocess if not already running
        if not state.mcp_process or state.mcp_process.poll() is not None:
            logger.info(f"Starting MCP server: {server_path}")
            state.mcp_process = subprocess.Popen(
                [sys.executable, server_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Give the server time to start
            await asyncio.sleep(1)
        
        client = Client(server_path)
        async with client:
            logger.info("MCP client connected successfully")
            yield client
    except Exception as e:
        logger.error(f"MCP client error: {e}")
        if client:
            try:
                await client.close()
            except:
                pass
        raise

async def get_tools(mcp_client: Client, state: AssistantState, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Get tools from MCP server with caching"""
    current_time = time.time()
    
    # Use cache if available and recent
    if (use_cache and 
        state.tools_cache and 
        current_time - state.tools_cache_time < 60):  # 1 minute cache
        return state.tools_cache
    
    tools = []
    try:
        for tool in await mcp_client.list_tools():
            try:
                # Try to get OpenAI schema
                if hasattr(tool, 'openai_schema'):
                    spec = tool.openai_schema()
                else:
                    # Fallback schema construction
                    spec = {
                        "name": tool.name,
                        "description": getattr(tool, "description", ""),
                        "parameters": getattr(tool, "inputSchema", {
                            "type": "object", 
                            "properties": {}
                        })
                    }
                
                # Ensure required fields
                if "name" not in spec:
                    spec["name"] = tool.name
                if "description" not in spec:
                    spec["description"] = getattr(tool, "description", "")
                
                tools.append({"type": "function", "function": spec})
                logger.debug(f"Added tool: {tool.name}")
                
            except Exception as e:
                logger.warning(f"Failed to process tool {tool.name}: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        # Return cached tools if available
        if state.tools_cache:
            logger.info("Using cached tools due to listing failure")
            return state.tools_cache
        raise
    
    # Update cache
    state.tools_cache = tools
    state.tools_cache_time = current_time
    logger.info(f"Loaded {len(tools)} tools")
    
    return tools

async def call_tool_with_timeout(mcp_client: Client, call, timeout: float) -> str:
    """Execute a tool call with timeout and error handling"""
    # Normalize call format
    if hasattr(call, "function"):
        name = call.function.name
        args_json = call.function.arguments or "{}"
        call_id = getattr(call, "id", str(uuid.uuid4()))
    else:
        name = call.get("name") or call["function"]["name"]
        args_json = call.get("function", call).get("arguments", "{}")
        call_id = call.get("id", str(uuid.uuid4()))
    
    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in tool arguments: {e}"
        logger.error(error_msg)
        return error_msg
    
    logger.info(f"Calling tool: {name} with args: {args}")
    
    async def execute_tool():
        try:
            response = await mcp_client.call_tool(name, args)
            
            # Handle different response types
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "content"):
                if isinstance(response.content, list):
                    text_parts = []
                    for content in response.content:
                        if hasattr(content, "text"):
                            text_parts.append(content.text)
                        elif hasattr(content, "content"):
                            text_parts.append(str(content.content))
                        else:
                            text_parts.append(str(content))
                    return " ".join(text_parts)
                else:
                    return str(response.content)
            else:
                return str(response)
                
        except Exception as e:
            error_msg = f"Tool {name} failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    try:
        # Execute with timeout
        result = await asyncio.wait_for(execute_tool(), timeout=timeout)
        logger.info(f"Tool {name} completed successfully")
        return result
    except asyncio.TimeoutError:
        error_msg = f"Tool {name} timed out after {timeout}s"
        logger.error(error_msg)
        return error_msg

async def shutdown_mcp_server(state: AssistantState):
    """Properly shut down MCP server"""
    # Close MCP client
    if state.mcp_client:
        try:
            await state.mcp_client.close()
        except Exception as e:
            logger.error(f"Error closing MCP client: {e}")
    
    # Terminate MCP server subprocess
    if state.mcp_process:
        try:
            logger.info("Terminating MCP server process...")
            state.mcp_process.terminate()
            
            # Wait for graceful shutdown
            try:
                state.mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("MCP server didn't terminate gracefully, forcing kill...")
                state.mcp_process.kill()
                state.mcp_process.wait()
            
            logger.info("MCP server process terminated")
        except Exception as e:
            logger.error(f"Error terminating MCP server: {e}")