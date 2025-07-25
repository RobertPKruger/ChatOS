# voice_assistant/mcp_client.py - UPDATED WITH CACHING
"""
MCP (Model Context Protocol) client management with performance optimizations
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

# Global tool cache for performance
TOOLS_CACHE = None
TOOLS_CACHE_TIME = 0
CACHE_DURATION = 300  # 5 minutes cache

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

async def get_tools_cached(mcp_client: Client, state: AssistantState) -> List[Dict[str, Any]]:
    """Get tools with global caching for better performance"""
    global TOOLS_CACHE, TOOLS_CACHE_TIME
    
    current_time = time.time()
    
    # DEBUG: Print cache status (Windows-safe)
    print("DEBUG: get_tools_cached called")
    print(f"DEBUG: TOOLS_CACHE exists: {TOOLS_CACHE is not None}")
    print(f"DEBUG: Cache time: {TOOLS_CACHE_TIME}, Current: {current_time}")
    print(f"DEBUG: Cache age: {current_time - TOOLS_CACHE_TIME:.1f}s, Duration: {CACHE_DURATION}s")
    
    # Return cached tools if still valid
    if TOOLS_CACHE and (current_time - TOOLS_CACHE_TIME) < CACHE_DURATION:
        print(f"Using cached tools ({len(TOOLS_CACHE)} tools)")
        logger.debug(f"Using cached tools ({len(TOOLS_CACHE)} tools)")
        return TOOLS_CACHE
    
    # Cache expired or doesn't exist - reload
    print("Refreshing tool cache...")
    logger.info("Refreshing tool cache...")
    start_time = time.time()
    
    try:
        TOOLS_CACHE = await get_tools(mcp_client, state, use_cache=False)
        TOOLS_CACHE_TIME = current_time
        
        load_time = time.time() - start_time
        print(f"Cached {len(TOOLS_CACHE)} tools in {load_time:.2f}s")
        logger.info(f"Cached {len(TOOLS_CACHE)} tools in {load_time:.2f}s")
        
        return TOOLS_CACHE
        
    except Exception as e:
        logger.error(f"Failed to refresh tool cache: {e}")
        # Return old cache if available, otherwise empty list
        return TOOLS_CACHE if TOOLS_CACHE else []

def invalidate_tools_cache():
    """Force cache refresh on next request"""
    global TOOLS_CACHE_TIME
    TOOLS_CACHE_TIME = 0
    logger.info("Tool cache invalidated")

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

async def call_tool_with_timeout_optimized(mcp_client: Client, call, timeout: float = 10) -> str:
    """Optimized tool execution with faster timeout and better error handling"""
    start_time = time.time()
    
    try:
        # Use the existing call_tool_with_timeout but with timing
        result = await call_tool_with_timeout(mcp_client, call, timeout)
        
        execution_time = time.time() - start_time
        logger.debug(f"Tool executed in {execution_time:.2f}s")
        
        return result
        
    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        logger.warning(f"Tool timed out after {execution_time:.2f}s")
        raise
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Tool failed after {execution_time:.2f}s: {e}")
        raise

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
        raise Exception(error_msg)  # Changed: Raise exception instead of returning
    
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
            raise Exception(error_msg)  # Changed: Raise exception instead of returning
    
    try:
        # Execute with timeout
        result = await asyncio.wait_for(execute_tool(), timeout=timeout)
        
        # NEW: Check if the result indicates failure
        result_lower = result.lower()
        
        # Define failure indicators from your MCP server
        failure_indicators = [
            "not found",
            "failed to launch",
            "application not found", 
            "path not found",
            "errors:",
            "did you mean:",
            "use 'list_apps' to see",
            "no apps found",
            "only .exe files are allowed",
            "path not in allowed directories",
            "executable not found",
            "invalid json",
            "permission denied",
            "access denied",
            "file not found",
            "command not found"
        ]
        
        # Check if result contains any failure indicators
        if any(indicator in result_lower for indicator in failure_indicators):
            error_msg = f"Tool {name} failed: {result}"
            logger.error(error_msg)
            raise Exception(error_msg)  # This will trigger failover
        
        # Success case
        logger.info(f"Tool {name} completed successfully")
        return result
        
    except asyncio.TimeoutError:
        error_msg = f"Tool {name} timed out after {timeout}s"
        logger.error(error_msg)
        raise Exception(error_msg)  # Changed: Raise exception instead of returning

async def shutdown_mcp_server(state: AssistantState):
    """Properly shut down MCP server and clear caches"""
    global TOOLS_CACHE, TOOLS_CACHE_TIME
    
    # Clear global cache
    TOOLS_CACHE = None
    TOOLS_CACHE_TIME = 0
    logger.info("Cleared tool cache on shutdown")
    
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

# Backward compatibility aliases
get_tools_fast = get_tools_cached
call_tool_fast = call_tool_with_timeout_optimized