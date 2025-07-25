# voice_assistant/mcp_client.py - PERFORMANCE OPTIMIZED
"""
MCP (Model Context Protocol) client management with significant performance optimizations
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

# Connection pool for faster MCP connections
MCP_CONNECTION_POOL = {}
MCP_PROCESS_SHARED = None

@asynccontextmanager
async def get_mcp_client(state: AssistantState):
    """Context manager for MCP client with connection pooling and subprocess reuse"""
    global MCP_PROCESS_SHARED
    
    server_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "mcp_os", "server.py")
    )
    
    client = None
    try:
        # Reuse shared MCP process for better performance
        if not MCP_PROCESS_SHARED or MCP_PROCESS_SHARED.poll() is not None:
            logger.info(f"Starting shared MCP server: {server_path}")
            MCP_PROCESS_SHARED = subprocess.Popen(
                [sys.executable, server_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            state.mcp_process = MCP_PROCESS_SHARED
            
            # Reduced startup wait time
            await asyncio.sleep(0.5)  # Down from 1.0s
        else:
            state.mcp_process = MCP_PROCESS_SHARED
        
        # Check if we have a cached client connection
        client_key = f"client_{id(state)}"
        if client_key in MCP_CONNECTION_POOL:
            client = MCP_CONNECTION_POOL[client_key]
        else:
            client = Client(server_path)
            MCP_CONNECTION_POOL[client_key] = client
        
        async with client:
            logger.info("MCP client connected successfully (cached/pooled)")
            yield client
            
    except Exception as e:
        logger.error(f"MCP client error: {e}")
        # Clean up failed connection from pool
        client_key = f"client_{id(state)}"
        if client_key in MCP_CONNECTION_POOL:
            del MCP_CONNECTION_POOL[client_key]
        
        if client:
            try:
                await client.close()
            except:
                pass
        raise

async def get_tools_cached(mcp_client: Client, state: AssistantState) -> List[Dict[str, Any]]:
    """Get tools with aggressive caching for better performance"""
    global TOOLS_CACHE, TOOLS_CACHE_TIME
    
    current_time = time.time()
    
    # Return cached tools if still valid
    if TOOLS_CACHE and (current_time - TOOLS_CACHE_TIME) < CACHE_DURATION:
        logger.debug(f"Using cached tools ({len(TOOLS_CACHE)} tools)")
        return TOOLS_CACHE
    
    # Cache expired or doesn't exist - reload with timeout
    logger.info("Refreshing tool cache...")
    start_time = time.time()
    
    try:
        # Use shorter timeout for tool loading
        tools_result = await asyncio.wait_for(
            get_tools(mcp_client, state, use_cache=False),
            timeout=10.0  # 10 second timeout
        )
        
        TOOLS_CACHE = tools_result
        TOOLS_CACHE_TIME = current_time
        
        load_time = time.time() - start_time
        logger.info(f"Cached {len(TOOLS_CACHE)} tools in {load_time:.2f}s")
        
        return TOOLS_CACHE
        
    except asyncio.TimeoutError:
        logger.error("Tool cache refresh timed out - using old cache if available")
        return TOOLS_CACHE if TOOLS_CACHE else []
        
    except Exception as e:
        logger.error(f"Failed to refresh tool cache: {e}")
        return TOOLS_CACHE if TOOLS_CACHE else []

def invalidate_tools_cache():
    """Force cache refresh on next request"""
    global TOOLS_CACHE_TIME
    TOOLS_CACHE_TIME = 0
    logger.info("Tool cache invalidated")

async def get_tools(mcp_client: Client, state: AssistantState, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Get tools from MCP server with caching and timeout"""
    current_time = time.time()
    
    # Use cache if available and recent
    if (use_cache and 
        state.tools_cache and 
        current_time - state.tools_cache_time < 30):  # 30 second local cache
        return state.tools_cache
    
    tools = []
    try:
        # Use timeout for tool listing
        tool_list = await asyncio.wait_for(
            mcp_client.list_tools(),
            timeout=5.0  # 5 second timeout
        )
        
        for tool in tool_list:
            try:
                # Try to get OpenAI schema with timeout
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
    
    except asyncio.TimeoutError:
        logger.error("Tool listing timed out")
        # Return cached tools if available
        if state.tools_cache:
            logger.info("Using cached tools due to timeout")
            return state.tools_cache
        raise
        
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
    """Execute a tool call with timeout and optimized error handling"""
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
        raise Exception(error_msg)
    
    logger.info(f"Calling tool: {name} with args: {args}")
    
    async def execute_tool():
        try:
            response = await mcp_client.call_tool(name, args)
            
            # Handle different response types more efficiently
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
            raise Exception(error_msg)
    
    try:
        # Execute with optimized timeout
        start_time = time.time()
        result = await asyncio.wait_for(execute_tool(), timeout=min(timeout, 15))  # Cap at 15s
        
        execution_time = time.time() - start_time
        logger.info(f"Tool {name} completed in {execution_time:.2f}s")
        
        # Check if result indicates failure (optimized check)
        result_lower = result.lower()
        
        # Quick failure detection
        failure_indicators = [
            "not found", "failed to launch", "application not found", 
            "path not found", "errors:", "did you mean:", "use 'list_apps' to see",
            "no apps found", "only .exe files are allowed", "path not in allowed directories",
            "executable not found", "invalid json", "permission denied", "access denied",
            "file not found", "command not found"
        ]
        
        # Optimized failure check - exit early on first match
        for indicator in failure_indicators:
            if indicator in result_lower:
                error_msg = f"Tool {name} failed: {result}"
                logger.error(error_msg)
                raise Exception(error_msg)
        
        # Success case
        return result
        
    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        error_msg = f"Tool {name} timed out after {execution_time:.1f}s"
        logger.error(error_msg)
        raise Exception(error_msg)

async def shutdown_mcp_server(state: AssistantState):
    """Properly shut down MCP server and clear caches"""
    global TOOLS_CACHE, TOOLS_CACHE_TIME, MCP_CONNECTION_POOL, MCP_PROCESS_SHARED
    
    # Clear global cache
    TOOLS_CACHE = None
    TOOLS_CACHE_TIME = 0
    logger.info("Cleared tool cache on shutdown")
    
    # Close all pooled connections
    for client_key, client in MCP_CONNECTION_POOL.items():
        try:
            await client.close()
        except Exception as e:
            logger.error(f"Error closing pooled client {client_key}: {e}")
    
    MCP_CONNECTION_POOL.clear()
    
    # Close MCP client
    if state.mcp_client:
        try:
            await state.mcp_client.close()
        except Exception as e:
            logger.error(f"Error closing MCP client: {e}")
    
    # Terminate shared MCP server subprocess
    if MCP_PROCESS_SHARED:
        try:
            logger.info("Terminating shared MCP server process...")
            MCP_PROCESS_SHARED.terminate()
            
            # Wait for graceful shutdown
            try:
                MCP_PROCESS_SHARED.wait(timeout=3)  # Reduced from 5s
            except subprocess.TimeoutExpired:
                logger.warning("MCP server didn't terminate gracefully, forcing kill...")
                MCP_PROCESS_SHARED.kill()
                MCP_PROCESS_SHARED.wait()
            
            logger.info("MCP server process terminated")
            MCP_PROCESS_SHARED = None
            
        except Exception as e:
            logger.error(f"Error terminating MCP server: {e}")

# Performance monitoring functions
def get_performance_stats():
    """Get performance statistics"""
    global TOOLS_CACHE, TOOLS_CACHE_TIME, MCP_CONNECTION_POOL
    
    current_time = time.time()
    cache_age = current_time - TOOLS_CACHE_TIME if TOOLS_CACHE_TIME > 0 else -1
    
    return {
        "tools_cached": len(TOOLS_CACHE) if TOOLS_CACHE else 0,
        "cache_age_seconds": cache_age,
        "active_connections": len(MCP_CONNECTION_POOL),
        "shared_process_running": MCP_PROCESS_SHARED is not None and MCP_PROCESS_SHARED.poll() is None
    }

# Backward compatibility aliases
get_tools_fast = get_tools_cached
call_tool_fast = call_tool_with_timeout