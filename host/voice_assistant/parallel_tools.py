# host/voice_assistant/parallel_tools.py

import asyncio
import logging
from typing import List, Dict, Any
import time
import os
import pathlib

logger = logging.getLogger(__name__)

async def execute_tools_parallel(mcp_client, tool_calls, timeout: float = 30) -> List[Dict[str, str]]:
    """
    ENHANCED: Execute multiple tool calls with comprehensive result validation
    """
    if not tool_calls:
        logger.warning("No tool calls provided to execute_tools_parallel")
        return []
    
    logger.info(f"🚀 Starting parallel execution of {len(tool_calls)} tools")
    
    # Validate tool calls first
    validated_calls = []
    for i, call in enumerate(tool_calls):
        try:
            tool_info = extract_tool_info(call)
            if tool_info:
                validated_calls.append((call, tool_info))
                logger.debug(f"✅ Validated tool {i}: {tool_info['name']}")
            else:
                logger.warning(f"❌ Invalid tool call {i}: {call}")
        except Exception as e:
            logger.error(f"❌ Error validating tool call {i}: {e}")
    
    if not validated_calls:
        logger.error("No valid tool calls found")
        return []
    
    # Execute validated tools in parallel
    async def execute_single_tool(call_tuple):
        call, tool_info = call_tuple
        tool_name = tool_info['name']
        call_id = tool_info['call_id']
        arguments = tool_info.get('arguments', '{}')
        
        start_time = time.time()
        
        try:
            logger.info(f"⚡ Executing tool in parallel: {tool_name}")
            
            # Import here to avoid circular imports
            from voice_assistant.mcp_client import call_tool_with_timeout
            
            result = await call_tool_with_timeout(mcp_client, call, timeout)
            execution_time = time.time() - start_time
            
            # CRITICAL: Validate the actual result
            validation_result = await validate_tool_result(tool_name, arguments, result)
            
            if validation_result["is_valid"]:
                logger.info(f"✅ Tool {tool_name} completed successfully in {execution_time:.2f}s")
                return {
                    "name": tool_name,
                    "result": validation_result["result"],
                    "call_id": call_id,
                    "success": True,
                    "error": None,
                    "execution_time": execution_time,
                    "validation": "passed"
                }
            else:
                # Tool claimed success but validation failed
                error_msg = validation_result["error"]
                logger.error(f"❌ Tool {tool_name} validation failed: {error_msg}")
                return {
                    "name": tool_name,
                    "result": None,
                    "call_id": call_id,
                    "success": False,
                    "error": f"Validation failed: {error_msg}",
                    "execution_time": execution_time,
                    "validation": "failed"
                }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.warning(f"❌ Tool {tool_name} failed after {execution_time:.2f}s: {error_msg}")
            
            return {
                "name": tool_name,
                "result": None,
                "call_id": call_id,
                "success": False,
                "error": error_msg,
                "execution_time": execution_time,
                "validation": "error"
            }
    
    # Execute all validated tools in parallel
    start_time = time.time()
    
    try:
        results = await asyncio.gather(
            *[execute_single_tool(call_tuple) for call_tuple in validated_calls],
            return_exceptions=True
        )
        
        total_time = time.time() - start_time
        
        # Process results and handle exceptions
        tool_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                call, tool_info = validated_calls[i]
                logger.error(f"❌ Tool execution {i} raised exception: {result}")
                tool_results.append({
                    "name": tool_info['name'],
                    "result": None,
                    "call_id": tool_info['call_id'],
                    "success": False,
                    "error": str(result),
                    "execution_time": total_time,
                    "validation": "exception"
                })
            else:
                tool_results.append(result)
        
        # Log detailed summary
        successful = sum(1 for r in tool_results if r["success"])
        failed = len(tool_results) - successful
        validation_failures = sum(1 for r in tool_results if r.get("validation") == "failed")
        
        logger.info(f"⚡ Parallel execution completed in {total_time:.2f}s")
        logger.info(f"📊 Results: {successful} succeeded, {failed} failed ({validation_failures} validation failures)")
        
        return tool_results
        
    except Exception as e:
        logger.error(f"❌ Critical error in parallel execution: {e}")
        return [
            {
                "name": tool_info['name'],
                "result": None,
                "call_id": tool_info['call_id'],
                "success": False,
                "error": f"Parallel execution failed: {e}",
                "execution_time": time.time() - start_time,
                "validation": "critical_error"
            }
            for call, tool_info in validated_calls
        ]

async def validate_tool_result(tool_name: str, arguments: str, result: Any) -> Dict[str, Any]:
    """
    CRITICAL: Validate that a tool actually accomplished what it claims
    """
    result_str = str(result).strip()
    
    # Basic validation - check for None/empty results
    if result is None or not result_str:
        return {
            "is_valid": False,
            "error": "Tool returned empty/null result",
            "result": None
        }
    
    # Check for explicit error indicators in the result
    error_indicators = [
        "error:",
        "failed:",
        "could not",
        "unable to",
        "permission denied",
        "access denied",
        "file not found",
        "directory not found",
        "invalid path"
    ]
    
    result_lower = result_str.lower()
    for indicator in error_indicators:
        if indicator in result_lower:
            return {
                "is_valid": False,
                "error": f"Tool result contains error: {result_str}",
                "result": None
            }
    
    # Tool-specific validation
    try:
        if tool_name == "create_file":
            return await validate_create_file_result(arguments, result_str)
        elif tool_name == "create_folder":
            return await validate_create_folder_result(arguments, result_str)
        elif tool_name == "launch_app":
            return await validate_launch_app_result(arguments, result_str)
        elif tool_name == "open_url":
            return await validate_open_url_result(arguments, result_str)
        else:
            # Generic validation for other tools
            return await validate_generic_result(tool_name, result_str)
            
    except Exception as e:
        logger.error(f"Error validating {tool_name} result: {e}")
        return {
            "is_valid": False,
            "error": f"Validation error: {e}",
            "result": None
        }

async def validate_create_file_result(arguments: str, result: str) -> Dict[str, Any]:
    """Validate file creation by checking if file actually exists"""
    try:
        import json
        args = json.loads(arguments) if arguments else {}
        file_path = args.get("path", "")
        
        if not file_path:
            return {
                "is_valid": False,
                "error": "No file path specified in arguments",
                "result": None
            }
        
        # Expand the path like the MCP server does
        expanded_path = os.path.expanduser(file_path)
        path_obj = pathlib.Path(expanded_path)
        
        # Check if file actually exists
        if path_obj.exists() and path_obj.is_file():
            # File was actually created
            file_size = path_obj.stat().st_size
            return {
                "is_valid": True,
                "error": None,
                "result": f"File successfully created at {expanded_path} ({file_size} bytes)"
            }
        else:
            # File was NOT created despite success message
            return {
                "is_valid": False,
                "error": f"File was not actually created at {expanded_path}",
                "result": None
            }
            
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Could not validate file creation: {e}",
            "result": None
        }

async def validate_create_folder_result(arguments: str, result: str) -> Dict[str, Any]:
    """Validate folder creation by checking if folder actually exists - FIXED"""
    try:
        import json
        args = json.loads(arguments) if arguments else {}
        folder_path = args.get("path", "")
        folder_name = args.get("name", "")
        
        if not folder_path and not folder_name:
            return {
                "is_valid": False,
                "error": "No folder path specified in arguments",
                "result": None
            }
        
        # FIXED: Properly handle path construction like the MCP server does
        if folder_name and folder_path:
            # Expand user path first (~/Desktop -> C:\Users\username\Desktop)
            expanded_base = os.path.expanduser(folder_path)
            # Join paths using os.path.join for proper path handling
            full_path = os.path.join(expanded_base, folder_name)
        elif folder_path:
            full_path = os.path.expanduser(folder_path)
        else:
            return {
                "is_valid": False,
                "error": "No folder path specified in arguments",
                "result": None
            }
        
        # FIXED: Normalize path to handle mixed slashes and resolve any issues
        full_path = os.path.normpath(full_path)
        path_obj = pathlib.Path(full_path)
        
        # Also check OneDrive desktop if regular desktop doesn't exist
        if not path_obj.exists() and "Desktop" in str(path_obj):
            # Try OneDrive desktop path
            onedrive_path = os.path.expanduser("~/OneDrive/Desktop")
            if folder_name:
                onedrive_full = os.path.join(onedrive_path, folder_name)
            else:
                onedrive_full = onedrive_path
            
            onedrive_path_obj = pathlib.Path(os.path.normpath(onedrive_full))
            if onedrive_path_obj.exists() and onedrive_path_obj.is_dir():
                return {
                    "is_valid": True,
                    "error": None,
                    "result": f"Folder successfully created at {onedrive_full}"
                }
        
        # Check if folder actually exists
        if path_obj.exists() and path_obj.is_dir():
            return {
                "is_valid": True,
                "error": None,
                "result": f"Folder successfully created at {full_path}"
            }
        else:
            # ENHANCED: Provide more debugging info
            debug_info = {
                "expected_path": str(full_path),
                "path_exists": path_obj.exists(),
                "is_directory": path_obj.is_dir() if path_obj.exists() else "N/A",
                "parent_exists": path_obj.parent.exists(),
                "args_received": args
            }
            
            return {
                "is_valid": False,
                "error": f"Folder was not found at expected location. Debug: {debug_info}",
                "result": None
            }
            
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Could not validate folder creation: {e}",
            "result": None
        }

async def validate_launch_app_result(arguments: str, result: str) -> Dict[str, Any]:
    """Validate app launch by checking result message"""
    try:
        # For app launches, we mainly check the result message for success indicators
        result_lower = result.lower()
        
        success_indicators = ["launched", "opened", "started", "running"]
        failure_indicators = ["failed", "not found", "error", "could not"]
        
        if any(indicator in result_lower for indicator in failure_indicators):
            return {
                "is_valid": False,
                "error": f"App launch failed: {result}",
                "result": None
            }
        elif any(indicator in result_lower for indicator in success_indicators):
            return {
                "is_valid": True,
                "error": None,
                "result": result
            }
        else:
            # Ambiguous result - assume success if no clear failure indicators
            return {
                "is_valid": True,
                "error": None,
                "result": result
            }
            
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Could not validate app launch: {e}",
            "result": None
        }

async def validate_open_url_result(arguments: str, result: str) -> Dict[str, Any]:
    """Validate URL opening by checking result message"""
    try:
        result_lower = result.lower()
        
        if "opened" in result_lower or "browser" in result_lower:
            return {
                "is_valid": True,
                "error": None,
                "result": result
            }
        elif "failed" in result_lower or "error" in result_lower:
            return {
                "is_valid": False,
                "error": f"URL opening failed: {result}",
                "result": None
            }
        else:
            # Assume success for URL operations
            return {
                "is_valid": True,
                "error": None,
                "result": result
            }
            
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Could not validate URL opening: {e}",
            "result": None
        }

async def validate_generic_result(tool_name: str, result: str) -> Dict[str, Any]:
    """Generic validation for tools without specific validators"""
    
    # Check length - very short results are often problematic
    if len(result.strip()) < 3:
        return {
            "is_valid": False,
            "error": "Tool result suspiciously short",
            "result": None
        }
    
    # Check for common failure patterns
    failure_patterns = [
        "command not found",
        "no such file",
        "permission denied",
        "access denied",
        "network error",
        "connection failed",
        "timeout",
        "unauthorized"
    ]
    
    result_lower = result.lower()
    for pattern in failure_patterns:
        if pattern in result_lower:
            return {
                "is_valid": False,
                "error": f"Tool result indicates failure: {result}",
                "result": None
            }
    
    # If no obvious failures, assume success
    return {
        "is_valid": True,
        "error": None,
        "result": result
    }

def extract_tool_info(call) -> Dict[str, str]:
    """Extract tool information with better validation"""
    try:
        # Handle OpenAI tool call format
        if hasattr(call, "function") and hasattr(call, "id"):
            return {
                "name": str(call.function.name),
                "call_id": str(call.id),
                "arguments": str(getattr(call.function, "arguments", "{}"))
            }
        
        # Handle MockToolCall with to_dict method
        elif hasattr(call, "to_dict"):
            call_dict = call.to_dict()
            function_info = call_dict.get("function", {})
            return {
                "name": str(function_info.get("name", "unknown")),
                "call_id": str(call_dict.get("id", "unknown_id")),
                "arguments": str(function_info.get("arguments", "{}"))
            }
        
        # Handle dictionary format
        elif isinstance(call, dict):
            function_info = call.get("function", {})
            return {
                "name": str(function_info.get("name", "unknown")),
                "call_id": str(call.get("id", "unknown_id")),
                "arguments": str(function_info.get("arguments", "{}"))
            }
        
        # Handle other object types with fallback
        else:
            name = str(getattr(call, "name", getattr(call, "function_name", "unknown")))
            call_id = str(getattr(call, "id", getattr(call, "call_id", f"fallback_{id(call)}")))
            
            return {
                "name": name,
                "call_id": call_id,
                "arguments": "{}"
            }
            
    except Exception as e:
        logger.error(f"Failed to extract tool info from {type(call)}: {e}")
        return None