# voice_assistant/model_providers/failover_chat.py - ENHANCED FOR LOCAL CONFIG AWARENESS
"""
Enhanced failover chat provider that checks local config before forcing OpenAI
"""

import json
import logging
import re
import time
import concurrent.futures
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import ChatCompletionProvider

logger = logging.getLogger(__name__)

class MockToolCall:
    """Mock tool call object for compatibility"""
    def __init__(self, id: str, function_name: str, function_arguments: str):
        self.id = id
        self.type = "function"
        self.function = MockFunction(function_name, function_arguments)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'type': self.type,
            'function': {
                'name': self.function.name,
                'arguments': self.function.arguments
            }
        }

class MockFunction:
    """Mock function object"""
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class MockMessage:
    """Mock message object"""
    def __init__(self, content: str, tool_calls: Optional[List] = None):
        self.content = content
        self.tool_calls = tool_calls or []

class MockChoice:
    """Mock choice object"""
    def __init__(self, message: MockMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason

class MockCompletion:
    """Mock completion object"""
    def __init__(self, choices: List[MockChoice]):
        self.choices = choices

class FailoverChatProvider:
    """Enhanced chat provider that checks local config before forcing OpenAI"""
    
    def __init__(self, primary: ChatCompletionProvider, backup: ChatCompletionProvider, timeout: float = 15):
        self.primary = primary
        self.backup = backup
        self.timeout = max(timeout, 10)
        self.last_provider = "unknown"
        self.call_count = 0
        self.consecutive_local_failures = 0
        self.max_consecutive_failures = 2
        self.force_backup_next = False
        
        # Load apps config to check for configured URLs
        self.apps_config = self._load_apps_config()

        self.use_openai_for_urls = os.getenv("USE_OPENAI_FOR_URLS", "true").lower() == "true"
        self.openai_available = self._check_openai_availability()
        
        # Keywords that indicate need for web search or real-time data
        self.web_keywords = [
            'current', 'latest', 'recent', 'today', 'now', 'real-time', 'live',
            'stock price', 'weather', 'search the web', 'web search', 'look up online',
            'reach out to the web', 'find out', 'check online'
        ]
        
        # Action keywords that require tools - but we'll be smarter about forcing OpenAI
        self.action_keywords = [
            'open', 'launch', 'start', 'run', 'execute', 'create', 'make', 'delete', 
            'remove', 'save', 'close', 'quit', 'exit', 'go to', 'navigate to',
            'find', 'get', 'fetch', 'download', 'upload', 'send',
            'play', 'pause', 'stop', 'record', 'capture', 'install', 'uninstall',
            'update', 'upgrade', 'configure', 'setup', 'enable', 'disable',
            'turn on', 'turn off', 'switch', 'toggle', 'set', 'change', 'modify',
            'edit', 'copy', 'move', 'rename', 'list', 'show', 'display',
            'please open', 'please launch', 'please create', 'please go to',
            'can you open', 'can you launch', 'can you create'
        ]
    
    def _load_apps_config(self) -> Dict[str, Any]:
        """Load apps configuration to check for pre-configured URLs"""
        try:
            # Try multiple locations for apps_config.json
            possible_paths = [
                "apps_config.json",
                "mcp_os/apps_config.json",
                "../mcp_os/apps_config.json",
                "../../mcp_os/apps_config.json"
            ]
            
            for path in possible_paths:
                full_path = Path(path)
                if full_path.exists():
                    with open(full_path, 'r') as f:
                        config = json.load(f)
                        logger.info(f"Loaded apps config from: {full_path}")
                        return config
            
            logger.warning("Could not find apps_config.json - URL checking disabled")
            return {}
            
        except Exception as e:
            logger.error(f"Error loading apps config: {e}")
            return {}
    
    def _has_configured_url_for_query(self, query: str) -> bool:
        """Check if the query matches a configured app/URL in local config"""
        if not self.apps_config:
            return False
        
        query_lower = query.lower()
        
        # Check Windows apps (main platform)
        windows_apps = self.apps_config.get("windows", {})
        
        for app_name, app_config in windows_apps.items():
            # Check if app name matches query
            if app_name in query_lower:
                # Check if it's a URL-type app
                if isinstance(app_config, dict):
                    paths = app_config.get("paths", [])
                elif isinstance(app_config, list):
                    paths = app_config
                else:
                    continue
                
                # If any path starts with "url:", this is a configured URL
                if any(str(path).startswith("url:") for path in paths):
                    logger.info(f"🎯 Found configured URL for '{app_name}' in local config")
                    return True
        
        # Check aliases
        aliases = self.apps_config.get("aliases", {}).get("windows", {})
        for primary_app, alias_list in aliases.items():
            for alias in alias_list:
                if alias.lower() in query_lower:
                    # Check if primary app has configured URL
                    primary_config = windows_apps.get(primary_app, {})
                    if isinstance(primary_config, dict):
                        paths = primary_config.get("paths", [])
                    elif isinstance(primary_config, list):
                        paths = primary_config
                    else:
                        continue
                    
                    if any(str(path).startswith("url:") for path in paths):
                        logger.info(f"🎯 Found configured URL for alias '{alias}' -> '{primary_app}' in local config")
                        return True
        
        return False
    
    def _check_openai_availability(self) -> bool:
        """Check if OpenAI is properly configured"""
        try:
            # Check if we have a valid API key
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            return len(api_key) > 10  # Basic validation
        except:
            return False

    # Updated _should_force_backup method:

    def _should_force_backup(self, messages: List[Dict[str, Any]]) -> bool:
        """Use OpenAI for URLs only if configured and enabled"""
        if not messages:
            return False
            
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "").lower()
                break
        
        if not last_user_msg:
            return False
        
        # Identify website/URL requests
        url_indicators = [
            'take me to', 'go to', 'navigate to', 'visit',
            '.com', '.org', '.net', '.gov', '.edu', '.io', '.ai'
        ]
        
        website_names = [
            'amazon', 'reddit', 'github', 'nugget news', 'nuggetnews',
            'life is a game', 'robert kruger', 'wizards', 'google'
        ]
        
        is_website_request = (
            any(indicator in last_user_msg for indicator in url_indicators) or
            any(site in last_user_msg for site in website_names)
        )
        
        if is_website_request:
            if self.use_openai_for_urls and self.openai_available:
                logger.info(f"🌐 Using OpenAI for URL request: '{last_user_msg}'")
                return True
            else:
                reason = "disabled" if not self.use_openai_for_urls else "not configured"
                logger.info(f"🏠 OpenAI {reason} - using local model for URL: '{last_user_msg}'")
                return False
        
        # Always use OpenAI for real-time data if available
        realtime_keywords = ['current', 'latest', 'stock price', 'weather', 'news']
        if any(keyword in last_user_msg for keyword in realtime_keywords):
            if self.openai_available:
                logger.info(f"🌐 Using OpenAI for real-time data")
                return True
        
        return False


    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Completion with enhanced failover logic"""
        self.call_count += 1
        logger.info(f"=== Enhanced Failover complete() call #{self.call_count} ===")
        
        # Check if backup is manually forced
        if self.force_backup_next:
            logger.info("🔄 Manual backup override - using frontier model")
            self.force_backup_next = False
            return self._try_backup(messages, tools, **kwargs)
        
        # Enhanced logic: check for configured URLs first
        if self._should_force_backup(messages):
            logger.info("🚀 Web query detected - using OpenAI for real-time data")
            logger.info(f"🔧 Passing {len(tools) if tools else 0} tools to backup provider")
            return self._try_backup(messages, tools, **kwargs)
        
        # Allow local model for configured apps and general actions
        logger.info("🏠 Local model handling request (may have config access)")
        
        # Rest of the existing logic...
        has_tool_results = any(msg.get("role") == "tool" for msg in messages[-3:])
        if has_tool_results and self.consecutive_local_failures >= 2:
            logger.info("Multiple recent failures + tool results - using backup immediately")
            return self._try_backup(messages, tools, **kwargs)
        
        # Try primary with appropriate timeout
        if has_tool_results:
            logger.info("Detected tool response scenario - using reduced timeout")
            timeout = min(self.timeout, 8)
        else:
            timeout = self.timeout
            
        try:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._try_primary_with_tools, messages, tools, **kwargs)
                try:
                    result = future.result(timeout=timeout)
                    elapsed = time.time() - start_time
                    
                    # Check if local model provided a useful response
                    if self._is_poor_local_response(result, messages):
                        logger.warning("🔄 Local model gave poor response for this query - trying backup")
                        self.consecutive_local_failures += 1
                        return self._try_backup(messages, tools, **kwargs)
                    
                    logger.info(f"✅ Primary model succeeded in {elapsed:.1f}s")
                    self.last_provider = "local"
                    self.consecutive_local_failures = 0
                    return result
                    
                except concurrent.futures.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.warning(f"🕒 Primary model timed out after {elapsed:.1f}s → fallback")
                    self.consecutive_local_failures += 1
                    future.cancel()
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"💥 Local LLM failed → fallback. Reason: {str(e)[:100]} (after {elapsed:.1f}s)")
            self.consecutive_local_failures += 1
        
        # Fallback to backup
        return self._try_backup(messages, tools, **kwargs)
    
    def _is_poor_local_response(self, result, messages: List[Dict[str, Any]]) -> bool:
        """Check if local model response is inadequate for the query"""
        if not result or not result.choices:
            return True
            
        content = result.choices[0].message.content or ""
        
        # Get the last user message to analyze
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "").lower()
                break
        
        # If user asked for current/web info but got generic response
        for keyword in self.web_keywords:
            if keyword in last_user_msg:
                # Local model shouldn't be able to provide current web data
                if any(phrase in content.lower() for phrase in [
                    "i don't have access", "i can't browse", "i cannot access",
                    "as of my last update", "i don't have real-time"
                ]):
                    return False  # This is actually a good response from local model
                elif any(phrase in content.lower() for phrase in [
                    "the current", "as of today", "latest price is"
                ]):
                    # Local model claiming to have current data - this is wrong
                    logger.warning("Local model claiming to have current data it doesn't have")
                    return True
        
        # CRITICAL: Check if user asked for action but got generic response
        for keyword in self.action_keywords:
            if keyword in last_user_msg:
                if any(phrase in content.lower() for phrase in [
                    "i've completed that task", "task completed", "done", "i've done that"
                ]):
                    logger.warning(f"🚫 Local model gave fake completion for action: '{keyword}'")
                    return True
        
        return False
    
    def _try_primary_with_tools(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Try primary provider with tool call parsing"""
        # Use the primary provider WITHOUT passing tools
        completion = self.primary.complete(messages=messages, **kwargs)
        
        # Extract the response content
        content = completion.choices[0].message.content or ""
        
        # Check if response contains tool calls (JSON format)
        tool_calls = self._extract_tool_calls_from_text(content)
        
        if tool_calls:
            logger.info(f"🔧 Parsed {len(tool_calls)} tool calls from local model text")
            mock_message = MockMessage(content="", tool_calls=tool_calls)
            mock_choice = MockChoice(mock_message, finish_reason="tool_calls")
            logger.info("✅ Local model with parsed tool calls")
            return MockCompletion([mock_choice])
        else:
            logger.info("✅ Local model with text response")
            return completion
    
    def _try_backup(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Try backup provider with proper error handling"""
        try:
            start_time = time.time()
            
            # Clean up conversation history for OpenAI compatibility
            clean_messages = self._clean_messages_for_openai(messages)
            
            # ADDITIONAL VALIDATION: Check for orphaned tool calls
            validated_messages = self._validate_tool_message_pairs(clean_messages)
            
            completion = self.backup.complete(
                messages=validated_messages,
                tools=tools,
                **kwargs
            )
            
            elapsed = time.time() - start_time
            logger.info(f"🔄 Used backup model in {elapsed:.1f}s")
            self.last_provider = "backup"
            self.consecutive_local_failures = 0
            return completion
            
        except Exception as e:
            logger.error(f"Backup provider also failed: {e}")
            # Return a basic error response
            mock_message = MockMessage("I'm having trouble processing your request right now. Please try again.")
            mock_choice = MockChoice(mock_message)
            return MockCompletion([mock_choice])
    

    def _validate_tool_message_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure all tool_calls have corresponding tool responses and vice versa - ENHANCED"""
        validated = []
        
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                
                # Always include system and user messages
                if role in ["system", "user"]:
                    validated.append(msg)
                    continue
                
                # Handle assistant messages
                elif role == "assistant":
                    # If it has tool_calls, validate them
                    if msg.get("tool_calls"):
                        tool_call_ids = set()
                        clean_tool_calls = []
                        
                        # Clean up tool_calls format
                        for tc in msg["tool_calls"]:
                            if isinstance(tc, dict) and "id" in tc:
                                tool_call_ids.add(tc["id"])
                                clean_tool_calls.append(tc)
                            elif hasattr(tc, 'id'):
                                clean_tc = {
                                    'id': tc.id,
                                    'type': getattr(tc, 'type', 'function'),
                                    'function': {
                                        'name': tc.function.name if hasattr(tc, 'function') else '',
                                        'arguments': tc.function.arguments if hasattr(tc, 'function') else '{}'
                                    }
                                }
                                tool_call_ids.add(clean_tc['id'])
                                clean_tool_calls.append(clean_tc)
                        
                        if not clean_tool_calls:
                            # No valid tool calls, treat as regular assistant message
                            clean_msg = {"role": "assistant", "content": msg.get("content", "")}
                            validated.append(clean_msg)
                            continue
                        
                        # Look ahead for matching tool responses
                        found_responses = []
                        j = i + 1
                        
                        while j < len(messages):
                            next_msg = messages[j]
                            if next_msg.get("role") == "tool":
                                tool_call_id = next_msg.get("tool_call_id")
                                if tool_call_id in tool_call_ids:
                                    found_responses.append(next_msg)
                                    tool_call_ids.discard(tool_call_id)  # Remove found ID
                                j += 1
                            else:
                                break
                        
                        # Only include if we found responses for all tool calls
                        if not tool_call_ids:  # All tool calls have responses
                            clean_msg = {
                                "role": "assistant",
                                "content": msg.get("content", ""),
                                "tool_calls": clean_tool_calls
                            }
                            validated.append(clean_msg)
                            validated.extend(found_responses)
                        else:
                            # Some tool calls are orphaned - convert to regular message
                            logger.warning(f"Removing orphaned tool calls: {tool_call_ids}")
                            clean_msg = {"role": "assistant", "content": msg.get("content", "") or "I'll help you with that."}
                            validated.append(clean_msg)
                            
                            # Don't include the orphaned tool responses
                            for response in found_responses:
                                if response.get("tool_call_id") not in tool_call_ids:
                                    logger.warning(f"Skipping orphaned tool response: {response.get('tool_call_id')}")
                    
                    else:
                        # Regular assistant message without tool calls
                        validated.append(msg)
                        
                # CRITICAL: Skip orphaned tool messages completely
                elif role == "tool":
                    logger.warning(f"Skipping orphaned tool message: {msg.get('tool_call_id', 'unknown')}")
                    # Don't add to validated - completely skip orphaned tool messages
                    continue
                    
                else:
                    # Unknown role - convert to assistant message
                    logger.warning(f"Unknown message role '{role}', converting to assistant")
                    clean_msg = {"role": "assistant", "content": str(msg.get("content", ""))}
                    validated.append(clean_msg)
        
        # Final validation - ensure no tool messages without preceding tool_calls
        final_validated = []
        last_had_tool_calls = False
        
        for msg in validated:
            if msg.get("role") == "tool":
                if last_had_tool_calls:
                    final_validated.append(msg)
                else:
                    logger.warning(f"Final cleanup: Removing orphaned tool message")
            else:
                final_validated.append(msg)
                last_had_tool_calls = (msg.get("role") == "assistant" and msg.get("tool_calls"))
        
        logger.info(f"Message validation: {len(messages)} → {len(final_validated)}")
        return final_validated

    def _clean_messages_for_openai(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate messages for OpenAI API compatibility"""
        clean_messages = []
        
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                clean_msg = dict(msg)
                
                # Handle tool_calls serialization
                if 'tool_calls' in clean_msg and clean_msg['tool_calls']:
                    clean_tool_calls = []
                    for tc in clean_msg['tool_calls']:
                        if isinstance(tc, dict):
                            clean_tool_calls.append(tc)
                        elif hasattr(tc, 'to_dict'):
                            clean_tool_calls.append(tc.to_dict())
                        elif hasattr(tc, 'id'):
                            clean_tc = {
                                'id': tc.id,
                                'type': getattr(tc, 'type', 'function'),
                                'function': {
                                    'name': tc.function.name if hasattr(tc, 'function') else '',
                                    'arguments': tc.function.arguments if hasattr(tc, 'function') else '{}'
                                }
                            }
                            clean_tool_calls.append(clean_tc)
                        else:
                            logger.warning(f"Unknown tool call type: {type(tc)}")
                            clean_tc = {
                                'id': str(getattr(tc, 'id', f'unknown_{i}')),
                                'type': 'function',
                                'function': {
                                    'name': str(tc),
                                    'arguments': '{}'
                                }
                            }
                            clean_tool_calls.append(clean_tc)
                    clean_msg['tool_calls'] = clean_tool_calls
                
                clean_messages.append(clean_msg)
            else:
                clean_messages.append({"role": "assistant", "content": str(msg)})
        
        return clean_messages
    
    def _extract_tool_calls_from_text(self, text: str) -> List[MockToolCall]:
        """Extract tool calls from text response with improved parsing"""
        if not text or not text.strip():
            return []
        
        tool_calls = []
        seen_calls = set()
        
        # Pattern 1: Direct JSON object
        json_pattern = r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}'
        
        # Pattern 2: JSON with extra whitespace/formatting
        json_pattern_loose = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'
        
        # Pattern 3: Multiline JSON
        multiline_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}'
        
        patterns = [json_pattern, json_pattern_loose, multiline_pattern]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                try:
                    name = match.group(1)
                    args_str = match.group(2)
                    
                    # Validate JSON arguments
                    args_dict = json.loads(args_str)
                    
                    # Create a signature for deduplication
                    call_signature = f"{name}:{json.dumps(args_dict, sort_keys=True)}"
                    
                    if call_signature in seen_calls:
                        logger.debug(f"Skipping duplicate tool call: {call_signature}")
                        continue
                    
                    seen_calls.add(call_signature)
                    
                    # Create tool call with unique ID
                    tool_id = f"call_{len(tool_calls)}_{int(time.time() * 1000) % 10000}"
                    tool_call = MockToolCall(tool_id, name, args_str)
                    tool_calls.append(tool_call)
                    
                    if len(tool_calls) >= 3:
                        break
                    
                except (json.JSONDecodeError, IndexError):
                    continue
        
        return tool_calls