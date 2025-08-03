# voice_assistant/model_providers/failover_chat.py - FIXED VERSION
"""
Fixed failover chat provider that forces OpenAI for all action requests
"""

import json
import logging
import re
import time
import concurrent.futures
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
    """Fixed chat provider that forces OpenAI for action requests"""
    
    def __init__(self, primary: ChatCompletionProvider, backup: ChatCompletionProvider, timeout: float = 15):
        self.primary = primary
        self.backup = backup
        self.timeout = max(timeout, 10)
        self.last_provider = "unknown"
        self.call_count = 0
        self.consecutive_local_failures = 0
        self.max_consecutive_failures = 2
        self.force_backup_next = False
        
        # Keywords that indicate need for web search or real-time data
        self.web_keywords = [
            'current', 'latest', 'recent', 'today', 'now', 'real-time', 'live',
            'stock price', 'weather', 'news', 'search', 'web', 'internet',
            'reach out to the web', 'look up', 'find out', 'check online',
            'go to the web', 'search the web', '.com', '.org', '.net', '.gov', '.edu',
            'website', 'site', 'page', 'url'
        ]
        
        # Action keywords that require tools - FORCE OPENAI FOR THESE
        self.action_keywords = [
            'open', 'launch', 'start', 'run', 'execute', 'create', 'make', 'delete', 
            'remove', 'save', 'close', 'quit', 'exit', 'go to', 'navigate to',
            'find', 'search for', 'get', 'fetch', 'download', 'upload', 'send',
            'play', 'pause', 'stop', 'record', 'capture', 'install', 'uninstall',
            'update', 'upgrade', 'configure', 'setup', 'enable', 'disable',
            'turn on', 'turn off', 'switch', 'toggle', 'set', 'change', 'modify',
            'edit', 'copy', 'move', 'rename', 'list', 'show', 'display',
            'please open', 'please launch', 'please create', 'please go to',
            'can you open', 'can you launch', 'can you create'
        ]
        
    def _should_force_backup(self, messages: List[Dict[str, Any]]) -> bool:
        """Determine if we should force backup based on message content"""
        if not messages:
            return False
            
        # Check the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "").lower()
                break
        
        if not last_user_msg:
            return False
        
        # Force backup for web-related queries
        for keyword in self.web_keywords:
            if keyword in last_user_msg:
                logger.info(f"Forcing backup due to web keyword: '{keyword}'")
                return True
        
        # FORCE BACKUP FOR ALL ACTION REQUESTS
        for keyword in self.action_keywords:
            if keyword in last_user_msg:
                logger.info(f"🎯 Forcing backup for action keyword: '{keyword}'")
                return True
        
        return False
        
    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Completion with improved failover logic"""
        self.call_count += 1
        logger.info(f"=== Failover complete() call #{self.call_count} ===")
        
        # Check if backup is manually forced
        if self.force_backup_next:
            logger.info("🔄 Manual backup override - using frontier model")
            self.force_backup_next = False
            return self._try_backup(messages, tools, **kwargs)
        
        # Check if we should force backup for web queries OR action requests
        if self._should_force_backup(messages):
            logger.info("🚀 Action/web query detected - using OpenAI for reliable tool calling")
            logger.info(f"🔧 Passing {len(tools) if tools else 0} tools to backup provider")
            return self._try_backup(messages, tools, **kwargs)
        
        # For non-action requests, allow local model
        logger.info("💬 Conversational query - allowing local model")
        
        # Quick heuristic for tool follow-ups
        has_tool_results = any(msg.get("role") == "tool" for msg in messages[-3:])
        if has_tool_results and self.consecutive_local_failures >= 2:
            logger.info("Multiple recent failures + tool results - using backup immediately")
            return self._try_backup(messages, tools, **kwargs)
        
        # Try primary with reduced timeout for tool follow-ups
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
        """Ensure all tool_calls have corresponding tool responses and vice versa - FIXED"""
        validated = []
        i = 0
        
        while i < len(messages):
            msg = messages[i]
            
            # Handle assistant messages with tool calls
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_call_ids = set()
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and "id" in tc:
                        tool_call_ids.add(tc["id"])
                
                # Add the assistant message first
                validated.append(msg)
                
                # Look ahead for tool responses
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    tool_msg = messages[j]
                    tool_call_id = tool_msg.get("tool_call_id")
                    
                    # FIXED: Include ALL tool messages, even if ID doesn't match perfectly
                    # The parallel execution system might generate different IDs
                    validated.append(tool_msg)
                    j += 1
                
                i = j  # Skip processed tool messages
                
            # Handle regular messages  
            elif msg.get("role") in ["system", "user", "assistant"]:
                validated.append(msg)
                i += 1
                
            # FIXED: Don't skip tool messages - include them
            elif msg.get("role") == "tool":
                validated.append(msg)
                i += 1
                
            else:
                i += 1
        
        return validated
    
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