# voice_assistant/model_providers/failover_chat.py - FIXED JSON SERIALIZATION
"""
Failover chat provider with JSON serialization fix
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
    """Chat provider with local-first + cloud backup strategy and JSON serialization fixes"""
    
    def __init__(self, primary: ChatCompletionProvider, backup: ChatCompletionProvider, timeout: float = 15):
        self.primary = primary
        self.backup = backup
        self.timeout = max(timeout, 10)  # Minimum 10 second timeout
        self.last_provider = "unknown"
        self.call_count = 0
        self.consecutive_local_failures = 0
        self.max_consecutive_failures = 2  # Switch to backup after 2 failures
        
    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Synchronous completion with failover logic and serialization fixes"""
        self.call_count += 1
        logger.info(f"=== Failover complete() call #{self.call_count} ===")
        
        # Quick heuristic: if this looks like a follow-up after tool execution, 
        # and we have consecutive failures, go straight to backup
        has_tool_results = any(msg.get("role") == "tool" for msg in messages[-3:])
        if has_tool_results and self.consecutive_local_failures >= 2:
            logger.info("Multiple recent failures + tool results - using backup immediately")
            return self._try_backup(messages, tools, **kwargs)
        
        # Try primary (local) first with reduced timeout for tool follow-ups
        if has_tool_results:
            logger.info("Detected tool response scenario - using reduced timeout")
            timeout = min(self.timeout, 8)  # Max 8 seconds for follow-ups
        else:
            timeout = self.timeout
            
        try:
            start_time = time.time()
            
            # Use a simple timeout mechanism with threading
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._try_primary_with_tools, messages, tools, **kwargs)
                try:
                    result = future.result(timeout=timeout)
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Primary model succeeded in {elapsed:.1f}s")
                    self.last_provider = "local"
                    self.consecutive_local_failures = 0
                    return result
                except concurrent.futures.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.warning(f"🕒 Primary model timed out after {elapsed:.1f}s → fallback")
                    self.consecutive_local_failures += 1
                    # Cancel the future
                    future.cancel()
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"💥 Local LLM failed → fallback. Reason: {str(e)[:100]} (after {elapsed:.1f}s)")
            self.consecutive_local_failures += 1
        
        # Fallback to backup
        return self._try_backup(messages, tools, **kwargs)
    
    def _try_primary_with_tools(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Try primary provider with tool call parsing"""
        # Use the primary provider WITHOUT passing tools (local models don't support OpenAI format)
        completion = self.primary.complete(messages=messages, **kwargs)
        
        # Extract the response content
        content = completion.choices[0].message.content or ""
        
        # Check if response contains tool calls (JSON format)
        tool_calls = self._extract_tool_calls_from_text(content)
        
        if tool_calls:
            logger.info(f"🔧 Parsed {len(tool_calls)} tool calls from local model text")
            # Create a mock completion with tool calls
            mock_message = MockMessage(content="", tool_calls=tool_calls)
            mock_choice = MockChoice(mock_message, finish_reason="tool_calls")
            logger.info("✅ Local model with parsed tool calls")
            return MockCompletion([mock_choice])
        else:
            # Regular text response
            logger.info("✅ Local model with text response")
            return completion
    
    def _try_backup(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Try backup provider with proper error handling and serialization fixes"""
        try:
            start_time = time.time()
            
            # Clean up conversation history for OpenAI compatibility
            clean_messages = self._clean_messages_for_openai(messages)
            
            completion = self.backup.complete(
                messages=clean_messages,
                tools=tools,
                **kwargs
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Used backup model in {elapsed:.1f}s. Response: {completion.choices[0].message.content[:100]}...")
            self.last_provider = "backup"
            self.consecutive_local_failures = 0  # Reset on successful backup
            return completion
            
        except Exception as e:
            logger.error(f"Backup provider also failed: {e}")
            # Return a basic error response
            mock_message = MockMessage("I'm having trouble processing your request right now. Please try again.")
            mock_choice = MockChoice(mock_message)
            return MockCompletion([mock_choice])
    
    def _clean_messages_for_openai(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate messages for OpenAI API compatibility with serialization fixes"""
        clean_messages = []
        
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                clean_msg = dict(msg)
                
                # Handle tool_calls serialization - FIXED VERSION
                if 'tool_calls' in clean_msg and clean_msg['tool_calls']:
                    clean_tool_calls = []
                    for tc in clean_msg['tool_calls']:
                        if isinstance(tc, dict):
                            # Already a dictionary
                            clean_tool_calls.append(tc)
                        elif hasattr(tc, 'to_dict'):
                            # Has serialization method
                            clean_tool_calls.append(tc.to_dict())
                        elif hasattr(tc, 'id'):  # MockToolCall or similar object
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
                            # Try to convert to string representation
                            logger.warning(f"Unknown tool call type: {type(tc)}, converting to string")
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
                # Convert non-dict messages
                clean_messages.append({"role": "assistant", "content": str(msg)})
        
        # Validate tool call/response pairs with better error handling
        validated_messages = []
        skip_orphaned_tools = False
        
        for i, msg in enumerate(clean_messages):
            if skip_orphaned_tools and msg.get('role') == 'tool':
                logger.debug(f"Skipping orphaned tool message: {msg.get('tool_call_id', 'unknown')}")
                continue
            
            skip_orphaned_tools = False
            
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                # Check if subsequent tool responses exist
                tool_call_ids = set()
                for tc in msg['tool_calls']:
                    if isinstance(tc, dict) and 'id' in tc:
                        tool_call_ids.add(tc['id'])
                
                found_responses = set()
                
                # Look ahead for tool responses
                j = i + 1
                while j < len(clean_messages) and clean_messages[j].get('role') == 'tool':
                    tool_msg = clean_messages[j]
                    tool_call_id = tool_msg.get('tool_call_id')
                    if tool_call_id in tool_call_ids:
                        found_responses.add(tool_call_id)
                    j += 1
                
                # If we're missing tool responses, remove orphaned tool_calls
                if len(found_responses) < len(tool_call_ids):
                    missing_ids = tool_call_ids - found_responses
                    logger.warning(f"Removing orphaned tool_calls: {missing_ids}")
                    
                    # Filter out orphaned tool calls
                    valid_tool_calls = [
                        tc for tc in msg['tool_calls'] 
                        if isinstance(tc, dict) and tc.get('id') in found_responses
                    ]
                    
                    clean_msg = dict(msg)
                    if valid_tool_calls:
                        clean_msg['tool_calls'] = valid_tool_calls
                    else:
                        # Remove tool_calls entirely if none are valid
                        clean_msg.pop('tool_calls', None)
                        if not clean_msg.get('content'):
                            clean_msg['content'] = "I'll help you with that."
                    
                    validated_messages.append(clean_msg)
                    
                    # Skip orphaned tool responses
                    if len(found_responses) != len(tool_call_ids):
                        skip_orphaned_tools = True
                else:
                    validated_messages.append(msg)
            else:
                validated_messages.append(msg)
        
        return validated_messages
    
    def _extract_tool_calls_from_text(self, text: str) -> List[MockToolCall]:
        """Extract tool calls from text response with improved parsing and deduplication"""
        if not text or not text.strip():
            return []
        
        tool_calls = []
        seen_calls = set()  # Track duplicates
        
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
                    
                    # Skip if we've already seen this exact call
                    if call_signature in seen_calls:
                        logger.debug(f"Skipping duplicate tool call: {call_signature}")
                        continue
                    
                    seen_calls.add(call_signature)
                    
                    # Create tool call with unique ID
                    tool_id = f"call_{len(tool_calls)}_{int(time.time() * 1000) % 10000}"
                    tool_call = MockToolCall(tool_id, name, args_str)
                    tool_calls.append(tool_call)
                    
                    # Limit to maximum 3 unique tool calls to prevent abuse
                    if len(tool_calls) >= 3:
                        break
                    
                except (json.JSONDecodeError, IndexError):
                    continue
        
        # Additional safety: if we have multiple identical calls, keep only the first
        if len(tool_calls) > 1:
            unique_calls = []
            seen_names_args = set()
            
            for tc in tool_calls:
                name_args = f"{tc.function.name}:{tc.function.arguments}"
                if name_args not in seen_names_args:
                    unique_calls.append(tc)
                    seen_names_args.add(name_args)
            
            if len(unique_calls) != len(tool_calls):
                logger.info(f"Deduplicated tool calls: {len(tool_calls)} → {len(unique_calls)}")
                tool_calls = unique_calls
        
        return tool_calls