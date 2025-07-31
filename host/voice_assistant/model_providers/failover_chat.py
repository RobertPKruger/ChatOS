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
    """OPTIMIZED failover provider with smart routing and aggressive timeouts"""
    
    def __init__(self, primary: ChatCompletionProvider, backup: ChatCompletionProvider, timeout: float = 15):
        self.primary = primary
        self.backup = backup
        self.timeout = max(timeout, 10)
        self.last_provider = "unknown"
        self.call_count = 0
        self.consecutive_local_failures = 0
        self.max_consecutive_failures = 2
        self.force_backup_next = False
        
        # OPTIMIZATION 1: Performance tracking for adaptive timeouts
        self.local_response_times = []  # Track last 10 response times
        self.max_history = 10
        
        # OPTIMIZATION 2: Immediate backup triggers (keywords)
        self.web_keywords = [
            'current', 'latest', 'recent', 'today', 'now', 'real-time', 'live',
            'stock price', 'weather', 'news', 'search', 'web', 'internet',
            'reach out to the web', 'look up', 'find out', 'check online',
            'go to the web', 'search the web', '.com', '.org', '.net', '.gov', '.edu',
            'website', 'site', 'page', 'url'
        ]
        
        # OPTIMIZATION 3: Fast local patterns (simple app launches)
        self.fast_local_patterns = [
            r'(?:please\s+)?(?:open|launch|start|run)\s+(?:microsoft\s+)?(excel|word|notepad|chrome|calculator)(?:\s+(?:for\s+me|please))?',
            r'(?:please\s+)?(?:close|quit|exit)\s+(?:microsoft\s+)?(excel|word|notepad|chrome|calculator)',
            r'(?:open|launch)\s+(?:file\s+)?explorer',
            r'(?:create|make)\s+(?:a\s+)?(?:file|folder|directory)',
        ]
        
        # OPTIMIZATION 4: Precompiled regex for speed  
        self.fast_local_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.fast_local_patterns]
        
        # OPTIMIZATION 5: Thread pool for non-blocking operations
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="failover")
        
    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def _get_adaptive_timeout(self, request_type: str = "normal") -> float:
        """Calculate adaptive timeout based on recent performance"""
        if request_type == "simple_app":
            # Very aggressive timeout for simple app launches
            base_timeout = 3.0
        elif request_type == "tool_followup":
            # Reduced timeout for tool follow-ups (they're usually faster)
            base_timeout = 4.0
        else:
            base_timeout = self.timeout
        
        # OPTIMIZATION: Adjust based on recent performance
        if len(self.local_response_times) >= 3:
            avg_response = sum(self.local_response_times) / len(self.local_response_times)
            
            # If local model has been consistently slow, reduce timeout more aggressively
            if avg_response > 4.0:
                adaptive_timeout = min(base_timeout * 0.6, 3.0)  # Very aggressive
                logger.debug(f"⚡ Adaptive timeout reduced to {adaptive_timeout:.1f}s (avg: {avg_response:.1f}s)")
            elif avg_response > 2.0:
                adaptive_timeout = base_timeout * 0.8  # Moderately aggressive
            else:
                adaptive_timeout = base_timeout  # Normal timeout
                
            return max(adaptive_timeout, 2.0)  # Never go below 2 seconds
        
        return base_timeout
    
    def _record_local_response_time(self, duration: float):
        """Track local model response times for adaptive timeout"""
        self.local_response_times.append(duration)
        if len(self.local_response_times) > self.max_history:
            self.local_response_times.pop(0)  # Keep only recent times
    
    def _is_simple_app_launch(self, messages: List[Dict[str, Any]]) -> bool:
        """FAST detection of simple app launches using precompiled regex"""
        if not messages:
            return False
            
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        if not last_user_msg:
            return False
        
        # OPTIMIZATION: Use precompiled regex for speed
        for pattern in self.fast_local_regex:
            if pattern.search(last_user_msg):
                # Ensure it's not a complex request
                complex_indicators = ['current', 'search', 'find me', 'get me', 'tell me', 'what', 'how', 'when', 'which']
                lower_msg = last_user_msg.lower()
                if not any(indicator in lower_msg for indicator in complex_indicators):
                    return True
        
        return False
    
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
        
        return False
    
    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """OPTIMIZED completion with smart routing and adaptive timeouts"""
        self.call_count += 1
        logger.info(f"=== Failover complete() call #{self.call_count} ===")
        
        # OPTIMIZATION 1: Immediate backup for web queries (no local attempt)
        if self._should_force_backup(messages):
            logger.info("🌐 Web-related query detected - using backup provider")
            logger.info(f"🔧 Passing {len(tools) if tools else 0} tools to backup provider")
            return self._try_backup(messages, tools, **kwargs)
        
        # OPTIMIZATION 2: Manual backup override
        if self.force_backup_next:
            logger.info("🔄 Manual backup override - using frontier model")
            self.force_backup_next = False
            return self._try_backup(messages, tools, **kwargs)
        
        # OPTIMIZATION 3: Determine request type for adaptive timeout
        request_type = "normal"
        has_tool_results = any(msg.get("role") == "tool" for msg in messages[-3:])
        
        if self._is_simple_app_launch(messages):
            request_type = "simple_app"
            logger.info("🚀 Simple app launch detected - using aggressive timeout")
        elif has_tool_results:
            request_type = "tool_followup"
            logger.info("Detected tool response scenario - using reduced timeout")
        
        # OPTIMIZATION 4: Skip local if too many recent failures
        if self.consecutive_local_failures >= self.max_consecutive_failures:
            logger.info("🔄 Too many recent local failures - using backup immediately")
            return self._try_backup(messages, tools, **kwargs)
        
        # OPTIMIZATION 5: Try primary with adaptive timeout
        adaptive_timeout = self._get_adaptive_timeout(request_type)
        
        try:
            start_time = time.time()
            
            # Use existing thread pool instead of creating new one each time
            future = self._executor.submit(self._try_primary_with_tools, messages, tools, **kwargs)
            
            try:
                result = future.result(timeout=adaptive_timeout)
                elapsed = time.time() - start_time
                
                # OPTIMIZATION 6: Track performance for future adaptive timeouts
                self._record_local_response_time(elapsed)
                
                # Check if local model provided a useful response
                if self._is_poor_local_response(result, messages):
                    logger.warning("🔄 Local model gave poor response for this query - trying backup")
                    self.consecutive_local_failures += 1
                    return self._try_backup(messages, tools, **kwargs)
                
                logger.info(f"✅ Primary model succeeded in {elapsed:.1f}s")
                self.last_provider = "local"
                self.consecutive_local_failures = 0  # Reset failure count on success
                return result
                
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - start_time
                logger.warning(f"🕒 Primary model timed out after {elapsed:.1f}s → fallback")
                self.consecutive_local_failures += 1
                
                # OPTIMIZATION 7: Don't wait for future to cancel, just move on
                # The future will complete in background but we don't wait
                
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
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
        
        return False
    
    def _try_primary_with_tools(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Try primary provider with tool call parsing - OPTIMIZED"""
        # Use the primary provider WITHOUT passing tools
        completion = self.primary.complete(messages=messages, **kwargs)
        
        # Extract the response content
        content = completion.choices[0].message.content or ""
        
        # OPTIMIZATION: Faster tool call extraction with early exit
        tool_calls = self._extract_tool_calls_from_text_fast(content)
        
        if tool_calls:
            logger.info(f"🔧 Parsed {len(tool_calls)} tool calls from local model text")
            mock_message = MockMessage(content="", tool_calls=tool_calls)
            mock_choice = MockChoice(mock_message, finish_reason="tool_calls")
            return MockCompletion([mock_choice])
        else:
            return completion
    
    def _extract_tool_calls_from_text_fast(self, text: str) -> List[MockToolCall]:
        """OPTIMIZED tool call extraction with early exit and better patterns"""
        if not text or not text.strip():
            return []
        
        tool_calls = []
        
        # OPTIMIZATION: Single comprehensive pattern for speed
        # Look for {"name": "...", "arguments": {...}}
        comprehensive_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'
        
        matches = re.finditer(comprehensive_pattern, text, re.IGNORECASE | re.MULTILINE)
        
        seen_calls = set()
        for match in matches:
            try:
                name = match.group(1)
                args_str = match.group(2)
                
                # Validate JSON arguments
                args_dict = json.loads(args_str)
                
                # Create a signature for deduplication
                call_signature = f"{name}:{json.dumps(args_dict, sort_keys=True)}"
                
                if call_signature in seen_calls:
                    continue
                
                seen_calls.add(call_signature)
                
                # Create tool call with unique ID
                tool_id = f"call_{len(tool_calls)}_{int(time.time() * 1000) % 10000}"
                tool_call = MockToolCall(tool_id, name, args_str)
                tool_calls.append(tool_call)
                
                # OPTIMIZATION: Early exit after finding 3 tool calls
                if len(tool_calls) >= 3:
                    break
                    
            except (json.JSONDecodeError, IndexError):
                continue
        
        return tool_calls
    
    def _try_backup(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Try backup provider with OPTIMIZED message cleaning"""
        try:
            start_time = time.time()
            
            # OPTIMIZATION: Faster message cleaning
            clean_messages = self._clean_messages_for_openai_fast(messages)
            
            completion = self.backup.complete(
                messages=clean_messages,
                tools=tools,
                **kwargs
            )
            
            elapsed = time.time() - start_time
            logger.info(f"🔄 Used backup model in {elapsed:.1f}s")
            self.last_provider = "backup"
            self.consecutive_local_failures = 0  # Reset on successful backup
            return completion
            
        except Exception as e:
            logger.error(f"Backup provider also failed: {e}")
            # Return a basic error response
            mock_message = MockMessage("I'm having trouble processing your request right now. Please try again.")
            mock_choice = MockChoice(mock_message)
            return MockCompletion([mock_choice])
    
    def _clean_messages_for_openai_fast(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OPTIMIZED message cleaning - faster validation"""
        clean_messages = []
        
        # OPTIMIZATION: Process messages in reverse to quickly identify issues
        valid_roles = {"system", "user", "assistant", "tool"}
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get("role")
            if role not in valid_roles:
                continue
            
            clean_msg = {"role": role}
            
            # Handle content
            if "content" in msg:
                clean_msg["content"] = msg["content"]
            
            # Handle tool_calls for assistant messages
            if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                clean_tool_calls = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and "id" in tc:
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
                
                if clean_tool_calls:
                    clean_msg['tool_calls'] = clean_tool_calls
            
            # Handle tool_call_id for tool messages
            if role == "tool" and "tool_call_id" in msg:
                clean_msg["tool_call_id"] = msg["tool_call_id"]
            
            clean_messages.append(clean_msg)
        
        return clean_messages
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_response_time = 0.0
        if self.local_response_times:
            avg_response_time = sum(self.local_response_times) / len(self.local_response_times)
        
        return {
            "total_calls": self.call_count,
            "consecutive_failures": self.consecutive_local_failures,
            "avg_local_response_time": avg_response_time,
            "last_provider": self.last_provider,
            "response_time_samples": len(self.local_response_times)
        }