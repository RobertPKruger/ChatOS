# voice_assistant/model_providers/failover_chat.py - FIXED WITH TOOL PARSING
import json
import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TimeoutError
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

# Re-use ONE thread so we can impose a timeout on blocking calls
_EXECUTOR = ThreadPoolExecutor(max_workers=1)

class StandardizedResponse:
    """Wrapper to standardize responses from different providers"""
    def __init__(self, content: str = None, tool_calls=None, raw_response=None):
        self.content = content
        self.tool_calls = tool_calls
        self.raw_response = raw_response
        
        # Mimic OpenAI response structure
        if tool_calls:
            # Create a mock message with tool_calls
            self.choices = [type('Choice', (), {
                'message': type('Message', (), {
                    'content': content or '',
                    'tool_calls': tool_calls,
                    'role': 'assistant'
                })(),
                'finish_reason': 'tool_calls' if tool_calls else 'stop'
            })()]
        else:
            # Simple text response
            self.choices = [type('Choice', (), {
                'message': type('Message', (), {
                    'content': content or '',
                    'role': 'assistant'
                })(),
                'finish_reason': 'stop'
            })()]


def _parse_tool_calls_from_text(content: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from text that might contain JSON arrays or objects
    """
    if not content or not content.strip():
        return []
    
    tool_calls = []
    content = content.strip()
    
    # First, extract content from code blocks if present
    code_block_patterns = [
        r'```(?:python|json|javascript)?\s*\n?(.*?)\n?```',  # ```python\n{...}\n```
        r'`([^`]+)`',  # Single backticks
    ]
    
    extracted_content = content
    for pattern in code_block_patterns:
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            # Replace the original content with the extracted content
            extracted_content = match.strip()
            break  # Use the first match
    
    logger.debug(f"Extracted content from code blocks: '{extracted_content[:200]}...'")
    
    # Look for JSON arrays or objects that might be tool calls
    json_patterns = [
        # Array of objects: [{"name": "...", "arguments": {...}}]
        r'\[\s*\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}\s*(?:,\s*\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}\s*)*\]',
        # Single object: {"name": "...", "arguments": {...}}
        r'\{\s*"name"\s*:\s*"([^"]*?)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
        # Relaxed single object: {"name": "..."}
        r'\{\s*"name"\s*:\s*"([^"]*?)"\s*(?:,\s*"arguments"\s*:\s*(\{[^}]*\}))?\s*\}',
        # Function call pattern: function_name(args)
        r'(\w+)\s*\(\s*([^)]*)\s*\)'
    ]
    
    # Track what we've already found to avoid duplicates
    found_calls = set()
    
    for pattern in json_patterns:
        matches = re.findall(pattern, extracted_content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                if pattern == json_patterns[3]:  # Function call pattern
                    func_name, args_str = match
                    
                    # Skip if we already found this call
                    call_signature = f"{func_name}:{args_str.strip()}"
                    if call_signature in found_calls:
                        continue
                    found_calls.add(call_signature)
                    
                    # Try to parse arguments
                    try:
                        args_dict = json.loads(args_str) if args_str.strip() else {}
                    except:
                        # Parse specific patterns for different tools
                        args_dict = {}
                        if args_str.strip():
                            if func_name == 'launch_app':
                                # Look for app name in various formats
                                app_patterns = [
                                    r"app_name['\"]?\s*[:=]\s*['\"]?([^'\"]+)['\"]?",
                                    r"app['\"]?\s*[:=]\s*['\"]?([^'\"]+)['\"]?",
                                    r"['\"]?([^'\"=,]+)['\"]?"  # Just the app name
                                ]
                                
                                app_name = None
                                for app_pattern in app_patterns:
                                    app_match = re.search(app_pattern, args_str, re.IGNORECASE)
                                    if app_match:
                                        app_name = app_match.group(1).strip()
                                        break
                                
                                if app_name:
                                    args_dict = {"app_name": app_name}
                                else:
                                    args_dict = {"app_name": args_str.strip()}
                            else:
                                args_dict = {"input": args_str.strip()}
                    
                    tool_call = {
                        'id': f'call_{len(tool_calls) + 1}',
                        'type': 'function',
                        'function': {
                            'name': func_name,
                            'arguments': json.dumps(args_dict)
                        }
                    }
                    tool_calls.append(tool_call)
                    
                elif pattern in [json_patterns[1], json_patterns[2]]:  # Named group patterns
                    if len(match) >= 2:
                        name = match[0]
                        arguments_str = match[1] if match[1] else "{}"
                        
                        # Skip if we already found this call
                        call_signature = f"{name}:{arguments_str.strip()}"
                        if call_signature in found_calls:
                            continue
                        found_calls.add(call_signature)
                        
                        logger.debug(f"Extracted name: '{name}', arguments: '{arguments_str}'")
                        
                        try:
                            arguments_dict = json.loads(arguments_str) if arguments_str else {}
                        except:
                            # If we can't parse the arguments, create a default based on the tool
                            if name == 'launch_app':
                                arguments_dict = {"app_name": "notepad"}  # Default fallback
                            else:
                                arguments_dict = {}
                        
                        tool_call = {
                            'id': f'call_{len(tool_calls) + 1}',
                            'type': 'function',
                            'function': {
                                'name': name,
                                'arguments': json.dumps(arguments_dict)
                            }
                        }
                        tool_calls.append(tool_call)
                
                else:
                    # JSON pattern (array or complex object)
                    json_text = match if isinstance(match, str) else match[0]
                    
                    try:
                        parsed = json.loads(json_text)
                    except json.JSONDecodeError:
                        continue
                    
                    # Handle array of tool calls
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and 'name' in item:
                                # Check for duplicates
                                item_signature = f"{item['name']}:{json.dumps(item.get('arguments', {}))}"
                                if item_signature not in found_calls:
                                    found_calls.add(item_signature)
                                    tool_call = _convert_to_openai_tool_call(item, len(tool_calls) + 1)
                                    if tool_call:
                                        tool_calls.append(tool_call)
                    
                    # Handle single tool call
                    elif isinstance(parsed, dict) and 'name' in parsed:
                        # Check for duplicates
                        item_signature = f"{parsed['name']}:{json.dumps(parsed.get('arguments', {}))}"
                        if item_signature not in found_calls:
                            found_calls.add(item_signature)
                            tool_call = _convert_to_openai_tool_call(parsed, len(tool_calls) + 1)
                            if tool_call:
                                tool_calls.append(tool_call)
                            
            except Exception as e:
                logger.debug(f"Failed to parse potential tool call: {match} - {e}")
                continue
    
    return tool_calls


def _convert_to_openai_tool_call(parsed_call: Dict[str, Any], call_id: int) -> Dict[str, Any]:
    """Convert a parsed tool call to OpenAI format"""
    try:
        name = parsed_call.get('name', '')
        arguments = parsed_call.get('arguments', {})
        
        # Handle different argument formats
        if isinstance(arguments, dict):
            # Already a dict - convert to JSON string
            arguments_str = json.dumps(arguments)
        elif isinstance(arguments, str):
            # String - might be JSON or key=value format
            arguments = arguments.strip()
            
            # Try to parse as JSON first
            try:
                json.loads(arguments)  # Test if valid JSON
                arguments_str = arguments
            except json.JSONDecodeError:
                # Not JSON - try to parse key=value format
                logger.debug(f"Parsing non-JSON arguments: '{arguments}'")
                
                # Handle specific patterns for common tools
                if name == 'launch_app':
                    # Look for app name patterns
                    app_patterns = [
                        r"app_name['\"]?\s*[:=]\s*['\"]?([^'\"]+)['\"]?",
                        r"app['\"]?\s*[:=]\s*['\"]?([^'\"]+)['\"]?",
                        r"['\"]?([^'\"]+)['\"]?"  # Just the app name
                    ]
                    
                    app_name = None
                    for pattern in app_patterns:
                        match = re.search(pattern, arguments, re.IGNORECASE)
                        if match:
                            app_name = match.group(1).strip()
                            break
                    
                    if app_name:
                        arguments_str = json.dumps({"app_name": app_name})
                    else:
                        # Fallback - use the whole thing as app_name
                        arguments_str = json.dumps({"app_name": arguments})
                else:
                    # For other tools, wrap in a generic input field
                    arguments_str = json.dumps({"input": arguments})
        else:
            # Other types - convert to string and wrap
            arguments_str = json.dumps({"input": str(arguments)})
        
        logger.debug(f"Converted tool call - name: {name}, arguments: {arguments_str}")
        
        return {
            'id': f'call_{call_id}',
            'type': 'function', 
            'function': {
                'name': name,
                'arguments': arguments_str
            }
        }
    except Exception as e:
        logger.debug(f"Failed to convert tool call: {e}")
        return None


def _serialize_tool_call(tool_call) -> Dict[str, Any]:
    """
    Convert an OpenAI tool call object (Pydantic) to a plain dict.
    Handles both Pydantic objects and already-serialized dicts.
    """
    if isinstance(tool_call, dict):
        return tool_call
    
    # Handle Pydantic model
    if hasattr(tool_call, 'model_dump'):
        return tool_call.model_dump()
    
    # Manual extraction for older OpenAI versions
    result = {}
    
    # Extract ID
    if hasattr(tool_call, 'id'):
        result['id'] = tool_call.id
    
    # Extract type (usually 'function')
    if hasattr(tool_call, 'type'):
        result['type'] = tool_call.type
    
    # Extract function details
    if hasattr(tool_call, 'function'):
        func = tool_call.function
        result['function'] = {
            'name': getattr(func, 'name', ''),
            'arguments': getattr(func, 'arguments', '{}')
        }
    
    return result


def _serialize_message(message: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Convert a message (which might contain Pydantic objects) to a plain dict.
    """
    # If it's already a dict, work with it
    if isinstance(message, dict):
        result = {
            'role': message.get('role', ''),
            'content': message.get('content') or ''  # Handle None content
        }
        
        # Copy other fields that might exist
        if 'name' in message:
            result['name'] = message['name']
        if 'tool_call_id' in message:
            result['tool_call_id'] = message['tool_call_id']
        
        # Handle tool_calls if present
        if 'tool_calls' in message and message['tool_calls']:
            result['tool_calls'] = [
                _serialize_tool_call(tc) for tc in message['tool_calls']
            ]
        
        return result
    
    # If it's a Pydantic model with model_dump
    if hasattr(message, 'model_dump'):
        data = message.model_dump()
        # Ensure content is never None
        if 'content' in data and data['content'] is None:
            data['content'] = ''
        return data
    
    # Manual extraction for other object types
    content = getattr(message, 'content', '')
    if content is None:
        content = ''
    
    result = {
        'role': getattr(message, 'role', ''),
        'content': content
    }
    
    # Copy additional attributes if they exist
    if hasattr(message, 'name'):
        result['name'] = message.name
    if hasattr(message, 'tool_call_id'):
        result['tool_call_id'] = message.tool_call_id
    
    # Check for tool_calls attribute
    if hasattr(message, 'tool_calls') and message.tool_calls:
        result['tool_calls'] = [
            _serialize_tool_call(tc) for tc in message.tool_calls
        ]
    
    return result


def _sanitize_for_ollama(history: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI message objects to plain dicts and remove unsupported roles.
    Keeps system/user/assistant; drops 'tool' role turns.
    """
    safe = []
    allowed_roles = {"system", "user", "assistant"}
    
    for i, msg in enumerate(history):
        # First serialize to plain dict
        serialized = _serialize_message(msg)
        
        # Handle tool responses by converting them to assistant messages
        if serialized['role'] == 'tool':
            # Convert tool response to assistant message
            tool_name = serialized.get('name', 'unknown')
            tool_content = serialized.get('content', '')
            if tool_content is None:
                tool_content = ''
            
            converted = {
                'role': 'assistant',
                'content': f"[Tool {tool_name} result]: {tool_content}"
            }
            
            safe.append(converted)
            continue
        
        # Skip other unsupported roles
        if serialized['role'] not in allowed_roles:
            logger.debug(f"Skipping message with role: {serialized['role']}")
            continue
        
        # Ensure content is never None
        if serialized.get('content') is None:
            serialized['content'] = ''
        
        # Remove tool_calls from messages since Ollama doesn't understand them
        serialized.pop('tool_calls', None)
        
        safe.append(serialized)
    
    return safe


class FailoverChatProvider:
    """
    Tries the primary chat provider (e.g. Mistral via Ollama) first.
    On any exception *or* timeout it logs a warning and transparently
    falls back to the backup provider (e.g. OpenAI).
    """
    def __init__(self, primary, backup, timeout: float = 30.0):
        self.primary = primary
        self.backup = backup
        self.timeout = timeout
        self.last_provider: str | None = None
        self.call_count = 0

    def complete(self, messages, **kwargs):
        """
        Return the assistant's reply in a standardized format.
        Sets `self.last_provider` to "local" or "backup".
        """
        self.call_count += 1
        logger.info(f"=== Failover complete() call #{self.call_count} ===")
        
        # Check if this looks like a tool response scenario
        has_tool_call = any(
            (isinstance(msg, dict) and msg.get('role') == 'assistant' and msg.get('tool_calls')) or
            (hasattr(msg, 'role') and msg.role == 'assistant' and hasattr(msg, 'tool_calls') and msg.tool_calls)
            for msg in messages[-3:]
        )
        has_tool_response = any(
            (isinstance(msg, dict) and msg.get('role') == 'tool') or
            (hasattr(msg, 'role') and msg.role == 'tool')
            for msg in messages[-2:]
        )
        
        if has_tool_call and has_tool_response:
            logger.info("Detected tool response scenario - likely second completion call")
        
        # 1. Try local model with timeout protection
        try:
            # Sanitize messages for Ollama
            if self.primary.__class__.__name__ == "OllamaChatProvider":
                sanitized_messages = _sanitize_for_ollama(messages)
                logger.debug(f"Sanitized {len(sanitized_messages)} messages for Ollama")
            else:
                sanitized_messages = messages
            
            future = _EXECUTOR.submit(self.primary.complete, sanitized_messages, **kwargs)
            reply = future.result(timeout=self.timeout)
            
            # Extract content and tool calls from various response types
            content = None
            tool_calls = None
            
            if isinstance(reply, str):
                content = reply
            elif hasattr(reply, 'content'):
                content = reply.content
                if hasattr(reply, 'tool_calls'):
                    tool_calls = reply.tool_calls
            elif hasattr(reply, 'text'):
                content = reply.text
            elif hasattr(reply, 'message'):
                content = reply.message
            elif hasattr(reply, 'choices') and reply.choices:
                # Already in OpenAI format
                choice = reply.choices[0]
                message = choice.message
                content = getattr(message, 'content', '')
                
                # Check if the content contains tool calls as text
                if content and not getattr(message, 'tool_calls', None):
                    logger.debug(f"Raw content from local model: '{content}'")
                    parsed_tool_calls = _parse_tool_calls_from_text(content)
                    if parsed_tool_calls:
                        logger.info(f"🔧 Parsed {len(parsed_tool_calls)} tool calls from local model text")
                        logger.debug(f"Parsed tool calls: {parsed_tool_calls}")
                        
                        # Create proper tool call objects that are JSON serializable
                        tool_call_objects = []
                        for tc in parsed_tool_calls:
                            # Create simple objects that can be easily serialized
                            tool_call_obj = type('ToolCall', (), {
                                'id': tc['id'],
                                'type': tc['type'],
                                'function': type('Function', (), {
                                    'name': tc['function']['name'],
                                    'arguments': tc['function']['arguments']
                                })()
                            })()
                            tool_call_objects.append(tool_call_obj)
                        
                        # Update the message with tool calls
                        message.tool_calls = tool_call_objects
                        choice.finish_reason = 'tool_calls'
                        
                        # Clear the content since it's now represented as tool calls
                        message.content = ''
                        
                        self.last_provider = "local"
                        logger.info(f"✅ Local model with parsed tool calls")
                        return reply
                
                self.last_provider = "local"
                logger.info(f"✅ Local model returned: '{content[:100] if content else 'Empty'}...'")
                return reply
            else:
                # Try to convert to string
                content = str(reply)
            
            # Check if content contains tool calls as text
            if content:
                parsed_tool_calls = _parse_tool_calls_from_text(content)
                if parsed_tool_calls:
                    logger.info(f"🔧 Parsed {len(parsed_tool_calls)} tool calls from local model")
                    self.last_provider = "local"
                    return StandardizedResponse(content='', tool_calls=parsed_tool_calls)
            
            # Validate the content
            if not content or content.strip() == "":
                logger.warning("Local model returned empty response")
                raise ValueError("Local model returned empty response")

            self.last_provider = "local"
            logger.info(f"✅ Local model response: '{content[:100] if content else 'None'}...'")
            
            # Return standardized response
            return StandardizedResponse(content=content, tool_calls=tool_calls)

        except (_TimeoutError, Exception) as err:
            if isinstance(err, _TimeoutError):
                logger.warning(f"Local LLM timed out after {self.timeout}s → fallback")
            else:
                logger.warning(f"Local LLM failed → fallback. Reason: {err}")
            
            # 2. Fallback to frontier model
            try:
                # For the backup provider, we need to ensure all messages are serializable
                serialized_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        # Already a dict - make sure tool_calls are serializable
                        clean_msg = dict(msg)
                        if 'tool_calls' in clean_msg and clean_msg['tool_calls']:
                            # Convert tool calls to serializable format
                            clean_tool_calls = []
                            for tc in clean_msg['tool_calls']:
                                if isinstance(tc, dict):
                                    clean_tool_calls.append(tc)
                                else:
                                    # Convert object to dict
                                    clean_tc = {
                                        'id': getattr(tc, 'id', ''),
                                        'type': getattr(tc, 'type', 'function'),
                                        'function': {
                                            'name': getattr(tc.function, 'name', '') if hasattr(tc, 'function') else '',
                                            'arguments': getattr(tc.function, 'arguments', '{}') if hasattr(tc, 'function') else '{}'
                                        }
                                    }
                                    clean_tool_calls.append(clean_tc)
                            clean_msg['tool_calls'] = clean_tool_calls
                        serialized_messages.append(clean_msg)
                    else:
                        # Convert to dict format for backup provider
                        msg_dict = _serialize_message(msg)
                        serialized_messages.append(msg_dict)
                
                reply = self.backup.complete(serialized_messages, **kwargs)
                
                # Check if it's already in the expected format
                if hasattr(reply, 'choices') and reply.choices:
                    self.last_provider = "backup"
                    message = reply.choices[0].message
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        logger.info("Used backup model - returning response with tool calls")
                    else:
                        logger.info(f"Used backup model. Response: {message.content[:100] if message.content else 'None'}...")
                    return reply
                else:
                    # Wrap in standardized response
                    content = str(reply)
                    self.last_provider = "backup"
                    logger.info(f"Used backup model. Response: {content[:100]}...")
                    return StandardizedResponse(content=content)
                    
            except Exception as e:
                logger.error(f"Backup provider also failed: {e}")
                raise

    def generate_stream(self, messages, **kwargs):
        """
        Yields token chunks.  Works only if BOTH providers expose
        a synchronous generator named `generate_stream`.
        """
        try:
            # Sanitize messages for Ollama
            if self.primary.__class__.__name__ == "OllamaChatProvider":
                sanitized_messages = _sanitize_for_ollama(messages)
            else:
                sanitized_messages = messages
                
            for chunk in self.primary.generate_stream(sanitized_messages, **kwargs):
                self.last_provider = "local"
                yield chunk
                
        except Exception as err:
            logger.warning(f"Local stream failed → fallback. Reason: {err}")
            for chunk in self.backup.generate_stream(messages, **kwargs):
                self.last_provider = "backup"
                yield chunk