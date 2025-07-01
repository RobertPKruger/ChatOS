# voice_assistant/model_providers/failover_chat.py
import json
import logging
import time
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
    
    Also handles tool_calls by converting them to text in the content.
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
            
            # Look for the previous assistant message that called this tool
            prev_msg = None
            if i > 0:
                prev_serialized = _serialize_message(history[i-1])
                if prev_serialized.get('role') == 'assistant' and prev_serialized.get('tool_calls'):
                    # Find which tool was called
                    for tc in prev_serialized['tool_calls']:
                        if tc.get('function', {}).get('name') == tool_name:
                            prev_msg = prev_serialized
                            break
            
            # Create a more informative message
            if tool_name == 'launch_app' and 'launched successfully' in tool_content.lower():
                app_name = 'the application'
                # Try to extract app name from previous call
                if prev_msg and prev_msg.get('tool_calls'):
                    for tc in prev_msg['tool_calls']:
                        if tc.get('function', {}).get('name') == 'launch_app':
                            args = tc.get('function', {}).get('arguments', '{}')
                            try:
                                args_dict = json.loads(args)
                                app_name = args_dict.get('app_name', 'the application')
                            except:
                                pass
                
                converted = {
                    'role': 'assistant',
                    'content': f"I've successfully opened {app_name} for you. The application should now be running."
                }
            else:
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
        
        # For Ollama, we need to handle tool_calls differently
        # since it doesn't support OpenAI's tool calling format
        if 'tool_calls' in serialized and serialized['tool_calls']:
            # Convert tool calls to text description
            tool_descriptions = []
            for tc in serialized['tool_calls']:
                func = tc.get('function', {})
                name = func.get('name', 'unknown')
                args = func.get('arguments', '{}')
                
                # Create more natural descriptions
                if name == 'launch_app':
                    try:
                        args_dict = json.loads(args)
                        app_name = args_dict.get('app_name', 'unknown')
                        tool_descriptions.append(f"I'll open {app_name} for you.")
                    except:
                        tool_descriptions.append(f"I'll call the {name} function with args: {args}")
                else:
                    tool_descriptions.append(f"I'll call the {name} function with args: {args}")
            
            # Append tool descriptions to content
            original_content = serialized.get('content', '').strip()
            if original_content:
                serialized['content'] = f"{original_content}\n\n" + "\n".join(tool_descriptions)
            else:
                serialized['content'] = "\n".join(tool_descriptions)
            
            # Remove tool_calls from the message since Ollama doesn't understand them
            serialized.pop('tool_calls', None)
        
        safe.append(serialized)
    
    return safe


class FailoverChatProvider:
    """
    Tries the primary chat provider (e.g. phi3 via Ollama) first.
    On any exception *or* timeout it logs a warning and transparently
    falls back to the backup provider (e.g. OpenAI).

    All calls are **synchronous**; no `async/await` required upstream.
    """
    def __init__(self, primary, backup, timeout: float = 30.0):
        self.primary = primary
        self.backup = backup
        self.timeout = timeout      # seconds
        self.last_provider: str | None = None
        self.call_count = 0  # Track number of calls

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
                self.last_provider = "local"
                logger.info(f"Local model returned OpenAI-style response")
                return reply
            else:
                # Try to convert to string
                content = str(reply)
            
            # Validate the content - empty or generic responses indicate failure
            if not content or content.strip() in ["", "Task completed.", "Task completed"]:
                logger.warning(f"Local model returned empty or generic response: '{content}'")
                raise ValueError("Invalid response from local model")

            self.last_provider = "local"
            logger.info(f"Successfully used local model. Response: {content[:100] if content else 'None'}...")
            
            # Return standardized response
            return StandardizedResponse(content=content, tool_calls=tool_calls)

        except (_TimeoutError, Exception) as err:
            if isinstance(err, _TimeoutError):
                logger.warning(f"Local LLM timed out after {self.timeout}s → fallback")
            else:
                logger.warning(f"Local LLM failed → fallback. Reason: {err}")
            
            # 2. Fallback to frontier model
            try:
                reply = self.backup.complete(messages, **kwargs)
                
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