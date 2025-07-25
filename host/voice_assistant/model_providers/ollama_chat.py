# voice_assistant/model_providers/ollama_chat.py
"""
Ollama Chat provider implementation with tool call parsing
"""

import json
import logging
import requests
from typing import Any, Dict, List, Optional
import time

from .base import ChatCompletionProvider

logger = logging.getLogger(__name__)

class OllamaMessage:
    """Mock message object for Ollama responses"""
    def __init__(self, content: str, tool_calls: Optional[List] = None):
        self.content = content
        self.tool_calls = tool_calls or []

class OllamaChoice:
    """Mock choice object for Ollama responses"""
    def __init__(self, message: OllamaMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason

class OllamaCompletion:
    """Mock completion object for Ollama responses"""
    def __init__(self, choices: List[OllamaChoice]):
        self.choices = choices

class OllamaChatProvider(ChatCompletionProvider):
    """Ollama Chat completion provider with tool call parsing"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.1:8b-instruct-q4_0"):
        self.host = host.rstrip('/')
        self.model = model
        self.last_provider = "ollama"
        self.session = requests.Session()
        # Set reasonable timeouts
        self.session.timeout = (5, 30)  # 5s connect, 30s read
    
    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Any:
        """Create a chat completion using Ollama"""
        start_time = time.time()
        
        try:
            # Convert messages to Ollama format
            ollama_messages = self._convert_messages(messages)
            
            # Build request payload
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 4096,  # Context window
                    "num_predict": 1024,  # Max tokens to generate
                }
            }
            
            # Make request to Ollama
            response = self.session.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error {response.status_code}: {response.text}")
            
            result = response.json()
            content = result.get("message", {}).get("content", "").strip()
            
            if not content:
                raise Exception("Ollama returned empty response")
            
            elapsed = time.time() - start_time
            logger.debug(f"Ollama responded in {elapsed:.2f}s: {content[:100]}...")
            
            # Create response object
            message = OllamaMessage(content)
            choice = OllamaChoice(message)
            return OllamaCompletion([choice])
            
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            logger.warning(f"Ollama request timed out after {elapsed:.1f}s")
            raise Exception("Ollama request timed out")
            
        except requests.exceptions.ConnectionError:
            logger.warning("Could not connect to Ollama server")
            raise Exception("Ollama connection failed")
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Ollama error after {elapsed:.1f}s: {e}")
            raise
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Ollama format"""
        ollama_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle system messages
            if role == "system":
                ollama_messages.append({
                    "role": "system",
                    "content": content
                })
            
            # Handle user messages
            elif role == "user":
                ollama_messages.append({
                    "role": "user", 
                    "content": content
                })
            
            # Handle assistant messages
            elif role == "assistant":
                # If assistant message has tool calls, we need to format differently
                if msg.get("tool_calls"):
                    # For Ollama, we'll include tool call info in the content
                    tool_info = []
                    for tc in msg["tool_calls"]:
                        if hasattr(tc, 'function'):
                            tool_info.append(f"Called {tc.function.name} with {tc.function.arguments}")
                        elif isinstance(tc, dict):
                            func_info = tc.get('function', {})
                            tool_info.append(f"Called {func_info.get('name', 'unknown')} with {func_info.get('arguments', '{}')}")
                    
                    combined_content = content
                    if tool_info:
                        combined_content = f"{content}\n[Tools used: {'; '.join(tool_info)}]"
                    
                    ollama_messages.append({
                        "role": "assistant",
                        "content": combined_content
                    })
                else:
                    ollama_messages.append({
                        "role": "assistant",
                        "content": content
                    })
            
            # Handle tool messages - convert to user messages for Ollama
            elif role == "tool":
                tool_content = f"Tool result: {content}"
                ollama_messages.append({
                    "role": "user",
                    "content": tool_content
                })
        
        return ollama_messages
    
    def test_connection(self) -> bool:
        """Test if Ollama server is accessible"""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models from Ollama"""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
        return []