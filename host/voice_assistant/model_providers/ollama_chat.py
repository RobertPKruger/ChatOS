# voice_assistant/model_providers/ollama_chat.py - IMPROVED VERSION
import json
import logging
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class OllamaChatProvider:
    """
    Synchronous wrapper for Ollama's REST API with improved error handling.
    Returns exactly the same object shape as the OpenAI client.
    """

    class _Choice:
        def __init__(self, content: str):
            self.message = type("Msg", (), {
                "content": content,
                "role": "assistant"
            })
            self.finish_reason = "stop"

    class _Completion:
        def __init__(self, content: str):
            self.choices = [OllamaChatProvider._Choice(content)]

    def __init__(self, model: str = "llama3.1:8b-instruct-q4_0",
                 host: str = "http://localhost:11434",
                 stream: bool = False):
        self.model = model
        self.host = host.rstrip('/')  # Remove trailing slash
        self.url = f"{self.host}/api/chat"
        self.stream = stream
        
        # Test connection on initialization
        self._test_connection()

    def _test_connection(self):
        """Test if Ollama is accessible and the model is available"""
        try:
            # Test basic connectivity
            health_url = f"{self.host}/api/tags"
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            
            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            if self.model not in model_names:
                logger.warning(f"Model '{self.model}' not found in Ollama. Available models: {model_names}")
                # Don't fail here - let the actual request fail with better error info
            else:
                logger.info(f"Ollama connection successful. Model '{self.model}' is available.")
                
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.host}: {e}")
            # Don't raise here - let the actual chat request fail and trigger fallback

    def complete(self, messages: List[Dict[str, Any]], **kwargs):
        """
        `messages` should be a list of dicts like
        { "role": "user", "content": "Hello" }
        Returns _Completion object mimicking OpenAI client.
        """
        # Validate input
        if not messages:
            raise ValueError("Messages cannot be empty")
        
        # Ensure messages have required fields
        processed_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(f"Message must be dict, got {type(msg)}")
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Message missing role or content: {msg}")
            
            # Ensure content is not None
            content = msg['content']
            if content is None:
                content = ""
            
            processed_messages.append({
                "role": msg['role'],
                "content": str(content)
            })

        payload = {
            "model": self.model,
            "messages": processed_messages,
            "stream": self.stream,
            **kwargs,
        }
        
        logger.debug(f"Ollama request payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                self.url,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            # Log response status for debugging
            logger.debug(f"Ollama response status: {response.status_code}")
            
            response.raise_for_status()
            
        except requests.Timeout:
            raise RuntimeError(f"Ollama request timed out after 60 seconds")
        except requests.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.host}. Is Ollama running?")
        except requests.HTTPError as e:
            error_detail = ""
            try:
                error_detail = f" - {response.text}"
            except:
                pass
            raise RuntimeError(f"Ollama HTTP error {response.status_code}{error_detail}")
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from Ollama: {e}")

        # Validate response structure
        if "message" not in data:
            raise RuntimeError(f"Unexpected Ollama response format: {data}")
        
        if "content" not in data["message"]:
            raise RuntimeError(f"Missing content in Ollama response: {data}")

        reply_text = data["message"]["content"]
        
        # Validate we got actual content
        if not reply_text or reply_text.strip() == "":
            raise RuntimeError("Ollama returned empty response")
        
        logger.debug(f"Ollama response: {reply_text[:100]}...")
        return OllamaChatProvider._Completion(reply_text)

    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs):
        """Optional: synchronous generator for streaming tokens"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        
        try:
            with requests.post(
                self.url,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                timeout=60,
                stream=True
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming chunk: {line}")
                        continue
                        
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama streaming failed: {e}")