# voice_assistant/model_providers/openai_chat.py
"""
OpenAI Chat provider implementation
"""

from typing import Any, Dict, List, Optional
from openai import OpenAI
import logging

from .base import ChatCompletionProvider

logger = logging.getLogger(__name__)

class OpenAIChatProvider(ChatCompletionProvider):
    """OpenAI Chat completion provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.last_provider = "openai"
    
    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.7,
        **kwargs
    ) -> Any:
        """Create a chat completion using OpenAI"""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
            
        return self.client.chat.completions.create(**params)