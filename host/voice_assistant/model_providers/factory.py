# voice_assistant/model_providers/factory.py - FIXED VERSION
"""
Factory for creating model providers based on configuration
"""

import os
from typing import Optional
import logging
from .openai_chat import OpenAIChatProvider
from .ollama_chat import OllamaChatProvider

from .base import TranscriptionProvider, ChatCompletionProvider, TextToSpeechProvider
from .openai_provider import OpenAITranscriptionProvider, OpenAIChatCompletionProvider, OpenAITextToSpeechProvider

logger = logging.getLogger(__name__)

class ModelProviderFactory:
    """Factory for creating model providers"""
    
    @staticmethod
    def create_transcription_provider(
        provider_type: str,
        fallback_type: Optional[str] = None,
        **kwargs
    ) -> TranscriptionProvider:
        """Create a transcription provider"""
        if provider_type not in ["openai", "hybrid"]:
            logger.warning(f"Provider {provider_type} not implemented, using OpenAI")
        
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        return OpenAITranscriptionProvider(
            api_key=api_key,
            model=kwargs.get("model", "whisper-1")
        )
    
    @staticmethod
    def create_chat_provider(
        provider_type: str,
        fallback_type: Optional[str] = None,
        **kwargs
    ) -> ChatCompletionProvider:
        """Create a chat completion provider"""
        if provider_type == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            return OpenAIChatProvider(
                api_key=api_key,
                model=kwargs.get("model", "gpt-4o"),  # Use passed model, not hardcoded
            )
        elif provider_type == "ollama":
            # Use the actual config values passed in
            model = kwargs.get("model", "llama3.1:8b-instruct-q4_0")
            host = kwargs.get("host", "http://localhost:11434")
            
            logger.info(f"Creating Ollama provider with model: {model}, host: {host}")
            
            return OllamaChatProvider(
                model=model,
                host=host,
            )
        else:
            raise ValueError(f"Unknown chat provider {provider_type}")
    
    @staticmethod
    def create_tts_provider(
        provider_type: str,
        fallback_type: Optional[str] = None,
        **kwargs
    ) -> TextToSpeechProvider:
        """Create a TTS provider"""
        if provider_type not in ["openai", "hybrid"]:
            logger.warning(f"Provider {provider_type} not implemented, using OpenAI")
            
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
            
        return OpenAITextToSpeechProvider(
            api_key=api_key,
            model=kwargs.get("model", "tts-1"),
            voice=kwargs.get("voice", "nova")
        )