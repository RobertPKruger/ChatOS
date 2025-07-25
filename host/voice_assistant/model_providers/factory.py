# voice_assistant/model_providers/factory.py - WITH OLLAMA SUPPORT
"""
Factory for creating model providers based on configuration
"""

import os
from typing import Optional
import logging

from .base import TranscriptionProvider, ChatCompletionProvider, TextToSpeechProvider
from .openai_provider import OpenAITranscriptionProvider, OpenAIChatCompletionProvider, OpenAITextToSpeechProvider
from .ollama_chat import OllamaChatProvider
from .openai_chat import OpenAIChatProvider

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
        # For now, only OpenAI is implemented for transcription
        if provider_type not in ["openai", "hybrid"]:
            logger.warning(f"Transcription provider {provider_type} not implemented, using OpenAI")
        
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required for transcription")
        
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
        
        if provider_type == "ollama":
            # Create Ollama provider
            host = kwargs.get("host", "http://localhost:11434")
            model = kwargs.get("model", "llama3.1:8b-instruct-q4_0")
            
            ollama_provider = OllamaChatProvider(host=host, model=model)
            
            # Test connection
            if not ollama_provider.test_connection():
                logger.warning(f"Cannot connect to Ollama at {host}")
                if fallback_type:
                    logger.info(f"Falling back to {fallback_type}")
                    return ModelProviderFactory.create_chat_provider(fallback_type, **kwargs)
                else:
                    raise ConnectionError(f"Cannot connect to Ollama server at {host}")
            
            logger.info(f"Connected to Ollama at {host} with model {model}")
            return ollama_provider
            
        elif provider_type == "openai":
            # Create OpenAI provider
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            model = kwargs.get("model", "gpt-4o")
            return OpenAIChatProvider(api_key=api_key, model=model)
            
        else:
            logger.warning(f"Chat provider {provider_type} not implemented, using OpenAI")
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            model = kwargs.get("model", "gpt-4o")
            return OpenAIChatProvider(api_key=api_key, model=model)
    
    @staticmethod
    def create_tts_provider(
        provider_type: str,
        fallback_type: Optional[str] = None,
        **kwargs
    ) -> TextToSpeechProvider:
        """Create a TTS provider"""
        # For now, only OpenAI is implemented for TTS
        if provider_type not in ["openai", "hybrid"]:
            logger.warning(f"TTS provider {provider_type} not implemented, using OpenAI")
            
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required for TTS")
            
        return OpenAITextToSpeechProvider(
            api_key=api_key,
            model=kwargs.get("model", "tts-1"),
            voice=kwargs.get("voice", "nova")
        )