"""
Factory for creating model providers based on configuration
"""

import os
from typing import Optional
import logging

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
        # For now, only OpenAI is implemented
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
        # For now, only OpenAI is implemented
        if provider_type not in ["openai", "hybrid"]:
            logger.warning(f"Provider {provider_type} not implemented, using OpenAI")
            
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
            
        return OpenAIChatCompletionProvider(api_key=api_key)
    
    @staticmethod
    def create_tts_provider(
        provider_type: str,
        fallback_type: Optional[str] = None,
        **kwargs
    ) -> TextToSpeechProvider:
        """Create a TTS provider"""
        # For now, only OpenAI is implemented
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
