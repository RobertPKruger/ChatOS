# voice_assistant/model_providers/factory.py - FIXED WITH GRACEFUL FALLBACK
"""
Factory for creating model providers with graceful Ollama fallback
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
        """Create a chat completion provider with graceful fallback"""
        
        if provider_type == "ollama":
            # Create Ollama provider
            host = kwargs.get("host", "http://localhost:11434")
            model = kwargs.get("model", "llama3.1:8b-instruct-q4_0")
            
            try:
                ollama_provider = OllamaChatProvider(host=host, model=model)
                
                # Test connection with a shorter timeout for startup
                logger.info(f"Testing Ollama connection at {host}...")
                if ollama_provider.test_connection():
                    logger.info(f"✅ Connected to Ollama at {host} with model {model}")
                    return ollama_provider
                else:
                    logger.warning(f"❌ Cannot connect to Ollama at {host}")
                    
            except Exception as e:
                logger.warning(f"❌ Ollama connection failed: {e}")
            
            # GRACEFUL FALLBACK: If Ollama fails, automatically fall back to OpenAI
            logger.info("🔄 Automatically falling back to OpenAI provider")
            
            # Check if we have OpenAI credentials for fallback
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("❌ No OpenAI API key available for fallback")
                raise ConnectionError(
                    f"Cannot connect to Ollama server at {host} and no OpenAI API key for fallback. "
                    f"Please either:\n"
                    f"1. Start Ollama server: 'ollama serve'\n"
                    f"2. Set OPENAI_API_KEY environment variable\n"
                    f"3. Set USE_LOCAL_FIRST=false to skip Ollama"
                )
            
            # Create OpenAI fallback
            fallback_model = kwargs.get("fallback_model", "gpt-4o")
            logger.info(f"✅ Using OpenAI {fallback_model} as fallback")
            return OpenAIChatProvider(api_key=api_key, model=fallback_model)
            
        elif provider_type == "openai":
            # Create OpenAI provider
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            model = kwargs.get("model", "gpt-4o")
            logger.info(f"✅ Using OpenAI {model}")
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