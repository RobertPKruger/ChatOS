#!/usr/bin/env python3
"""
Complete setup script to create all model provider files
Run this from your ChatOS/host directory
"""

import os
import sys

def create_file(path, content):
    """Create a file with the given content"""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# Check if we're in the right directory
if not os.path.exists("voice_assistant"):
    print("Error: Please run this script from the ChatOS/host directory")
    print("Current directory:", os.getcwd())
    sys.exit(1)

print("Setting up model providers...")

# Create the model_providers directory
model_providers_dir = os.path.join("voice_assistant", "model_providers")
os.makedirs(model_providers_dir, exist_ok=True)

# 1. Create __init__.py
init_content = '''"""
Model provider implementations for voice assistant
"""

from .base import (
    TranscriptionProvider,
    ChatCompletionProvider,
    TextToSpeechProvider
)

from .factory import ModelProviderFactory

__all__ = [
    'TranscriptionProvider',
    'ChatCompletionProvider', 
    'TextToSpeechProvider',
    'ModelProviderFactory'
]
'''

create_file(os.path.join(model_providers_dir, "__init__.py"), init_content)

# 2. Create base.py
base_content = '''"""
Base interfaces for model providers
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncGenerator
import io

class TranscriptionProvider(ABC):
    """Base interface for speech-to-text providers"""
    
    @abstractmethod
    async def transcribe(
        self, 
        audio_buffer: io.BytesIO, 
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """Transcribe audio to text"""
        pass

class ChatCompletionProvider(ABC):
    """Base interface for chat completion providers"""
    
    @abstractmethod
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Create a chat completion"""
        pass
    
    @abstractmethod
    async def create_streaming_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        """Create a streaming chat completion"""
        pass

class TextToSpeechProvider(ABC):
    """Base interface for text-to-speech providers"""
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """Convert text to speech audio data"""
        pass
'''

create_file(os.path.join(model_providers_dir, "base.py"), base_content)

# 3. Create openai_provider.py
openai_provider_content = '''"""
OpenAI implementation of model providers
"""

import asyncio
import io
from typing import Optional, Dict, Any, List, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
import logging

from .base import TranscriptionProvider, ChatCompletionProvider, TextToSpeechProvider

logger = logging.getLogger(__name__)

class OpenAITranscriptionProvider(TranscriptionProvider):
    """OpenAI Whisper API implementation"""
    
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    async def transcribe(
        self, 
        audio_buffer: io.BytesIO, 
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """Transcribe audio using OpenAI Whisper"""
        loop = asyncio.get_event_loop()
        
        def _transcribe():
            audio_buffer.seek(0)
            # Handle both whisper-1 and gpt-4o-transcribe models
            if self.model == "gpt-4o-transcribe":
                # Use whisper-1 as the actual API model
                params = {"model": "whisper-1", "file": audio_buffer}
            else:
                params = {"model": self.model, "file": audio_buffer}
            
            if language:
                params["language"] = language
            response = self.client.audio.transcriptions.create(**params)
            return response.text.strip()
        
        return await loop.run_in_executor(None, _transcribe)

class OpenAIChatCompletionProvider(ChatCompletionProvider):
    """OpenAI Chat API implementation"""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.sync_client = OpenAI(api_key=api_key)
    
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Create a chat completion using OpenAI"""
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if tools:
            params["tools"] = tools
            
        if stream:
            return await self.client.chat.completions.create(**params, stream=True)
        else:
            return await self.client.chat.completions.create(**params)
    
    async def create_streaming_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        """Create a streaming chat completion"""
        response = await self.create_completion(
            messages, model, tools, temperature, stream=True, **kwargs
        )
        
        async for chunk in response:
            yield chunk

class OpenAITextToSpeechProvider(TextToSpeechProvider):
    """OpenAI TTS API implementation"""
    
    def __init__(self, api_key: str, model: str = "tts-1", voice: str = "nova"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.default_voice = voice
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """Convert text to speech using OpenAI"""
        loop = asyncio.get_event_loop()
        
        def _synthesize():
            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice or self.default_voice,
                input=text,
                response_format=kwargs.get("response_format", "wav")
            )
            return response.read()
        
        return await loop.run_in_executor(None, _synthesize)
'''

create_file(os.path.join(model_providers_dir, "openai_provider.py"), openai_provider_content)

# 4. Create factory.py
factory_content = '''"""
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
'''

create_file(os.path.join(model_providers_dir, "factory.py"), factory_content)

print("\n✅ Setup complete!")
print("\nCreated files:")
print(f"  - {os.path.join(model_providers_dir, '__init__.py')}")
print(f"  - {os.path.join(model_providers_dir, 'base.py')}")
print(f"  - {os.path.join(model_providers_dir, 'openai_provider.py')}")
print(f"  - {os.path.join(model_providers_dir, 'factory.py')}")

print("\n📝 Next steps:")
print("1. Update your main file to initialize the providers")
print("2. Update speech.py and other modules to use the providers")
print("3. Run your voice assistant!")

# Check if we can import the modules
print("\n🔍 Verifying imports...")
try:
    from voice_assistant.model_providers import ModelProviderFactory
    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")