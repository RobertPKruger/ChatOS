"""
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
