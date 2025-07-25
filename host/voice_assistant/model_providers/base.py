# voice_assistant/model_providers/base.py - FIXED
"""
Base interfaces for model providers
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
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
    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Any:
        """Create a chat completion"""
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