"""
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
