# voice_assistant/__init__.py
"""
Voice Assistant Package
"""

from .config import Config, setup_logging
from .state import AssistantState, AssistantMode
from .audio import ContinuousAudioRecorder
from .speech import transcribe_audio, speak_text
from .mcp_client import get_mcp_client, get_tools, call_tool_with_timeout
from .conversation import ConversationManager
from .utils import retry_with_backoff, signal_handler

__all__ = [
    'Config',
    'setup_logging',
    'AssistantState',
    'AssistantMode',
    'ContinuousAudioRecorder',
    'transcribe_audio',
    'speak_text',
    'get_mcp_client',
    'get_tools',
    'call_tool_with_timeout',
    'ConversationManager',
    'retry_with_backoff',
    'signal_handler',
]