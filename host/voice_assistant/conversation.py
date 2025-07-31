"""
Thin bootstrapper that wires together the high-level pieces
and starts the async conversation loop.
"""

import asyncio
import logging

from voice_assistant.config import Config
from voice_assistant.state import AssistantState
from voice_assistant.audio import ContinuousAudioRecorder
from voice_assistant.conversation_manager import ConversationManager

log = logging.getLogger(__name__)


async def main() -> None:
    cfg   = Config()
    state = AssistantState(cfg)
    audio = ContinuousAudioRecorder(cfg)

    manager = ConversationManager(cfg, state, audio)
    await manager.conversation_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutdown requested by user.")
