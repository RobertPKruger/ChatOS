"""
Monitors for ‘stuck’ processing intervals and listens for a wake phrase.
Runs forever until the owning manager cancels the task.
"""

import asyncio
import logging
from voice_assistant.speech import transcribe_audio, speak_text
from voice_assistant.state import AssistantMode

log = logging.getLogger(__name__)


class StuckDetector:
    def __init__(self, cfg, state, recorder):
        self.cfg      = cfg
        self.state    = state
        self.recorder = recorder

    async def run(self):
        while self.state.running:
            try:
                await asyncio.sleep(self.cfg.stuck_check_interval)
                if self.state.is_stuck(self.cfg.processing_timeout):
                    log.warning("Assistant appears stuck.")
                    self.state.set_mode(AssistantMode.STUCK_CHECK)

                    buffer = await self.recorder.record_until_silence(
                        self.state, self.cfg, check_stuck_phrase=True
                    )
                    if not buffer:
                        continue

                    text = await transcribe_audio(buffer, self.state, self.cfg, check_stuck_phrase=True)
                    if not text:
                        continue

                    if self._wake_phrase_detected(text):
                        self.state.interrupt_flag.set()
                        self.state.set_mode(AssistantMode.LISTENING)
                        await speak_text("I'm back! How can I help you?", self.state, self.cfg)
            except Exception as e:
                log.error(f"StuckDetector error: {e}")
                await asyncio.sleep(1)

    # ------------------------------------------------------------------ #
    def _wake_phrase_detected(self, text: str) -> bool:
        words = set(text.lower().replace(",", "").replace("?", "").split())
        return len(words.intersection(self.cfg.stuck_phrase.split())) >= 2
