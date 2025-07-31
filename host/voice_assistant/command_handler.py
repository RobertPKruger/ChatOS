"""Encapsulates ‘special’ voice commands (reset, help, sleep…)."""

import logging
from voice_assistant.speech import speak_text
from voice_assistant.state import AssistantMode

log = logging.getLogger(__name__)


class CommandHandler:
    def __init__(self, config, state, manager):
        self.cfg    = config
        self.state  = state
        self.parent = manager  # to call shutdown_system if needed

    async def handle(self, text: str) -> bool:
        """Returns True iff the input was a handled command."""
        t = text.lower().strip()

        # ---- Reset conversation ----
        if any(p in t for p in ("reset conversation", "clear conversation", "new conversation", "start over")):
            self.state.reset_conversation()
            await speak_text("Conversation reset. Starting fresh!", self.state, self.cfg)
            return True

        # ---- Force frontier model ----
        if any(p in t for p in ("use openai", "use gpt", "force backup")):
            if hasattr(self.state.chat_provider, "force_backup_next"):
                self.state.chat_provider.force_backup_next = True
            await speak_text("I'll use the frontier model for the next response.", self.state, self.cfg)
            return True

        # ---- Launch acknowledgement toggle ----
        if "turn on launch acknowledgment" in t:
            self.cfg.acknowledge_launches = True
            await speak_text("Launch acknowledgments enabled.", self.state, self.cfg)
            return True
        if "turn off launch acknowledgment" in t:
            self.cfg.acknowledge_launches = False
            await speak_text("Launch acknowledgments disabled.", self.state, self.cfg)
            return True

        # ---- Sleep / Wake ----
        if any(p in t for p in ("go to sleep", "sleep mode", "sleep now")):
            await speak_text("Going to sleep. Say 'wake up' to wake me.", self.state, self.cfg)
            self.state.set_mode(AssistantMode.SLEEPING)
            return True

        if self.state.get_mode() == AssistantMode.SLEEPING and any(p in t for p in ("wake up", "wake", "hello", "hey")):
            self.state.set_mode(AssistantMode.LISTENING)
            await speak_text("I'm awake! How can I help you?", self.state, self.cfg)
            return True

        # ---- Exit ----
        if any(p in t for p in ("exit", "quit", "goodbye", "shut down")):
            await speak_text("Goodbye! Shutting down the system…", self.state, self.cfg)
            await self.parent.shutdown_system()
            return True

        # ---- Help ----
        if "help" in t:
            await speak_text(
                "I can open apps, fetch the weather, get stock prices, "
                "and manage files. Say 'reset conversation' to start over.",
                self.state,
                self.cfg,
            )
            return True

        return False
