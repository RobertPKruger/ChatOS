import asyncio
import logging

logger = logging.getLogger(__name__)

class FailoverChatProvider:
    """
    Tries the primary chat provider first (e.g. Ollama phi3:mini).
    On error it logs a warning and transparently uses the backup provider
    (e.g. OpenAI o3).  All public methods you need—complete(), stream()—should
    be proxied here.
    """
    def __init__(self, primary, backup, timeout=30):
        self.primary = primary
        self.backup = backup
        self.timeout = timeout   # seconds
        self.last_provider: str | None = None

    async def complete(self, messages, **kwargs):
        try:
            return await asyncio.wait_for(
                self.primary.complete(messages, **kwargs),
                timeout=self.timeout
            )
            self.last_provider = "local"           
            return reply
        except Exception as e:
            logger.warning(f"Local LLM failed → fallback: {e}")
            reply = await self.backup.complete(messages, **kwargs)
            self.last_provider = "backup"          # <── NEW
            return reply

    # Optional: if you plan to stream tokens
    async def generate_stream(self, messages, **kwargs):
        try:
            async for chunk in self.primary.generate_stream(messages, **kwargs):
                self.last_provider = "local"
                yield chunk
        except Exception as e:
            logger.warning(f"Stream failed → fallback: {e}")
            async for chunk in self.backup.generate_stream(messages, **kwargs):
                self.last_provider = "backup"
                yield chunk
