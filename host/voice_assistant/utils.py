# voice_assistant/utils.py
"""
Utility functions for the voice assistant
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

async def retry_with_backoff(coro, max_retries: int = 3, base_delay: float = 1.0):
    """Retry a coroutine with exponential backoff"""
    for attempt in range(max_retries + 1):
        try:
            return await coro
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Final retry attempt failed: {e}")
                raise
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)

def signal_handler(signum, frame, state):
    """Handle Ctrl-C / SIGTERM gracefully."""
    logger.info(f"Received signal {signum}; shutting down…")
    state.running = False
    loop = asyncio.get_event_loop()
    loop.call_soon_threadsafe(loop.stop)