# enhanced_chat_host_v3.py
"""
Enhanced MCP Chat Host - Version 3 with improved voice responsiveness
Modularized architecture for better maintainability
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from voice_assistant.model_providers.factory import ModelProviderFactory

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from voice_assistant.config import Config, setup_logging
from voice_assistant.state import AssistantState, AssistantMode
from voice_assistant.audio import ContinuousAudioRecorder
from voice_assistant.conversation import ConversationManager
from voice_assistant.utils import signal_handler

# Global configuration and state
config = Config.from_env()
setup_logging(config)
logger = logging.getLogger(__name__)

state = AssistantState(config.vad_aggressiveness)
audio_recorder = ContinuousAudioRecorder(config.sample_rate)


def initialize_providers(config: Config, state: AssistantState):
    """Initialize model providers based on configuration"""
    try:
        # Create transcription provider
        state.transcription_provider = ModelProviderFactory.create_transcription_provider(
            provider_type=config.transcription_provider,
            api_key=config.openai_api_key,
            model=config.stt_model
        )
        logger.info(f"Initialized transcription provider: {config.transcription_provider}")
        
        # Create chat provider
        state.chat_provider = ModelProviderFactory.create_chat_provider(
            provider_type=config.chat_provider,
            api_key=config.openai_api_key
        )
        logger.info(f"Initialized chat provider: {config.chat_provider}")
        
        # Create TTS provider
        state.tts_provider = ModelProviderFactory.create_tts_provider(
            provider_type=config.tts_provider,
            api_key=config.openai_api_key,
            model=config.tts_model,
            voice=config.tts_voice
        )
        logger.info(f"Initialized TTS provider: {config.tts_provider}")
        
        # For backward compatibility, also set openai_client if using OpenAI
        if config.transcription_provider == "openai" or config.chat_provider == "openai" or config.tts_provider == "openai":
            from openai import OpenAI
            state.openai_client = OpenAI(api_key=config.openai_api_key)
            
    except Exception as e:
        logger.error(f"Failed to initialize providers: {e}")
        raise

async def run_forever():
    """Run the assistant with automatic restart on failure"""
    restart_count = 0

    initialize_providers(config, state)
    
    conversation_manager = ConversationManager(config, state, audio_recorder)
    
    while state.running:
        try:
            logger.info(f"Starting voice assistant (restart #{restart_count})")
            await conversation_manager.conversation_loop()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            await conversation_manager.shutdown_system()
            break
            
        except Exception as e:
            restart_count += 1
            logger.exception(f"Fatal error in conversation loop (restart #{restart_count}): {e}")
            
            if restart_count > 10:  # Prevent infinite restart loops
                logger.error("Too many restarts, giving up...")
                break
                
            logger.info(f"Restarting in {config.reconnect_delay} seconds...")
            await asyncio.sleep(config.reconnect_delay)

def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, state))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, state))
    
    logger.info("Voice Assistant starting up...")
    logger.info(f"Configuration: STT={config.stt_model}, Chat={config.chat_model}, TTS={config.tts_model}")
    logger.info(f"Mode: Smart listening with processing timeout ({config.processing_timeout}s)")
    logger.info(f"Wake phrase when stuck: '{config.stuck_phrase}'")
    
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        logger.info("\nShutdown complete")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    
    logger.info("Voice Assistant shutdown complete")

if __name__ == "__main__":
    main()