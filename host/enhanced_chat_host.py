# enhanced_chat_host.py - WITH CLI SUPPORT
"""
Enhanced MCP Chat Host - Version 3 with CLI mode support
Modularized architecture for better maintainability
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from voice_assistant.model_providers.factory import ModelProviderFactory
from voice_assistant.model_providers.failover_chat import FailoverChatProvider

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

def check_cli_mode():
    """Check if CLI mode is enabled via environment variable"""
    return os.getenv("CHATOS_CLI_MODE", "false").lower() in ["true", "1", "yes", "on"]

def initialize_providers(config: Config, state: AssistantState):
    """Initialize model providers based on configuration"""
    try:
        # 1. Build the primary (via Ollama)
        primary_chat = ModelProviderFactory.create_chat_provider(
            provider_type="ollama",
            model=config.local_chat_model,          
            host=config.ollama_host
        )

        # 2. Build the backup (OpenAI)
        backup_chat = ModelProviderFactory.create_chat_provider(
            provider_type="openai",
            api_key=config.openai_api_key,
            model=config.frontier_chat_model
        )

        # 3. Wrap them
        state.chat_provider = FailoverChatProvider(
            primary=primary_chat,
            backup=backup_chat,
            timeout=config.local_chat_timeout or 30
        )

        logger.info("Chat provider: local-first with frontier fallback")

        logger.info(
            f"Configuration: STT={config.stt_model}, "
            f"Chat={config.local_chat_model} (+fallback {config.frontier_chat_model}), "
            f"TTS={config.tts_model}"
        )

        # Create transcription and TTS providers only for voice mode
        if not check_cli_mode():
            # Create transcription provider
            state.transcription_provider = ModelProviderFactory.create_transcription_provider(
                provider_type=config.transcription_provider,
                api_key=config.openai_api_key,
                model=config.stt_model
            )
            logger.info(f"Initialized transcription provider: {config.transcription_provider}")
            
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

async def run_voice_mode():
    """Run the voice assistant"""
    audio_recorder = ContinuousAudioRecorder(config.sample_rate)
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

async def run_cli_mode():
    """Run the CLI interface"""
    from enhanced_cli_chat_host import enhanced_cli_main
    await enhanced_cli_main()

def main():
    """Main entry point - chooses between CLI and voice mode"""
    
    # Check mode
    cli_mode = check_cli_mode()
    
    if cli_mode:
        logger.info("🖥️  Starting ChatOS in CLI mode")
        print("🖥️  ChatOS CLI Mode")
        print("📝 Set CHATOS_CLI_MODE=false to use voice mode")
    else:
        logger.info("🎤 Starting ChatOS in Voice mode")
        logger.info(f"Mode: Smart listening with processing timeout ({config.processing_timeout}s)")
        logger.info(f"Wake phrase when stuck: '{config.stuck_phrase}'")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, state))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, state))
    
    try:
        if cli_mode:
            asyncio.run(run_cli_mode())
        else:
            asyncio.run(run_voice_mode())
    except KeyboardInterrupt:
        logger.info("\nShutdown complete")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    
    logger.info("ChatOS shutdown complete")

if __name__ == "__main__":
    main()