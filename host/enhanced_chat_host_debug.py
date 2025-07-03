# enhanced_chat_host_debug.py
"""
Debug version with enhanced logging to identify local model issues
"""

import asyncio
import logging
import signal
import sys
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

# Enhanced logging for debugging
logging.getLogger("voice_assistant.model_providers").setLevel(logging.DEBUG)
logging.getLogger("voice_assistant.model_providers.ollama_chat").setLevel(logging.DEBUG)
logging.getLogger("voice_assistant.model_providers.failover_chat").setLevel(logging.DEBUG)

state = AssistantState(config.vad_aggressiveness)
audio_recorder = ContinuousAudioRecorder(config.sample_rate)


def test_providers_standalone():
    """Test the providers in isolation before starting the full system"""
    logger.info("🧪 Testing providers in standalone mode...")
    
    try:
        # Test Ollama provider directly
        logger.info("Testing Ollama provider directly...")
        ollama_provider = ModelProviderFactory.create_chat_provider(
            provider_type="ollama",
            model=config.local_chat_model,
            host=config.ollama_host
        )
        
        test_messages = [{"role": "user", "content": "Say 'Local model working!' to confirm."}]
        response = ollama_provider.complete(test_messages)
        logger.info(f"✅ Ollama direct test: {response.choices[0].message.content}")
        
        # Test OpenAI provider directly
        logger.info("Testing OpenAI provider directly...")
        openai_provider = ModelProviderFactory.create_chat_provider(
            provider_type="openai",
            api_key=config.openai_api_key,
            model=config.frontier_chat_model
        )
        
        response = openai_provider.complete(test_messages)
        logger.info(f"✅ OpenAI direct test: {response.choices[0].message.content}")
        
        # Test failover provider
        logger.info("Testing failover provider...")
        failover_provider = FailoverChatProvider(
            primary=ollama_provider,
            backup=openai_provider,
            timeout=config.local_chat_timeout or 30
        )
        
        response = failover_provider.complete(test_messages)
        logger.info(f"✅ Failover test: {response.choices[0].message.content}")
        logger.info(f"✅ Provider used: {failover_provider.last_provider}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Provider test failed: {e}", exc_info=True)
        return False


def initialize_providers(config: Config, state: AssistantState):
    """Initialize model providers based on configuration"""
    try:
        logger.info("🔧 Initializing providers...")
        
        # 1. Build the primary (via Ollama)
        logger.info(f"Creating Ollama provider: model={config.local_chat_model}, host={config.ollama_host}")
        primary_chat = ModelProviderFactory.create_chat_provider(
            provider_type="ollama",
            model=config.local_chat_model,          
            host=config.ollama_host
        )

        # 2. Build the backup (OpenAI)
        logger.info(f"Creating OpenAI provider: model={config.frontier_chat_model}")
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
        logger.error(f"Failed to initialize providers: {e}", exc_info=True)
        raise

async def run_forever():
    """Run the assistant with automatic restart on failure"""
    restart_count = 0

    # Test providers before starting the main loop
    if not test_providers_standalone():
        logger.error("❌ Provider tests failed - check configuration")
        return

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
    
    logger.info("🚀 Voice Assistant starting up (DEBUG MODE)...")
    logger.info(f"Local model: {config.local_chat_model}")
    logger.info(f"Frontier model: {config.frontier_chat_model}")
    logger.info(f"Ollama host: {config.ollama_host}")
    logger.info(f"Timeout: {config.local_chat_timeout}")

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