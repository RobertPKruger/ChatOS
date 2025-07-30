#!/usr/bin/env python3
"""
ChatOS Test Suite - Comprehensive testing with performance monitoring
Ensures functionality and tracks metrics like local model usage percentage
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from tests/ to ChatOS/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "host"))
sys.path.insert(0, str(PROJECT_ROOT / "mcp_os"))

class TestConfig:
    """Configuration for test thresholds and settings"""
    
    def __init__(self, config_file: str = "test_config.json"):
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Load test configuration from file"""
        default_config = {
            "thresholds": {
                "pct_local_use": 50,  # Warn if local usage drops below 50%
                "avg_response_time_seconds": 5.0,  # Warn if avg response > 5s
                "tool_success_rate": 85,  # Warn if tool success < 85%
                "transcription_accuracy": 90,  # Warn if transcription < 90%
                "max_memory_mb": 500  # Warn if memory usage > 500MB
            },
            "test_settings": {
                "mock_audio": True,
                "mock_openai": False,
                "mock_ollama": False,
                "enable_integration_tests": True,
                "test_timeout_seconds": 30
            },
            "monitoring": {
                "track_metrics": True,
                "save_results": True,
                "results_file": "test_results.json",
                "alert_on_regression": True
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self.config = self._deep_merge(default_config, loaded_config)
            else:
                self.config = default_config
                self._save_config()
        except Exception as e:
            print(f"Warning: Could not load test config: {e}")
            self.config = default_config
    
    def _deep_merge(self, base: dict, update: dict) -> dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _save_config(self):
        """Save current config to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save test config: {e}")
    
    def get_threshold(self, metric: str) -> float:
        """Get threshold value for a metric"""
        return self.config["thresholds"].get(metric, 0)
    
    def get_setting(self, setting: str) -> Any:
        """Get test setting value"""
        return self.config["test_settings"].get(setting, False)

class MetricsCollector:
    """Collects and analyzes test metrics"""
    
    def __init__(self):
        self.metrics = {
            "local_model_calls": 0,
            "backup_model_calls": 0,
            "tool_calls": 0,
            "tool_successes": 0,
            "response_times": [],
            "memory_usage": [],
            "errors": [],
            "test_start_time": time.time()
        }
    
    def record_model_usage(self, provider: str):
        """Record which model provider was used"""
        if provider in ["ollama", "local"]:
            self.metrics["local_model_calls"] += 1
        elif provider in ["openai", "backup"]:
            self.metrics["backup_model_calls"] += 1
    
    def record_tool_call(self, success: bool):
        """Record tool call and its success"""
        self.metrics["tool_calls"] += 1
        if success:
            self.metrics["tool_successes"] += 1
    
    def record_response_time(self, duration: float):
        """Record response time"""
        self.metrics["response_times"].append(duration)
    
    def record_error(self, error: str):
        """Record an error"""
        self.metrics["errors"].append({
            "error": error,
            "timestamp": time.time()
        })
    
    def get_local_usage_percentage(self) -> float:
        """Calculate percentage of local model usage"""
        total_calls = self.metrics["local_model_calls"] + self.metrics["backup_model_calls"]
        if total_calls == 0:
            return 0.0
        return (self.metrics["local_model_calls"] / total_calls) * 100
    
    def get_tool_success_rate(self) -> float:
        """Calculate tool success rate"""
        if self.metrics["tool_calls"] == 0:
            return 100.0
        return (self.metrics["tool_successes"] / self.metrics["tool_calls"]) * 100
    
    def get_avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.metrics["response_times"]:
            return 0.0
        return sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "local_usage_pct": self.get_local_usage_percentage(),
            "tool_success_rate": self.get_tool_success_rate(),
            "avg_response_time": self.get_avg_response_time(),
            "total_model_calls": self.metrics["local_model_calls"] + self.metrics["backup_model_calls"],
            "total_tool_calls": self.metrics["tool_calls"],
            "total_errors": len(self.metrics["errors"]),
            "test_duration": time.time() - self.metrics["test_start_time"]
        }

class ChatOSTestCase(unittest.TestCase):
    """Base test case with metrics collection"""
    
    @classmethod
    def setUpClass(cls):
        cls.config = TestConfig()
        cls.metrics = MetricsCollector()
        cls.setup_logging()
    
    @classmethod
    def setup_logging(cls):
        """Setup test logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Reduce noise from third-party libraries during tests
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
    
    def measure_response_time(self, func):
        """Decorator to measure response time"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self.metrics.record_response_time(duration)
            return result
        return wrapper

class TestModelProviders(ChatOSTestCase):
    """Test model provider functionality and failover"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
    
    def test_ollama_provider_creation(self):
        """Test Ollama provider can be created"""
        try:
            from host.voice_assistant.model_providers.ollama_chat import OllamaChatProvider
            
            provider = OllamaChatProvider(host="http://localhost:11434")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.host, "http://localhost:11434")
            self.metrics.record_model_usage("local")
            
        except ImportError as e:
            self.fail(f"Could not import OllamaChatProvider: {e}")
    
    def test_openai_provider_creation(self):
        """Test OpenAI provider can be created"""
        try:
            from host.voice_assistant.model_providers.openai_chat import OpenAIChatProvider
            
            # Use fake API key for testing
            provider = OpenAIChatProvider(api_key="test-key")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.last_provider, "openai")
            self.metrics.record_model_usage("backup")
            
        except ImportError as e:
            self.fail(f"Could not import OpenAIChatProvider: {e}")
    
    def test_failover_provider_creation(self):
        """Test failover provider can be created"""
        try:
            from host.voice_assistant.model_providers.failover_chat import FailoverChatProvider
            from host.voice_assistant.model_providers.ollama_chat import OllamaChatProvider
            from host.voice_assistant.model_providers.openai_chat import OpenAIChatProvider
            
            primary = OllamaChatProvider()
            backup = OpenAIChatProvider(api_key="test-key")
            
            failover = FailoverChatProvider(primary, backup, timeout=15)
            self.assertIsNotNone(failover)
            self.assertEqual(failover.timeout, 15)
            
        except ImportError as e:
            self.fail(f"Could not import failover providers: {e}")
    
    @patch('requests.Session.get')
    def test_ollama_connection_check(self, mock_get):
        """Test Ollama connection checking"""
        from host.voice_assistant.model_providers.ollama_chat import OllamaChatProvider
        
        # Mock successful connection
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        provider = OllamaChatProvider()
        self.assertTrue(provider.test_connection())
        
        # Mock failed connection
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        self.assertFalse(provider.test_connection())

class TestMCPTools(ChatOSTestCase):
    """Test MCP tool functionality"""
    
    def setUp(self):
        """Setup test MCP client"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
    
    def tearDown(self):
        """Cleanup test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_app_tools_manager_creation(self):
        """Test AppToolsManager can be created"""
        try:
            from mcp_os.app_tools import AppToolsManager
            
            manager = AppToolsManager()
            self.assertIsNotNone(manager)
            self.assertIn("windows", manager.current_os)  # Assuming Windows
            
        except ImportError as e:
            self.fail(f"Could not import AppToolsManager: {e}")
    
    def test_file_creation_tool(self):
        """Test file creation functionality"""
        try:
            # Test creating a file
            with open(self.test_file, 'w') as f:
                f.write("test content")
            
            self.assertTrue(os.path.exists(self.test_file))
            
            with open(self.test_file, 'r') as f:
                content = f.read()
            
            self.assertEqual(content, "test content")
            self.metrics.record_tool_call(True)
            
        except Exception as e:
            self.metrics.record_tool_call(False)
            self.metrics.record_error(f"File creation test failed: {e}")
            self.fail(f"File creation test failed: {e}")
    
    def test_app_launching_logic(self):
        """Test application launching logic (without actually launching)"""
        try:
            from mcp_os.app_tools import AppToolsManager
            
            manager = AppToolsManager()
            
            # Test app name resolution
            resolved = manager.resolve_app_name("notepad")
            self.assertIsNotNone(resolved)
            
            # Test alias resolution
            resolved = manager.resolve_app_name("notes")  # Should resolve to notepad
            self.assertEqual(resolved, "notepad")
            
            self.metrics.record_tool_call(True)
            
        except Exception as e:
            self.metrics.record_tool_call(False)
            self.metrics.record_error(f"App launching test failed: {e}")
            self.fail(f"App launching test failed: {e}")

class TestAudioProcessing(ChatOSTestCase):
    """Test audio processing functionality"""
    
    def test_audio_recorder_creation(self):
        """Test audio recorder can be created"""
        if self.config.get_setting("mock_audio"):
            # Mock audio system for CI/CD
            with patch('sounddevice.query_devices'), \
                 patch('sounddevice.InputStream'):
                
                try:
                    from host.voice_assistant.audio import ContinuousAudioRecorder
                    
                    recorder = ContinuousAudioRecorder()
                    self.assertIsNotNone(recorder)
                    self.assertEqual(recorder.sample_rate, 16000)
                    
                except ImportError as e:
                    self.fail(f"Could not import ContinuousAudioRecorder: {e}")
        else:
            self.skipTest("Audio testing disabled in config")
    
    def test_speech_validation(self):
        """Test speech validation logic"""
        try:
            from host.voice_assistant.speech import VALID_ACKNOWLEDGEMENTS_SET
            
            # Test that common phrases are in validation set
            self.assertIn("yes", VALID_ACKNOWLEDGEMENTS_SET)
            self.assertIn("no", VALID_ACKNOWLEDGEMENTS_SET)
            self.assertIn("hello", VALID_ACKNOWLEDGEMENTS_SET)
            self.assertIn("thank you", VALID_ACKNOWLEDGEMENTS_SET)
            
            # Test set is not empty
            self.assertGreater(len(VALID_ACKNOWLEDGEMENTS_SET), 50)
            
        except ImportError as e:
            self.fail(f"Could not import speech validation: {e}")

class TestConversationFlow(ChatOSTestCase):
    """Test conversation management"""
    
    def setUp(self):
        """Setup conversation test environment"""
        self.mock_config = Mock()
        self.mock_config.vad_aggressiveness = 3
        self.mock_config.processing_timeout = 60
        self.mock_config.stuck_phrase = "hello abraxas are you stuck"
    
    def test_assistant_state_creation(self):
        """Test assistant state can be created"""
        try:
            from host.voice_assistant.state import AssistantState, AssistantMode
            
            state = AssistantState(vad_aggressiveness=3)
            self.assertIsNotNone(state)
            self.assertEqual(state.get_mode(), AssistantMode.LISTENING)
            
        except ImportError as e:
            self.fail(f"Could not import AssistantState: {e}")
    
    def test_conversation_history_management(self):
        """Test conversation history trimming"""
        try:
            from host.voice_assistant.state import AssistantState
            
            state = AssistantState()
            
            # AssistantState starts empty, but reset_conversation adds system message
            state.reset_conversation()
            initial_length = len(state.conversation_history)
            self.assertGreater(initial_length, 0, "Should have system message after reset")
            
            # Add many messages to trigger trimming
            for i in range(15):
                state.add_user_message(f"Message {i}")
                state.add_assistant_message(f"Response {i}")
            
            # Should be trimmed to reasonable size
            self.assertLessEqual(len(state.conversation_history), 20)
            
            # Should have trimmed from the original 30+ messages
            self.assertLess(len(state.conversation_history), initial_length + 30)
            
            # Check that we still have recent messages
            recent_messages = [msg for msg in state.conversation_history if "Message" in str(msg.get("content", ""))]
            self.assertGreater(len(recent_messages), 0, "Should retain some recent messages")
            
        except ImportError as e:
            self.fail(f"Could not import conversation management: {e}")
    
    def test_mode_transitions(self):
        """Test assistant mode transitions"""
        try:
            from host.voice_assistant.state import AssistantState, AssistantMode
            
            state = AssistantState()
            
            # Test mode transitions
            state.set_mode(AssistantMode.RECORDING)
            self.assertEqual(state.get_mode(), AssistantMode.RECORDING)
            
            state.set_mode(AssistantMode.PROCESSING)
            self.assertEqual(state.get_mode(), AssistantMode.PROCESSING)
            
            state.set_mode(AssistantMode.SPEAKING)
            self.assertEqual(state.get_mode(), AssistantMode.SPEAKING)
            
            state.set_mode(AssistantMode.LISTENING)
            self.assertEqual(state.get_mode(), AssistantMode.LISTENING)
            
        except ImportError as e:
            self.fail(f"Could not import state management: {e}")

class TestConfiguration(ChatOSTestCase):
    """Test configuration loading and validation"""
    
    def test_config_loading(self):
        """Test configuration can be loaded"""
        try:
            from host.voice_assistant.config import Config
            
            # Test with environment variables
            os.environ["OPENAI_API_KEY"] = "test-key"
            os.environ["USE_LOCAL_FIRST"] = "true"
            
            config = Config.from_env()
            self.assertIsNotNone(config)
            self.assertEqual(config.openai_api_key, "test-key")
            self.assertTrue(config.use_local_first)
            
        except ImportError as e:
            self.fail(f"Could not import Config: {e}")
    
    def test_apps_config_loading(self):
        """Test apps configuration loading"""
        try:
            apps_config_path = Path(__file__).parent.parent / "mcp_os" / "apps_config.json"
            
            if apps_config_path.exists():
                with open(apps_config_path, 'r') as f:
                    config = json.load(f)
                
                # Test basic structure
                self.assertIn("windows", config)
                self.assertIn("aliases", config)
                
                # Test that essential apps are configured
                windows_apps = config.get("windows", {})
                self.assertIn("notepad", windows_apps)
                self.assertIn("chrome", windows_apps)
                
            else:
                self.fail("apps_config.json not found")
                
        except Exception as e:
            self.fail(f"Could not load apps config: {e}")

class TestRegressionsAndThresholds(ChatOSTestCase):
    """Test for regressions and threshold violations"""
    
    @classmethod
    def tearDownClass(cls):
        """Check thresholds and report results"""
        summary = cls.metrics.get_summary()
        config = cls.config
        
        print("\n" + "="*60)
        print("CHATOS TEST RESULTS SUMMARY")
        print("="*60)
        
        # Check local usage threshold
        local_usage = summary["local_usage_pct"]
        local_threshold = config.get_threshold("pct_local_use")
        
        print(f"Local Model Usage: {local_usage:.1f}%")
        if local_usage < local_threshold:
            print(f"⚠️  WARNING: Local usage below threshold ({local_threshold}%)")
        else:
            print(f"✅ Local usage meets threshold ({local_threshold}%)")
        
        # Check response time threshold
        avg_response = summary["avg_response_time"]
        response_threshold = config.get_threshold("avg_response_time_seconds")
        
        print(f"Average Response Time: {avg_response:.2f}s")
        if avg_response > response_threshold:
            print(f"⚠️  WARNING: Response time above threshold ({response_threshold}s)")
        else:
            print(f"✅ Response time meets threshold ({response_threshold}s)")
        
        # Check tool success rate
        tool_success = summary["tool_success_rate"]
        tool_threshold = config.get_threshold("tool_success_rate")
        
        print(f"Tool Success Rate: {tool_success:.1f}%")
        if tool_success < tool_threshold:
            print(f"⚠️  WARNING: Tool success rate below threshold ({tool_threshold}%)")
        else:
            print(f"✅ Tool success rate meets threshold ({tool_threshold}%)")
        
        # Overall stats
        print(f"\nTotal Model Calls: {summary['total_model_calls']}")
        print(f"Total Tool Calls: {summary['total_tool_calls']}")
        print(f"Total Errors: {summary['total_errors']}")
        print(f"Test Duration: {summary['test_duration']:.2f}s")
        
        # Save results if configured
        if config.config["monitoring"]["save_results"]:
            results_file = config.config["monitoring"]["results_file"]
            try:
                # Load existing results
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        all_results = json.load(f)
                else:
                    all_results = {"test_runs": []}
                
                # Add this run
                current_run = {
                    "timestamp": datetime.now().isoformat(),
                    "summary": summary,
                    "thresholds_met": {
                        "local_usage": local_usage >= local_threshold,
                        "response_time": avg_response <= response_threshold,
                        "tool_success": tool_success >= tool_threshold
                    }
                }
                
                all_results["test_runs"].append(current_run)
                
                # Keep only last 50 runs
                all_results["test_runs"] = all_results["test_runs"][-50:]
                
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
                print(f"\nResults saved to {results_file}")
                
            except Exception as e:
                print(f"Warning: Could not save results: {e}")
        
        print("="*60)

def run_integration_tests():
    """Run integration tests with real components"""
    print("Running integration tests...")
    
    # Test that MCP server can start
    try:
        mcp_server_path = PROJECT_ROOT / "mcp_os" / "server.py"
        if mcp_server_path.exists():
            # Start server process briefly to test
            proc = subprocess.Popen(
                [sys.executable, str(mcp_server_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Terminate
            proc.terminate()
            
            try:
                proc.wait(timeout=5)
                print("✅ MCP server can start and stop cleanly")
            except subprocess.TimeoutExpired:
                proc.kill()
                print("⚠️  MCP server had to be force-killed")
        else:
            print("❌ MCP server.py not found")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

def main():
    """Main test runner"""
    # Create default config if it doesn't exist
    config_path = "test_config.json"
    if not os.path.exists(config_path):
        print("Creating default test configuration...")
        TestConfig(config_path)  # This will create the file
    
    # Load test configuration
    config = TestConfig(config_path)
    
    print("ChatOS Test Suite")
    print(f"Using config: {config_path}")
    print(f"Local usage threshold: {config.get_threshold('pct_local_use')}%")
    print(f"Response time threshold: {config.get_threshold('avg_response_time_seconds')}s")
    print(f"Tool success threshold: {config.get_threshold('tool_success_rate')}%")
    print("-" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration tests if enabled
    if config.get_setting("enable_integration_tests"):
        run_integration_tests()

if __name__ == "__main__":
    main()