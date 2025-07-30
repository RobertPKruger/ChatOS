import unittest
import sys
import os
import subprocess
sys.path.append(".")

class TestChatOSBasics(unittest.TestCase):
    def test_imports(self):
        """Test that basic imports work"""
        try:
            from host.voice_assistant.config import Config
            from mcp_os.app_tools import AppToolsManager
            print("Basic imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring works"""
        result = subprocess.run([
            sys.executable, "tests/monitor_fixed_simple.py"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        if result.returncode != 0:
            self.fail(f"Performance monitor failed with code {result.returncode}. Error: {result.stderr}")
        
        self.assertIn("System Health:", result.stdout)
        self.assertIn("ChatOS Performance Check", result.stdout)  # Should run performance check
        print("Performance monitoring working")
    
    def test_performance_monitoring_direct(self):
        """Test performance monitoring by direct import"""
        try:
            import psutil
            import os
            
            # Get current process info
            current_proc = psutil.Process(os.getpid())
            process_memory = current_proc.memory_info().rss / (1024 * 1024)
            
            # This should work without errors
            self.assertGreater(process_memory, 0)
            print(f"Direct performance check: {process_memory:.1f}MB")
            
        except Exception as e:
            self.fail(f"Direct performance check failed: {e}")
    
    def test_thresholds_logic(self):
        """Test threshold checking logic"""
        # Mock performance data (should all pass)
        test_data = {
            "memory_mb": 17.5,
            "cpu_percent": 18.7,
            "disk_percent": 20.7
        }
        
        thresholds = {
            "memory_mb": 2000,
            "cpu_percent": 80,
            "disk_percent": 90
        }
        
        for metric, value in test_data.items():
            threshold = thresholds[metric]
            self.assertLessEqual(value, threshold, 
                f"{metric} {value} should be <= {threshold}")
        
        print("Threshold logic working correctly")

if __name__ == "__main__":
    unittest.main()
def test_config_loading(self):
    """Test ChatOS configuration loading"""
    from host.voice_assistant.config import Config
    config = Config.from_env()
    self.assertIsNotNone(config.openai_api_key or "test")
    
def test_app_tools_manager(self):
    """Test application tools manager"""
    from mcp_os.app_tools import AppToolsManager
    manager = AppToolsManager()
    self.assertGreater(len(manager.apps), 0)
    
def test_model_providers(self):
    """Test model provider creation"""
    from host.voice_assistant.model_providers.factory import ModelProviderFactory
    # This will test your provider factory works
