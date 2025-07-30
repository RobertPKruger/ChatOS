#!/usr/bin/env python3
"""
Windows-compatible fix for performance monitor
No emoji characters that cause encoding issues
"""

import os
import subprocess
import sys

def create_fixed_monitor():
    """Create the fixed performance monitor file"""
    
    # Simple fixed code without emoji
    fixed_code = '''#!/usr/bin/env python3
import psutil
import json
import os
from datetime import datetime

class SimplePerformanceMonitor:
    def __init__(self):
        self.thresholds = {
            "chatos_memory_mb": 200,
            "system_memory_percent": 85,
            "cpu_usage_percent": 80,
            "disk_usage_percent": 90
        }
    
    def get_chatos_processes(self):
        """Find ChatOS processes"""
        total_memory = 0
        process_count = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline'] or []
                    if any('chatos' in str(arg).lower() or 
                           'enhanced' in str(arg).lower() or
                           'mcp' in str(arg).lower() or
                           'server.py' in str(arg).lower() or
                           'ollama' in str(arg).lower()
                           for arg in cmdline):
                        memory_mb = proc.memory_info().rss / (1024 * 1024)
                        total_memory += memory_mb
                        process_count += 1
            except:
                continue
        
        return total_memory, process_count
    
    def run_check(self):
        """Run performance check"""
        print("ChatOS Performance Check (FIXED)")
        print("=" * 40)
        
        # Get system metrics
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('C:' if os.name == 'nt' else '/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Get ChatOS processes
        chatos_memory, process_count = self.get_chatos_processes()
        
        print("System Health:")
        
        # Check ChatOS memory (realistic threshold)
        if chatos_memory <= self.thresholds["chatos_memory_mb"]:
            print(f"  [OK] ChatOS memory: {chatos_memory:.1f}MB (threshold: {self.thresholds['chatos_memory_mb']}MB)")
        else:
            print(f"  [WARN] ChatOS memory: {chatos_memory:.1f}MB (threshold: {self.thresholds['chatos_memory_mb']}MB)")
        
        # Check system memory percentage (not absolute)
        if memory.percent <= self.thresholds["system_memory_percent"]:
            print(f"  [OK] System memory: {memory.percent:.1f}% (threshold: {self.thresholds['system_memory_percent']}%)")
        else:
            print(f"  [WARN] System memory: {memory.percent:.1f}% (threshold: {self.thresholds['system_memory_percent']}%)")
        
        # Check CPU
        if cpu <= self.thresholds["cpu_usage_percent"]:
            print(f"  [OK] CPU usage: {cpu:.1f}% (threshold: {self.thresholds['cpu_usage_percent']}%)")
        else:
            print(f"  [WARN] CPU usage: {cpu:.1f}% (threshold: {self.thresholds['cpu_usage_percent']}%)")
        
        # Check disk
        if disk_percent <= self.thresholds["disk_usage_percent"]:
            print(f"  [OK] Disk usage: {disk_percent:.1f}% (threshold: {self.thresholds['disk_usage_percent']}%)")
        else:
            print(f"  [WARN] Disk usage: {disk_percent:.1f}% (threshold: {self.thresholds['disk_usage_percent']}%)")
        
        print(f"\\nChatOS Processes: {process_count} detected")
        if process_count > 0:
            print(f"  Total memory: {chatos_memory:.1f}MB")
        else:
            print("  (No ChatOS processes currently running)")
        
        print("=" * 40)
        print("FIXED: No more false memory alerts!")
        print("FIXED: Realistic thresholds for ChatOS")

if __name__ == "__main__":
    monitor = SimplePerformanceMonitor()
    monitor.run_check()
'''

    try:
        with open('monitor_fixed_simple.py', 'w', encoding='utf-8') as f:
            f.write(fixed_code)
        print("Created monitor_fixed_simple.py")
        return True
    except Exception as e:
        print(f"Error creating file: {e}")
        return False

def test_new_monitor():
    """Test the new monitor"""
    print("\\nTesting the new monitor:")
    print("-" * 30)
    
    try:
        result = subprocess.run([
            sys.executable, 'monitor_fixed_simple.py'
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"Error running new monitor: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("New monitor timed out")
        return False
    except FileNotFoundError:
        print("Could not find monitor_fixed_simple.py")
        return False

def main():
    """Simple main function"""
    print("AUTO-FIXING YOUR PERFORMANCE MONITOR")
    print("=" * 50)
    
    # Create the fixed monitor
    print("Step 1: Creating fixed monitor...")
    if create_fixed_monitor():
        print("[OK] Fixed monitor created")
    else:
        print("[ERROR] Could not create fixed monitor")
        return
    
    # Test it
    print("\\nStep 2: Testing fixed monitor...")
    if test_new_monitor():
        print("\\n[SUCCESS] Fixed monitor is working!")
        print("\\nTo use it: python monitor_fixed_simple.py")
    else:
        print("\\n[ERROR] Fixed monitor failed to run")
    
    print("=" * 50)

if __name__ == "__main__":
    main()