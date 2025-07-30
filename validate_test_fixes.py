#!/usr/bin/env python3
"""
Validate that all test fixes are working
"""

import subprocess
import sys
from pathlib import Path

def validate_apps_config():
    """Test that apps_config.json can be found"""
    print("🔍 Testing apps_config.json path...")
    
    # Test the path logic that the test uses
    test_dir = Path("tests")
    project_root = test_dir.parent if test_dir.exists() else Path(".")
    apps_config_path = project_root / "mcp_os" / "apps_config.json"
    
    if apps_config_path.exists():
        print(f"  ✅ apps_config.json found at: {apps_config_path}")
        return True
    else:
        print(f"  ❌ apps_config.json NOT found at: {apps_config_path}")
        return False

def validate_performance_monitor():
    """Test that performance monitor works"""
    print("\n🔍 Testing performance monitor...")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/monitor_fixed_simple.py"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0 and "System Health:" in result.stdout:
            print("  ✅ Performance monitor working")
            return True
        else:
            print(f"  ❌ Performance monitor failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Performance monitor error: {e}")
        return False

def validate_test_structure():
    """Validate test file structure"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "tests/test_suite.py",
        "tests/test_basics.py", 
        "tests/monitor_fixed_simple.py",
        "mcp_os/apps_config.json",
        "host/voice_assistant/state.py"
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_good = False
    
    return all_good

def main():
    """Run all validations"""
    print("🧪 VALIDATING TEST FIXES")
    print("=" * 40)
    
    all_passed = True
    
    if not validate_test_structure():
        all_passed = False
    
    if not validate_apps_config():
        all_passed = False
    
    if not validate_performance_monitor():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("\n🚀 Ready to run tests:")
        print("   python run_tests_fixed.py")
    else:
        print("❌ Some validations failed")
        print("   Check the errors above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
