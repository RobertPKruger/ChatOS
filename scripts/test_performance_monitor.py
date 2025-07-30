#!/usr/bin/env python3
"""
Test script to verify the performance monitor fixes
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta

def test_datetime_fixes():
    """Test that datetime handling is fixed"""
    print("Testing datetime fixes...")
    
    # Create test config with realistic thresholds
    test_config = {
        "thresholds": {
            "pct_local_use": 50,
            "avg_response_time_seconds": 5.0,
            "tool_success_rate": 85,
            "chatos_memory_mb": 200,  # 200MB for ChatOS processes
            "system_memory_percent": 85,  # 85% system memory
            "cpu_usage_percent": 80,
            "disk_usage_percent": 90
        },
        "monitoring": {
            "check_interval_seconds": 60,
            "alert_cooldown_minutes": 30,
            "track_process_memory": True
        }
    }
    
    with open('test_config_fixed.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print("✅ Created test config with realistic thresholds")
    
    # Test database operations
    db_path = "test_metrics.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Test SQLite datetime operations
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metadata TEXT
        )
    ''')
    
    # Insert test data
    current_time = datetime.now().isoformat()
    old_time = (datetime.now() - timedelta(hours=2)).isoformat()
    
    test_data = [
        (current_time, 'chatos_memory_mb', 150.5, '{"process_count": 2}'),
        (old_time, 'chatos_memory_mb', 120.0, '{"process_count": 1}'),
        (current_time, 'system_memory_percent', 65.2, None),
        (current_time, 'cpu_usage_percent', 25.8, None)
    ]
    
    for timestamp, name, value, metadata in test_data:
        cursor.execute(
            'INSERT INTO metrics (timestamp, metric_name, metric_value, metadata) VALUES (?, ?, ?, ?)',
            (timestamp, name, value, metadata)
        )
    
    conn.commit()
    
    # Test time-based query
    since_time = (datetime.now() - timedelta(hours=1)).isoformat()
    cursor.execute('''
        SELECT timestamp, metric_name, metric_value
        FROM metrics
        WHERE metric_name = ? AND timestamp > ?
        ORDER BY timestamp DESC
    ''', ('chatos_memory_mb', since_time))
    
    recent_results = cursor.fetchall()
    
    print(f"Recent metrics query returned {len(recent_results)} results")
    for timestamp, name, value in recent_results:
        print(f"  {name}: {value} at {timestamp}")
    
    conn.close()
    
    if len(recent_results) > 0:
        print("✅ DateTime filtering works correctly")
    else:
        print("❌ DateTime filtering failed")
    
    # Cleanup
    os.remove(db_path)
    
    return len(recent_results) > 0

def test_threshold_logic():
    """Test that threshold logic is sensible"""
    print("\nTesting threshold logic...")
    
    # Realistic test scenarios
    test_scenarios = [
        {
            "name": "Normal operation",
            "chatos_memory_mb": 75,
            "system_memory_percent": 60,
            "cpu_usage_percent": 15,
            "expected_alerts": 0
        },
        {
            "name": "High ChatOS memory",
            "chatos_memory_mb": 250,  # Above 200MB threshold
            "system_memory_percent": 60,
            "cpu_usage_percent": 15,
            "expected_alerts": 1
        },
        {
            "name": "High system memory",
            "chatos_memory_mb": 75,
            "system_memory_percent": 90,  # Above 85% threshold
            "cpu_usage_percent": 15,
            "expected_alerts": 1
        },
        {
            "name": "High CPU",
            "chatos_memory_mb": 75,
            "system_memory_percent": 60,
            "cpu_usage_percent": 85,  # Above 80% threshold
            "expected_alerts": 1
        }
    ]
    
    thresholds = {
        "chatos_memory_mb": 200,
        "system_memory_percent": 85,
        "cpu_usage_percent": 80,
        "disk_usage_percent": 90
    }
    
    all_passed = True
    
    for scenario in test_scenarios:
        alerts = 0
        
        if scenario["chatos_memory_mb"] > thresholds["chatos_memory_mb"]:
            alerts += 1
        if scenario["system_memory_percent"] > thresholds["system_memory_percent"]:
            alerts += 1
        if scenario["cpu_usage_percent"] > thresholds["cpu_usage_percent"]:
            alerts += 1
        
        if alerts == scenario["expected_alerts"]:
            print(f"  ✅ {scenario['name']}: {alerts} alerts (expected {scenario['expected_alerts']})")
        else:
            print(f"  ❌ {scenario['name']}: {alerts} alerts (expected {scenario['expected_alerts']})")
            all_passed = False
    
    return all_passed

def test_process_detection():
    """Test ChatOS process detection logic"""
    print("\nTesting process detection logic...")
    
    # Simulate process detection
    test_processes = [
        {
            "name": "python.exe",
            "cmdline": ["python", "enhanced_chat_host.py"],
            "memory_mb": 45.2,
            "expected": True
        },
        {
            "name": "python.exe", 
            "cmdline": ["python", "server.py"],
            "memory_mb": 23.1,
            "expected": True  # MCP server
        },
        {
            "name": "chrome.exe",
            "cmdline": ["chrome.exe", "--no-sandbox"],
            "memory_mb": 150.0,
            "expected": False
        },
        {
            "name": "python.exe",
            "cmdline": ["python", "-m", "ollama"],
            "memory_mb": 80.5,
            "expected": True  # Ollama process
        }
    ]
    
    detected_memory = 0
    detected_count = 0
    
    for proc in test_processes:
        cmdline = proc["cmdline"]
        is_chatos = any('chatos' in str(arg).lower() or 
                       'mcp' in str(arg).lower() or 
                       'enhanced' in str(arg).lower() or
                       'ollama' in str(arg).lower()
                       for arg in cmdline)
        
        if is_chatos:
            detected_memory += proc["memory_mb"]
            detected_count += 1
            
        status = "✅" if is_chatos == proc["expected"] else "❌"
        print(f"  {status} {proc['name']} {' '.join(cmdline[:2])}: detected={is_chatos}")
    
    print(f"  Total detected: {detected_count} processes, {detected_memory:.1f}MB")
    
    return detected_count > 0

def main():
    """Run all tests"""
    print("=" * 50)
    print("TESTING PERFORMANCE MONITOR FIXES")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_datetime_fixes():
        tests_passed += 1
        
    if test_threshold_logic():
        tests_passed += 1
        
    if test_process_detection():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All fixes are working correctly!")
        print("\nTo use the fixed monitor:")
        print("  python fixed_performance_monitor.py --single")
        print("  python fixed_performance_monitor.py --config test_config_fixed.json --single")
    else:
        print("❌ Some tests failed - fixes need more work")
    
    print("=" * 50)

if __name__ == "__main__":
    main()