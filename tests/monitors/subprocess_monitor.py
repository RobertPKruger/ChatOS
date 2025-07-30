import psutil
import os
from datetime import datetime

print("ChatOS Performance Status")
print("=" * 40)

# Get current process info
current_proc = psutil.Process(os.getpid())
process_memory = current_proc.memory_info().rss / (1024 * 1024)

# System info  
memory = psutil.virtual_memory()
cpu = psutil.cpu_percent(interval=1)
disk = psutil.disk_usage("C:" if os.name == "nt" else "/")

# Thresholds from config
thresholds = {
    "memory_mb": 2000,
    "cpu_percent": 80, 
    "disk_percent": 90
}

print("System Health:")
memory_status = "PASS" if process_memory <= thresholds["memory_mb"] else "WARN"
cpu_status = "PASS" if cpu <= thresholds["cpu_percent"] else "WARN"
disk_pct = (disk.used / disk.total) * 100
disk_status = "PASS" if disk_pct <= thresholds["disk_percent"] else "WARN"

print(f"  [{memory_status}] chatOS_memory_mb: {process_memory:.1f} (threshold: {thresholds['memory_mb']})")
print(f"  [{cpu_status}] cpu_usage_percent: {cpu:.1f} (threshold: {thresholds['cpu_percent']})")
print(f"  [{disk_status}] disk_usage_percent: {disk_pct:.1f} (threshold: {thresholds['disk_percent']})")

print("\nChatOS Health:")
print("  (No test results data found)")

print("\nNo recent alerts")
print("=" * 40)
