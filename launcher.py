#!/usr/bin/env python3
"""
ChatOS Process Manager - Advanced launcher with full lifecycle management
Handles server startup, health checking, graceful shutdown, and error recovery
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
import atexit
from pathlib import Path
from datetime import datetime
from typing import Optional

class ChatOSLauncher:
    """Advanced ChatOS process manager with full lifecycle control"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.server_process: Optional[subprocess.Popen] = None
        self.host_process: Optional[subprocess.Popen] = None
        self.server_venv = self.project_root / ".venv-server"
        self.host_venv = self.project_root / ".venv-host"
        self.logs_dir = self.project_root / "logs"
        self.running = True
        self.show_console_output = True  # NEW: Control console output
        
        # Create logs directory
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging with UTF-8 encoding for Windows
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"launcher_{timestamp}.log"
        
        # Configure console handler with proper encoding
        console_handler = logging.StreamHandler(sys.stdout)
        if os.name == 'nt':  # Windows
            # Set UTF-8 encoding for console output
            import codecs
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                console_handler
            ]
        )
        self.logger = logging.getLogger("ChatOSLauncher")
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def check_prerequisites(self) -> bool:
        """Verify all prerequisites are met"""
        self.logger.info("Checking prerequisites...")
        
        # Check virtual environments
        if not (self.server_venv / "Scripts" / "activate.bat").exists():
            self.logger.error(f"Server venv not found: {self.server_venv}")
            return False
            
        if not (self.host_venv / "Scripts" / "activate.bat").exists():
            self.logger.error(f"Host venv not found: {self.host_venv}")
            return False
        
        # Check critical files
        if not (self.project_root / "mcp_os" / "server.py").exists():
            self.logger.error("MCP server script not found")
            return False
            
        if not (self.project_root / "host" / "enhanced_chat_host.py").exists():
            self.logger.error("Host script not found")
            return False
        
        # Check .env file (optional but warn)
        if not (self.project_root / ".env").exists():
            self.logger.warning(".env file not found - using defaults")
        
        self.logger.info("[OK] Prerequisites check passed")
        return True
    
    def start_server(self) -> bool:
        """Start the MCP server with proper process management"""
        self.logger.info("Starting MCP server...")
        
        # Build server command
        if os.name == 'nt':  # Windows
            activate_script = self.server_venv / "Scripts" / "activate.bat"
            # Use a more robust approach for Windows
            python_exe = self.server_venv / "Scripts" / "python.exe"
            server_script = self.project_root / "mcp_os" / "server.py"
            
            # Check if python executable exists
            if not python_exe.exists():
                self.logger.error(f"Python executable not found: {python_exe}")
                return False
                
            cmd = [str(python_exe), str(server_script)]
            self.logger.info(f"Server command: {' '.join(cmd)}")
            
        else:  # Unix-like
            activate_script = self.server_venv / "bin" / "activate"
            cmd = [
                "bash", "-c", 
                f'source "{activate_script}" && cd "{self.project_root}" && python mcp_os/server.py'
            ]
        
        try:
            # Start server process
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.logs_dir / f"server_{timestamp}.log"
            
            self.logger.info(f"Server logs will be written to: {log_file}")
            
            with open(log_file, 'w') as log:
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=self.project_root,
                    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
                )
            
            self.logger.info(f"Server started with PID: {self.server_process.pid}")
            
            # Give it a moment to start
            time.sleep(1)
            
            # Check if it's still alive
            if self.server_process.poll() is not None:
                self.logger.error(f"Server died immediately with exit code: {self.server_process.returncode}")
                # Read the log file to see what went wrong
                try:
                    with open(log_file, 'r') as log:
                        error_output = log.read()
                        if error_output:
                            self.logger.error(f"Server error output:\n{error_output}")
                        else:
                            self.logger.error("No error output captured")
                except Exception as e:
                    self.logger.error(f"Could not read server log: {e}")
                return False
            
            return self.wait_for_server_ready()
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False
    
    def wait_for_server_ready(self, timeout: int = 30) -> bool:
        """Wait for server to be ready with health checking"""
        self.logger.info("Waiting for server to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.running:
                return False
                
            # Check if process is still alive
            if self.server_process and self.server_process.poll() is not None:
                self.logger.error("Server process died during startup")
                return False
            
            # Simple health check - you could enhance this
            # For now, just wait a reasonable time and check process is alive
            time.sleep(1)
            
            if time.time() - start_time > 5:  # After 5 seconds, assume ready
                self.logger.info("[OK] Server appears to be ready")
                return True
        
        self.logger.error(f"Server health check timeout after {timeout} seconds")
        return False
    
    def start_host(self) -> bool:
        """Start the host application"""
        self.logger.info("Starting ChatOS host application...")
        
        # Build host command
        if os.name == 'nt':  # Windows
            python_exe = self.host_venv / "Scripts" / "python.exe"
            host_script = self.project_root / "host" / "enhanced_chat_host.py"
            
            # Check if python executable exists
            if not python_exe.exists():
                self.logger.error(f"Host Python executable not found: {python_exe}")
                return False
                
            cmd = [str(python_exe), str(host_script)]
            self.logger.info(f"Host command: {' '.join(cmd)}")
            
        else:  # Unix-like
            activate_script = self.host_venv / "bin" / "activate"
            cmd = [
                "bash", "-c",
                f'source "{activate_script}" && cd "{self.project_root}" && python host/enhanced_chat_host.py'
            ]
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.logs_dir / f"host_{timestamp}.log"
            
            self.logger.info(f"Host logs will be written to: {log_file}")
            
            # Choose output destination based on console output setting
            if self.show_console_output:
                # Show host output in console (but still log to file)
                import subprocess
                from threading import Thread
                
                def log_output(process, log_file):
                    """Log output to both file and console"""
                    with open(log_file, 'w', encoding='utf-8') as log:
                        while True:
                            output = process.stdout.readline()
                            if output == b'' and process.poll() is not None:
                                break
                            if output:
                                decoded_line = output.decode('utf-8', errors='replace').rstrip()
                                # Show in console with timestamp and prefix
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                print(f"[{timestamp}] {decoded_line}")
                                # Write to log file
                                log.write(decoded_line + '\n')
                                log.flush()
                                # Ensure console output is flushed
                                sys.stdout.flush()
                
                self.host_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=self.project_root,
                    bufsize=1,  # Line buffered
                    universal_newlines=False,  # Keep as bytes for proper decoding
                    env={**os.environ, "CHATOS_CONSOLE_OUTPUT": "1"}  # Signal to host for console logging
                )
                
                # Start background thread to handle output
                output_thread = Thread(target=log_output, args=(self.host_process, log_file))
                output_thread.daemon = True
                output_thread.start()
                
            else:
                # Original behavior - output only to log file
                with open(log_file, 'w') as log:
                    self.host_process = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=self.project_root
                    )
            
            self.logger.info(f"Host started with PID: {self.host_process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start host: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor both processes and handle failures"""
        self.logger.info("Starting process monitoring...")
        
        while self.running:
            try:
                # Check server process
                if self.server_process and self.server_process.poll() is not None:
                    self.logger.error("Server process died unexpectedly")
                    self.running = False
                    break
                
                # Check host process
                if self.host_process and self.host_process.poll() is not None:
                    exit_code = self.host_process.returncode
                    if exit_code == 0:
                        self.logger.info("Host process exited normally")
                    else:
                        self.logger.error(f"Host process exited with code: {exit_code}")
                        # Try to read the host log to show the error
                        try:
                            # Find the most recent host log
                            host_logs = list(self.logs_dir.glob("host_*.log"))
                            if host_logs:
                                latest_log = max(host_logs, key=lambda p: p.stat().st_mtime)
                                with open(latest_log, 'r') as log:
                                    error_output = log.read()
                                    if error_output:
                                        self.logger.error(f"Host error output:\n{error_output}")
                        except Exception as e:
                            self.logger.error(f"Could not read host log: {e}")
                    self.running = False
                    break
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in process monitoring: {e}")
                break
    
    def cleanup(self):
        """Clean shutdown of all processes"""
        self.logger.info("Starting cleanup...")
        
        # Stop host process
        if self.host_process and self.host_process.poll() is None:
            self.logger.info("Stopping host process...")
            try:
                self.host_process.terminate()
                self.host_process.wait(timeout=10)
                self.logger.info("[OK] Host process stopped")
            except subprocess.TimeoutExpired:
                self.logger.warning("Host process didn't stop gracefully, killing...")
                self.host_process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping host: {e}")
        
        # Stop server process
        if self.server_process and self.server_process.poll() is None:
            self.logger.info("Stopping server process...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                self.logger.info("[OK] Server process stopped")
            except subprocess.TimeoutExpired:
                self.logger.warning("Server process didn't stop gracefully, killing...")
                self.server_process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping server: {e}")
        
        self.logger.info("Cleanup completed")
    
    def run(self) -> int:
        """Main execution flow"""
        try:
            print("╔══════════════════════════════════════════════════════════════════╗")
            print("║                        ChatOS Launcher                          ║")
            print("║                   Python Process Manager                        ║")
            print("╚══════════════════════════════════════════════════════════════════╝")
            print()
            
            # Prerequisites check
            if not self.check_prerequisites():
                return 1
            
            # Start server
            if not self.start_server():
                return 1
            
            # Start host
            if not self.start_host():
                return 1
            
            print("╔══════════════════════════════════════════════════════════════════╗")
            print("║                    ChatOS Running                               ║")
            print("║                                                                  ║")
            print("║  • Server: Running with process monitoring                      ║")
            print("║  • Host: Running with full logging                              ║")
            print("║  • Logs: logs/ directory                                        ║")
            if self.show_console_output:
                print("║  • Console: Showing host output below                           ║")
            print("║                                                                  ║")
            print("║  Press Ctrl+C to stop ChatOS gracefully                        ║")
            print("╚══════════════════════════════════════════════════════════════════╝")
            print()
            
            if self.show_console_output:
                print("=" * 70)
                print("HOST APPLICATION OUTPUT:")
                print("=" * 70)
            
            # Monitor processes
            self.monitor_processes()
            
            # Determine exit code
            if self.host_process:
                return self.host_process.returncode or 0
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            return 0
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return 1
        finally:
            self.cleanup()

def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ChatOS Process Manager")
    parser.add_argument("--console", action="store_true", 
                       help="Show host application output in console")
    parser.add_argument("--quiet", action="store_true",
                       help="Hide host application output (logs only)")
    
    args = parser.parse_args()
    
    launcher = ChatOSLauncher()
    
    # Set console output preference
    if args.console:
        launcher.show_console_output = True
    elif args.quiet:
        launcher.show_console_output = False
    # Default is True (show console output)
    
    return launcher.run()

if __name__ == "__main__":
    sys.exit(main())