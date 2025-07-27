# app_tools.py - COMPLETE IMPLEMENTATION with PDF support
"""
Application management tools for the MCP server with comprehensive file opening capabilities
"""

import subprocess
import os
import pathlib
import shutil
import platform
import json
import logging
import webbrowser
import re
from typing import Dict, List, Optional, Any, Union

class AppToolsManager:
    """Manages application launching and configuration with file opening support"""
    
    def __init__(self, config_path: str = "apps_config.json"):
        self.config_path = config_path
        self.apps: Dict[str, Any] = {}
        self.aliases: Dict[str, Any] = {}
        self.current_os = self._detect_os()
        self.load_config()
    
    def _detect_os(self) -> str:
        """Detect current operating system"""
        system = platform.system().lower()
        os_map = {
            "windows": "windows",
            "darwin": "darwin",  # macOS
            "linux": "linux"
        }
        return os_map.get(system, "linux")
    
    def load_config(self) -> None:
        """Load OS-specific application configurations from JSON file"""
        try:
            # Try to load from the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(script_dir, self.config_path)
            
            if not os.path.exists(config_file):
                # Try current working directory
                config_file = self.config_path
                
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            if self.current_os in config:
                self.apps = config[self.current_os]
                logging.info(f"Loaded {len(self.apps)} apps for {self.current_os}")
            else:
                logging.warning(f"No configuration found for OS: {self.current_os}")
                self.apps = {}
                
            # Load aliases if available
            if "aliases" in config and self.current_os in config["aliases"]:
                self.aliases = config["aliases"][self.current_os]
                # Build reverse mapping for quick lookup
                aliases_reverse = {}
                for primary_name, alias_list in self.aliases.items():
                    for alias in alias_list:
                        aliases_reverse[alias.lower()] = primary_name
                self.aliases["_reverse"] = aliases_reverse
                logging.info(f"Loaded {len(self.aliases)-1} alias groups for {self.current_os}")
                
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            self.apps = {}
            self.aliases = {}
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in configuration file: {e}")
            self.apps = {}
            self.aliases = {}
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            self.apps = {}
            self.aliases = {}
    
    def _extract_paths_from_app_config(self, app_config: Union[List[str], Dict[str, Any]]) -> List[str]:
        """Extract paths from app configuration, handling both old and new formats"""
        if isinstance(app_config, list):
            # Old format: direct list of paths
            return app_config
        elif isinstance(app_config, dict):
            # New format: dictionary with 'paths' key
            return app_config.get('paths', [])
        else:
            # Fallback: convert to list
            return [str(app_config)]
    
    def find_executable(self, app_paths: List[str]) -> Optional[Union[str, List[str]]]:
        """Find the first available executable from a list of possible paths"""
        username = os.getenv('USERNAME', '') if platform.system() == "Windows" else os.getenv('USER', '')
        
        for path in app_paths:
            # Handle special protocols (Windows)
            if path.startswith("ms-"):
                return path
                
            # Replace {username} placeholder
            if '{username}' in path:
                path = path.replace('{username}', username)
                
            # For macOS 'open' commands, return the full command list
            if platform.system() == "Darwin" and isinstance(app_paths, list) and len(app_paths) > 1 and app_paths[0] == "open":
                return app_paths
                
            # First try to find in PATH
            if not os.path.sep in path:  # It's just an executable name
                if shutil.which(path):
                    return path
            else:  # It's a full path
                if os.path.exists(path):
                    return path
        
        return None
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        return re.sub(r'[^a-z0-9]', '', text.lower().strip())
    
    def resolve_app_name(self, app_name: str) -> str:
        """Resolve app name using aliases and fuzzy matching"""
        lower_name = app_name.lower().strip()
        normalized_name = self.normalize_text(lower_name)

        # Check for exact app match
        if lower_name in self.apps:
            return lower_name

        # Check normalized aliases
        if "_reverse" in self.aliases:
            for alias, primary in self.aliases["_reverse"].items():
                if self.normalize_text(alias) == normalized_name:
                    return primary

        # Raw domain-as-command fallback
        for key in self.apps:
            if key.replace("www.", "").startswith(lower_name):
                return key

        # Fuzzy match
        for alias, primary in self.aliases.get("_reverse", {}).items():
            if normalized_name in self.normalize_text(alias):
                return primary
            
        # Fallback: If the input looks like a domain, treat it that way
        if '.' in lower_name and not lower_name.startswith(('http://', 'https://')):
            return lower_name

        return app_name
    
    def find_similar_apps(self, app_name: str, threshold: float = 0.6) -> List[str]:
        """Find apps with similar names using simple string matching"""
        similar = []
        lower_name = app_name.lower()
        
        # Check all primary app names and aliases
        all_names = list(self.apps.keys())
        if "_reverse" in self.aliases:
            all_names.extend(self.aliases["_reverse"].keys())
        
        for name in all_names:
            # Simple substring matching
            if lower_name in name or name in lower_name:
                similar.append(name)
            # Check if they share significant characters
            elif len(set(lower_name) & set(name)) > len(lower_name) * threshold:
                similar.append(name)
        
        return list(set(similar))[:5]  # Return up to 5 suggestions
    
    def get_app_config(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Get application configuration, returns dict format regardless of storage format"""
        resolved_name = self.resolve_app_name(app_name)
        app_config = self.apps.get(resolved_name)
        
        if not app_config:
            return None
        
        # Convert to consistent dict format
        if isinstance(app_config, list):
            return {
                "paths": app_config,
                "description": f"{resolved_name} application",
                "category": "unknown",
                "keywords": [],
                "supports_file_opening": False
            }
        elif isinstance(app_config, dict):
            return app_config
        else:
            return {
                "paths": [str(app_config)],
                "description": f"{resolved_name} application",
                "category": "unknown", 
                "keywords": [],
                "supports_file_opening": False
            }
    
    def launch_app(self, app_name: str, file_path: str = None) -> str:
        """Launch the given application or protocol, optionally with a file"""
        # Resolve any aliases to the primary app name
        resolved_name = self.resolve_app_name(app_name)

        logging.info(f"[launch_app] Input: '{app_name}' → Resolved: '{resolved_name}'" + 
                    (f" with file: '{file_path}'" if file_path else ""))
        
        app_config = self.get_app_config(resolved_name)

        if not app_config:
            # If it looks like a domain or navigation intent, try smart_navigate
            if '.' in app_name or app_name.lower().startswith(('www.', 'http')):
                return self.smart_navigate(app_name)

            # Try to find similar apps for helpful error message
            similar = self.find_similar_apps(app_name)
            if similar:
                return f"Application '{app_name}' not found. Did you mean: {', '.join(similar)}?"
            return f"Application '{app_name}' not found. Use 'list_apps' to see available apps."
        
        # Extract paths from configuration
        executables = app_config.get('paths', [])
        
        # Check for special 'url:' pseudo-apps
        if isinstance(executables, list) and any(e.startswith("url:") for e in executables):
            url = next((e[4:] for e in executables if e.startswith("url:")), None)
            if url:
                try:
                    webbrowser.open(url)
                    return f"Opened browser to {url}"
                except Exception as e:
                    return f"Failed to open browser: {e}"

        username = os.getenv('USERNAME', '') if platform.system() == "Windows" else os.getenv('USER', '')
        errors = []
        
        # Handle macOS 'open' command format
        if platform.system() == "Darwin" and isinstance(executables, list) and len(executables) > 1 and executables[0] == "open":
            try:
                cmd = list(executables)  # Copy the command
                if file_path:
                    cmd.append(file_path)
                subprocess.Popen(cmd)
                return f"Launched '{app_name}'" + (f" with {file_path}" if file_path else "")
            except Exception as e:
                return f"Failed to launch '{app_name}': {str(e)}"
        
        # Handle single executable or list of alternatives
        exe_list = executables if isinstance(executables, list) else [executables]
        
        for exe in exe_list:
            try:
                # Expand potential placeholders
                path = exe.replace('{username}', username)
                
                # Windows-specific handling
                if platform.system() == "Windows":
                    # Protocol handlers (e.g. ms-settings:)
                    if path.endswith(":"):
                        os.startfile(path)
                        return f"Launched '{app_name}'"
                        
                    # Handle 'start' commands (for Office apps)
                    elif path.startswith("start "):
                        cmd = path
                        if file_path:
                            cmd += f' "{file_path}"'
                        subprocess.Popen(cmd, shell=True)
                        return f"Launched '{app_name}'" + (f" with {file_path}" if file_path else "")
                        
                    # Absolute path - check if it exists first
                    elif os.path.isabs(path):
                        if os.path.exists(path):
                            cmd = [path]
                            if file_path:
                                cmd.append(file_path)
                            subprocess.Popen(cmd)
                            return f"Launched '{app_name}'" + (f" with {file_path}" if file_path else "")
                        else:
                            errors.append(f"Path not found: {path}")
                            continue
                            
                    # Try to find in PATH
                    elif shutil.which(path):
                        cmd = [path]
                        if file_path:
                            cmd.append(file_path)
                        subprocess.Popen(cmd, shell=False)
                        return f"Launched '{app_name}'" + (f" with {file_path}" if file_path else "")
                    else:
                        errors.append(f"Not found in PATH: {path}")
                        
                # Linux/Unix handling
                else:
                    if shutil.which(path) or os.path.exists(path):
                        cmd = [path]
                        if file_path:
                            cmd.append(file_path)
                        subprocess.Popen(cmd, shell=False)
                        return f"Launched '{app_name}'" + (f" with {file_path}" if file_path else "")
                    else:
                        errors.append(f"Not found: {path}")
                        
            except Exception as e:
                errors.append(f"Failed to launch '{exe}': {str(e)}")
                continue

        return f"Failed to launch '{app_name}'. Errors: {'; '.join(errors)}"
    
    def open_file_with_app(self, file_path: str, app_name: str = None) -> str:
        """Open a specific file with an application (auto-detect app if not specified)"""
        try:
            # Expand path
            if file_path.startswith("~/"):
                file_path = os.path.expanduser(file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            
            # If no app specified, try to auto-detect based on file extension
            if not app_name:
                file_ext = pathlib.Path(file_path).suffix.lower()
                
                # Find apps that support this file extension
                candidates = []
                for app_key, app_config in self.apps.items():
                    if isinstance(app_config, dict):
                        supported_exts = app_config.get('file_extensions', [])
                        if file_ext in supported_exts:
                            candidates.append(app_key)
                
                if candidates:
                    # Prefer specific apps for specific file types
                    if file_ext == '.pdf' and 'acrobat' in candidates:
                        app_name = 'acrobat'
                    elif file_ext in ['.doc', '.docx'] and 'word' in candidates:
                        app_name = 'word'
                    elif file_ext in ['.xls', '.xlsx'] and 'excel' in candidates:
                        app_name = 'excel'
                    elif file_ext == '.txt' and 'notepad' in candidates:
                        app_name = 'notepad'
                    else:
                        app_name = candidates[0]  # Use first available
                else:
                    # No specific app found, try system default
                    if platform.system() == "Windows":
                        os.startfile(file_path)
                        return f"Opened {file_path} with system default application"
                    else:
                        subprocess.Popen(['open', file_path] if platform.system() == "Darwin" else ['xdg-open', file_path])
                        return f"Opened {file_path} with system default application"
            
            # Launch the specified or detected app with the file
            return self.launch_app(app_name, file_path)
            
        except Exception as e:
            return f"Failed to open file: {e}"
    
    def find_pdf_files(self, directory_path: str = "~/Desktop") -> List[str]:
        """Find PDF files in a directory - FIXED for OneDrive"""
        try:
            # Handle the OneDrive desktop issue
            if directory_path == "~/Desktop":
                # Try OneDrive desktop first
                onedrive_desktop = os.path.expanduser("~/OneDrive/Desktop")
                if os.path.exists(onedrive_desktop):
                    directory_path = onedrive_desktop
                else:
                    directory_path = os.path.expanduser(directory_path)
            elif directory_path.startswith("~/"):
                directory_path = os.path.expanduser(directory_path)
            
            pdf_files = []
            if os.path.exists(directory_path) and os.path.isdir(directory_path):
                for file in os.listdir(directory_path):
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(directory_path, file))
            
            return pdf_files
        except Exception as e:
            logging.error(f"Error finding PDF files: {e}")
            return []
    
    def smart_navigate(self, query: str) -> str:
        """Interprets user intent to open a website"""
        normalized = query.strip().lower()

        # Remove common leading phrases
        normalized = re.sub(r'^(go to|navigate to|open|launch)\s+', '', normalized)
        normalized = re.sub(r'( for me| please)$', '', normalized)

        # Check for existing domain indicators
        if re.search(r'\.(com|net|org|gov|edu|io|ai)(/|$)', normalized):
            if not normalized.startswith("http"):
                normalized = "https://" + normalized
            try:
                webbrowser.open(normalized)
                return f"Opened {normalized}"
            except Exception as e:
                return f"Failed to open URL '{normalized}': {e}"

        # If no domain, fallback to search
        search_url = f"https://www.google.com/search?q={query}"
        try:
            webbrowser.open(search_url)
            return f"Opened browser to search: {query}"
        except Exception as e:
            return f"Failed to perform search for '{query}': {e}"
    
    def open_url(self, url: str) -> str:
        """Launch the default web browser to the specified URL"""
        try:
            if not url.startswith("http"):
                url = "https://" + url
            webbrowser.open(url)
            return f"Opened {url}"
        except Exception as e:
            return f"Failed to open URL '{url}': {e}"
    
    def list_apps(self) -> str:
        """List all available applications and their aliases"""
        apps = sorted(self.apps.keys())
        result = f"Available applications on {platform.system()}:\n\n"
        
        if self.aliases and "_reverse" in self.aliases:
            # Group by primary app with aliases
            for app in apps:
                aliases = []
                for alias, primary in self.aliases["_reverse"].items():
                    if primary == app and alias != app:
                        aliases.append(alias)
                
                if aliases:
                    result += f"• {app} (aliases: {', '.join(sorted(aliases))})\n"
                else:
                    result += f"• {app}\n"
        else:
            # Simple list if no aliases defined
            result += ", ".join(apps)
        
        return result
    
    def test_app_availability(self) -> str:
        """Test which applications are actually available on this system"""
        available = []
        unavailable = []
        
        for app_name, app_config in self.apps.items():
            # Extract paths from configuration
            app_paths = self._extract_paths_from_app_config(app_config)
            exe = self.find_executable(app_paths)
            if exe:
                if isinstance(exe, list):
                    available.append(f"{app_name} -> {' '.join(exe)}")
                else:
                    available.append(f"{app_name} -> {exe}")
            else:
                unavailable.append(app_name)
        
        result = f"=== AVAILABLE APPS ({platform.system()}) ===\n"
        result += "\n".join(available)
        result += "\n\n=== UNAVAILABLE APPS ===\n"
        result += ", ".join(unavailable)
        
        # Add alias information
        if self.aliases and "_reverse" in self.aliases:
            result += f"\n\n=== LOADED ALIASES ===\n"
            result += f"Total aliases: {len(self.aliases['_reverse'])}"
        
        return str(result)

def register_app_tools(mcp, app_manager: AppToolsManager):
    """Register all app-related tools with the MCP server"""
    
    @mcp.tool()
    def launch_app(app_name: str, file_path: str = None) -> str:
        """Launch an application, optionally with a specific file.
        
        Args:
            app_name: Name of the application to launch
            file_path: Optional path to a file to open with the application
        """
        return app_manager.launch_app(app_name, file_path)
    
    @mcp.tool()
    def open_file_with_app(file_path: str, app_name: str = None) -> str:
        """Open a specific file with an application (auto-detects app if not specified).
        
        Args:
            file_path: Path to the file to open
            app_name: Optional application name (will auto-detect if not provided)
        """
        return app_manager.open_file_with_app(file_path, app_name)
    
    @mcp.tool()
    def open_pdf_with_acrobat(file_path: str) -> str:
        """Open a PDF file specifically with Adobe Acrobat/Reader.
        
        Args:
            file_path: Path to the PDF file
        """
        return app_manager.open_file_with_app(file_path, "acrobat")
    
    @mcp.tool()
    def find_and_open_first_pdf(directory_path: str = "~/Desktop", app_name: str = "acrobat") -> str:
        """Find the first PDF file in a directory and open it with the specified application.
        
        Args:
            directory_path: Directory to search for PDFs (default: ~/Desktop)
            app_name: Application to use for opening (default: acrobat)
        """
        try:
            pdf_files = app_manager.find_pdf_files(directory_path)
            
            if not pdf_files:
                return f"No PDF files found in {directory_path}"
            
            # Sort files to get consistent "first" file
            pdf_files.sort()
            first_pdf = pdf_files[0]
            
            result = app_manager.open_file_with_app(first_pdf, app_name)
            return f"Found {len(pdf_files)} PDF(s). Opening first one: {os.path.basename(first_pdf)}\n{result}"
            
        except Exception as e:
            return f"Error finding or opening PDF: {e}"
    
    @mcp.tool()
    def list_apps() -> str:
        """List all available applications and their aliases."""
        return app_manager.list_apps()
    
    @mcp.tool()
    def smart_navigate(query: str) -> str:
        """
        Interprets user intent to open a website, even from vague commands like 'go to Reddit' or 'navigate to Amazon'.
        If a domain is detected, it constructs a URL. Otherwise, it performs a fallback search.

        Args:
            query (str): The user intent (e.g., 'go to Reddit', 'navigate to irs.gov').

        Returns:
            str: Result of the navigation attempt.
        """
        return app_manager.smart_navigate(query)
    
    @mcp.tool()
    def open_url(url: str) -> str:
        """Launches the default web browser to the specified URL."""
        return app_manager.open_url(url)
    
    @mcp.tool()
    def test_app_availability() -> str:
        """Test which applications are actually available on this system."""
        return app_manager.test_app_availability()
    
    @mcp.tool()
    def search_app(query: str) -> str:
        """Search for apps by name or alias."""
        query_lower = query.lower()
        matches = []
        
        # Search in primary app names
        for app in app_manager.apps.keys():
            if query_lower in app:
                matches.append(f"{app} (primary)")
        
        # Search in aliases
        if "_reverse" in app_manager.aliases:
            for alias, primary in app_manager.aliases["_reverse"].items():
                if query_lower in alias and alias not in matches:
                    matches.append(f"{alias} → {primary}")
        
        if matches:
            return f"Found {len(matches)} matches for '{query}':\n" + "\n".join(matches)
        else:
            return f"No apps found matching '{query}'"
    
    @mcp.tool()
    def launch_by_path(executable_path: str, args: str = "") -> str:
        """Launch an application by its full path (with safety restrictions).
        
        Args:
            executable_path: Full path to the executable
            args: Optional command line arguments
        """
        # OS-specific safety checks
        if platform.system() == "Windows":
            allowed_dirs = [
                "C:\\Windows\\System32",
                "C:\\Windows",
                "C:\\Program Files",
                "C:\\Program Files (x86)",
                os.path.expanduser("~\\AppData")
            ]
            
            if not executable_path.lower().endswith('.exe'):
                return str("Only .exe files are allowed on Windows")
                
        elif platform.system() == "Darwin":  # macOS
            allowed_dirs = [
                "/Applications",
                "/System/Applications",
                "/usr/bin",
                "/usr/local/bin",
                os.path.expanduser("~/Applications")
            ]
            
        else:  # Linux
            allowed_dirs = [
                "/usr/bin",
                "/usr/local/bin",
                "/bin",
                "/snap/bin",
                "/opt",
                os.path.expanduser("~/.local/bin")
            ]
        
        if not any(executable_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            return str(f"Path not in allowed directories. Allowed: {', '.join(allowed_dirs)}")
        
        if not os.path.exists(executable_path):
            return str(f"Executable not found: {executable_path}")
        
        try:
            cmd = [executable_path]
            if args:
                cmd.extend(args.split())
            subprocess.Popen(cmd, shell=False)
            return str(f"Launched: {executable_path} {args}")
        except Exception as e:
            return str(f"Failed to launch: {str(e)}")
    
    @mcp.tool()
    def reload_apps_config(config_path: str = "apps_config.json") -> str:
        """Reload the apps configuration from file.
        
        Args:
            config_path: Path to the configuration file (default: apps_config.json)
        """
        app_manager.config_path = config_path
        app_manager.load_config()
        return f"Reloaded configuration. Loaded {len(app_manager.apps)} apps for {platform.system()}"
    
    @mcp.tool()
    def get_current_os() -> str:
        """Get information about the current operating system."""
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "apps_loaded": len(app_manager.apps)
        }
        return json.dumps(info, indent=2)