from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import subprocess
import os
import pathlib
import shutil
import platform
import json
import logging

mcp = FastMCP("local-os")

import fs_tools
fs_tools.register_fs_tools(mcp)

# Global variables to store OS-specific apps and aliases
APPS = {}
ALIASES = {}

def load_apps_config(config_path="apps_config.json"):
    """Load OS-specific application configurations from JSON file."""
    global APPS, ALIASES
    
    # Determine the current OS
    system = platform.system().lower()
    
    # Map platform.system() values to our config keys
    os_map = {
        "windows": "windows",
        "darwin": "darwin",  # macOS
        "linux": "linux"
    }
    
    current_os = os_map.get(system, "linux")  # Default to linux if unknown
    
    try:
        # Try to load from the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, config_path)
        
        if not os.path.exists(config_file):
            # Try current working directory
            config_file = config_path
            
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        if current_os in config:
            APPS = config[current_os]
            logging.info(f"Loaded {len(APPS)} apps for {current_os}")
        else:
            logging.warning(f"No configuration found for OS: {current_os}")
            APPS = {}
            
        # Load aliases if available
        if "aliases" in config and current_os in config["aliases"]:
            ALIASES = config["aliases"][current_os]
            # Build reverse mapping for quick lookup
            ALIASES_REVERSE = {}
            for primary_name, alias_list in ALIASES.items():
                for alias in alias_list:
                    ALIASES_REVERSE[alias.lower()] = primary_name
            ALIASES["_reverse"] = ALIASES_REVERSE
            logging.info(f"Loaded {len(ALIASES)-1} alias groups for {current_os}")
            
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        APPS = {}
        ALIASES = {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in configuration file: {e}")
        APPS = {}
        ALIASES = {}
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        APPS = {}
        ALIASES = {}

# Load apps configuration on startup
load_apps_config()

def find_executable(app_paths):
    """Find the first available executable from a list of possible paths."""
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

def resolve_app_name(app_name: str) -> str:
    """Resolve an app name or alias to the primary app name."""
    lower_name = app_name.lower()
    
    # First check if it's already a primary app name
    if lower_name in APPS:
        return lower_name
    
    # Check if it's an alias
    if "_reverse" in ALIASES and lower_name in ALIASES["_reverse"]:
        return ALIASES["_reverse"][lower_name]
    
    # Return original if no match found
    return lower_name

@mcp.tool()
def launch_app(app_name: str) -> str:
    """Attempts to launch the given application or protocol."""
    # Resolve any aliases to the primary app name
    resolved_name = resolve_app_name(app_name)
    
    executables = APPS.get(resolved_name)
    if not executables:
        # Try to find similar apps for helpful error message
        similar = find_similar_apps(app_name)
        if similar:
            return f"Application '{app_name}' not found. Did you mean: {', '.join(similar)}?"
        return f"Application '{app_name}' not found. Use 'list_apps' to see available apps."

    username = os.getenv('USERNAME', '') if platform.system() == "Windows" else os.getenv('USER', '')
    errors = []
    
    # Handle macOS 'open' command format
    if platform.system() == "Darwin" and isinstance(executables, list) and len(executables) > 1 and executables[0] == "open":
        try:
            subprocess.Popen(executables)
            return f"Launched '{app_name}'"
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
                    subprocess.Popen(path, shell=True)
                    return f"Launched '{app_name}'"
                    
                # Absolute path - check if it exists first
                elif os.path.isabs(path):
                    if os.path.exists(path):
                        os.startfile(path)
                        return f"Launched '{app_name}' from {path}"
                    else:
                        errors.append(f"Path not found: {path}")
                        continue
                        
                # Try to find in PATH
                elif shutil.which(path):
                    subprocess.Popen([path], shell=False)
                    return f"Launched '{app_name}'"
                else:
                    errors.append(f"Not found in PATH: {path}")
                    
            # Linux/Unix handling
            else:
                if shutil.which(path) or os.path.exists(path):
                    subprocess.Popen([path], shell=False)
                    return f"Launched '{app_name}'"
                else:
                    errors.append(f"Not found: {path}")
                    
        except Exception as e:
            errors.append(f"Failed to launch '{exe}': {str(e)}")
            continue

    return f"Failed to launch '{app_name}'. Errors: {'; '.join(errors)}"

@mcp.tool()
def test_app_availability() -> str:
    """Test which applications are actually available on this system."""
    available = []
    unavailable = []
    
    for app_name, app_paths in APPS.items():
        exe = find_executable(app_paths)
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
    if ALIASES and "_reverse" in ALIASES:
        result += f"\n\n=== LOADED ALIASES ===\n"
        result += f"Total aliases: {len(ALIASES['_reverse'])}"
    
    return str(result)

def find_similar_apps(app_name: str, threshold: float = 0.6) -> list:
    """Find apps with similar names using simple string matching."""
    similar = []
    lower_name = app_name.lower()
    
    # Check all primary app names and aliases
    all_names = list(APPS.keys())
    if "_reverse" in ALIASES:
        all_names.extend(ALIASES["_reverse"].keys())
    
    for name in all_names:
        # Simple substring matching
        if lower_name in name or name in lower_name:
            similar.append(name)
        # Check if they share significant characters
        elif len(set(lower_name) & set(name)) > len(lower_name) * threshold:
            similar.append(name)
    
    return list(set(similar))[:5]  # Return up to 5 suggestions

@mcp.tool()
def list_apps() -> str:
    """List all available applications and their aliases."""
    apps = sorted(APPS.keys())
    result = f"Available applications on {platform.system()}:\n\n"
    
    if ALIASES and "_reverse" in ALIASES:
        # Group by primary app with aliases
        for app in apps:
            aliases = []
            for alias, primary in ALIASES["_reverse"].items():
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

@mcp.tool()
def search_app(query: str) -> str:
    """Search for apps by name or alias."""
    query_lower = query.lower()
    matches = []
    
    # Search in primary app names
    for app in APPS.keys():
        if query_lower in app:
            matches.append(f"{app} (primary)")
    
    # Search in aliases
    if "_reverse" in ALIASES:
        for alias, primary in ALIASES["_reverse"].items():
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
    load_apps_config(config_path)
    return f"Reloaded configuration. Loaded {len(APPS)} apps for {platform.system()}"

@mcp.tool()
def get_current_os() -> str:
    """Get information about the current operating system."""
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "apps_loaded": len(APPS)
    }
    return json.dumps(info, indent=2)

# Remove Windows-specific debug function and import
# (winreg import and debug_office_launch function are no longer needed)

if __name__ == "__main__":
    # prints a one-line JSON schema, then listens for tool calls
    mcp.run(transport="stdio")