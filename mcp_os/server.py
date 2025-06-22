from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent  # Import TextContent explicitly
import subprocess, os, pathlib, shutil
import winreg  # For registry lookups on Windows

mcp = FastMCP("local-os")

import fs_tools
fs_tools.register_fs_tools(mcp)

# Updated APPS dictionary with multiple potential paths
APPS = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe", "calculator.exe"],
    "calc": ["calc.exe", "calculator.exe"],
    "explorer": ["explorer.exe"],
    "file explorer": ["explorer.exe"],
    "cmd": ["cmd.exe"],
    "command prompt": ["cmd.exe"],
    "powershell": ["powershell.exe", "pwsh.exe"],  # pwsh for PowerShell 7+
    "paint": ["mspaint.exe"],
    "wordpad": ["wordpad.exe"],
    "task manager": ["taskmgr.exe"],
    "taskmgr": ["taskmgr.exe"],
    "control panel": ["control.exe"],
    "regedit": ["regedit.exe"],
    "registry editor": ["regedit.exe"],
    # Special protocols
    "settings": ["ms-settings:"],
    "windows settings": ["ms-settings:"],
    # Common third-party apps with multiple possible names/paths
    "chrome": [
        "chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ],
    "firefox": [
        "firefox.exe", 
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
    ],
    "edge": ["msedge.exe", "microsoftedge.exe"],
    "code": [
        "code.exe",
        r"C:\Users\{username}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
        r"C:\Program Files\Microsoft VS Code\Code.exe"
    ],
    "vscode": [
        "code.exe",
        r"C:\Users\{username}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
        r"C:\Program Files\Microsoft VS Code\Code.exe"
    ],
    "visual studio code": [
        "code.exe",
        r"C:\Users\{username}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
        r"C:\Program Files\Microsoft VS Code\Code.exe"
    ],
}

def find_executable(app_paths):
    """Find the first available executable from a list of possible paths."""
    username = os.getenv('USERNAME', '')
    
    for path in app_paths:
        # Handle special protocols
        if path.startswith("ms-"):
            return path
            
        # Replace {username} placeholder
        if '{username}' in path:
            path = path.replace('{username}', username)
            
        # First try to find in PATH
        if not os.path.sep in path:  # It's just an executable name
            if shutil.which(path):
                return path
        else:  # It's a full path
            if os.path.exists(path):
                return path
    
    return None

@mcp.tool()
def launch_app(app: str = "notepad") -> str:
    """Open an approved desktop application.
    
    Args:
        app: Name of the application to launch (e.g., 'notepad', 'calculator', 'chrome')
    """
    app_key = app.lower().strip()
    app_paths = APPS.get(app_key)
    
    if not app_paths:
        available = ", ".join(sorted(APPS.keys()))
        return f"'{app}' is not on the allow-list. Available apps: {available}"

    exe = find_executable(app_paths)
    if not exe:
        return f"Could not find executable for '{app}'. It may not be installed or not in expected locations."

    try:
        # Special handling for Windows Settings and other ms- protocols
        if exe.startswith("ms-"):
            os.system(f"start {exe}")
            result = f"Opened {app}."
        else:
            # Use shell=True to let Windows find the executable in PATH
            subprocess.Popen([exe], shell=True)
            result = f"Opened {app} using: {exe}"
        
        # Return just the string, FastMCP should handle the TextContent wrapping
        return result
    except Exception as e:
        return f"Failed to launch {app}: {str(e)}"

@mcp.tool()
def test_app_availability() -> str:
    """Test which applications are actually available on this system."""
    available = []
    unavailable = []
    
    for app_name, app_paths in APPS.items():
        exe = find_executable(app_paths)
        if exe:
            available.append(f"{app_name} -> {exe}")
        else:
            unavailable.append(app_name)
    
    result = "=== AVAILABLE APPS ===\n"
    result += "\n".join(available)
    result += "\n\n=== UNAVAILABLE APPS ===\n"
    result += ", ".join(unavailable)
    
    return str(result)

@mcp.tool()
def list_apps() -> str:
    """List all available applications that can be launched."""
    apps = sorted(APPS.keys())
    return str(f"Available applications: {', '.join(apps)}")

@mcp.tool()
def open_folder(path: str = None) -> str:
    """Open a folder in Windows Explorer.
    
    Args:
        path: Path to the folder to open (optional, defaults to user's home folder)
    """
    if not path:
        path = os.path.expanduser("~")  # Default to user's home directory

    try:
        # Expand environment variables and resolve path
        full_path = os.path.expandvars(os.path.expanduser(path))
        if os.path.exists(full_path):
            subprocess.Popen(f'explorer "{full_path}"', shell=True)
            return str(f"Opened folder: {full_path}")
        else:
            return str(f"Folder does not exist: {full_path}")
    except Exception as e:
        return str(f"Failed to open folder: {str(e)}")

@mcp.tool()
def launch_by_path(executable_path: str, args: str = "") -> str:
    """Launch an application by its full path (with safety restrictions).
    
    Args:
        executable_path: Full path to the executable
        args: Optional command line arguments
    """
    # Safety check - only allow .exe files in common directories
    allowed_dirs = [
        "C:\\Windows\\System32",
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        os.path.expanduser("~\\AppData")
    ]
    
    if not executable_path.lower().endswith('.exe'):
        return str("Only .exe files are allowed")
    
    if not any(executable_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
        return str(f"Path not in allowed directories. Allowed: {', '.join(allowed_dirs)}")
    
    if not os.path.exists(executable_path):
        return str(f"Executable not found: {executable_path}")
    
    try:
        cmd = [executable_path]
        if args:
            cmd.extend(args.split())
        subprocess.Popen(cmd, shell=True)
        return str(f"Launched: {executable_path} {args}")
    except Exception as e:
        return str(f"Failed to launch: {str(e)}")

if __name__ == "__main__":
    # prints a one-line JSON schema, then listens for tool calls
    mcp.run(transport="stdio")