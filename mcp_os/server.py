from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent  # Import TextContent explicitly
import subprocess, os, pathlib, shutil
import winreg  # For registry lookups on Windows
import logging

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

    # For Office apps, we'll use a different approach
    "word": [
        r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        "winword.exe",
        "start winword"
    ],
    "microsoft word": [
        r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        "winword.exe",
        "start winword"
    ],
    "winword": [
        r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        "winword.exe",
        "start winword"
    ],
    "excel": [
        r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        "excel.exe",
        "start excel"
    ],
    "microsoft excel": [
        r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        "excel.exe",
        "start excel"
    ],
    "powerpoint": [
        r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
        "powerpnt.exe",
        "start powerpnt"
    ],
    "outlook": [
        r"C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE",
        "outlook.exe",
        "start outlook"
    ],

    # Common third-party apps with multiple possible names/paths
    "chrome": [
        "chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "start chrome"
    ],
    "firefox": [
        "firefox.exe", 
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
    ],
    "edge": ["msedge.exe", "microsoftedge.exe", "start msedge"],
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
def launch_app(app_name: str) -> str:
    """Attempts to launch the given application or protocol."""
    key = app_name.lower()
    executables = APPS.get(key)
    if not executables:
        return f"Application '{app_name}' not found."

    username = os.getenv('USERNAME', '')
    errors = []
    
    for exe in executables:
        try:
            # Expand potential placeholders
            path = exe.replace('{username}', username)
            
            # Protocol handlers (e.g. ms-settings:)
            if path.endswith(":"):
                os.startfile(path)
                return f"Launched '{app_name}'"
                
            # Handle 'start' commands (for Office apps)
            elif path.startswith("start "):
                # Use shell=True with string command for 'start' commands
                subprocess.Popen(path, shell=True)
                return f"Launched '{app_name}'"
                
            # Absolute path - check if it exists first
            elif os.path.isabs(path):
                if os.path.exists(path):
                    # For absolute paths, use os.startfile on Windows
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
        subprocess.Popen(cmd, shell=False)
        return str(f"Launched: {executable_path} {args}")
    except Exception as e:
        return str(f"Failed to launch: {str(e)}")

@mcp.tool()
def debug_office_launch() -> str:
    """Debug Office launch issues by checking paths and trying different methods."""
    office_path = r"C:\Program Files\Microsoft Office\root\Office16"
    results = []
    
    # Check if the Office directory exists
    if os.path.exists(office_path):
        results.append(f"✓ Office directory exists: {office_path}")
        
        # Check for specific executables
        office_apps = {
            "WINWORD.EXE": "Word",
            "EXCEL.EXE": "Excel", 
            "POWERPNT.EXE": "PowerPoint",
            "OUTLOOK.EXE": "Outlook"
        }
        
        for exe_name, app_name in office_apps.items():
            exe_path = os.path.join(office_path, exe_name)
            if os.path.exists(exe_path):
                results.append(f"✓ {app_name} found: {exe_path}")
                
                # Try different launch methods
                try:
                    # Method 1: os.startfile
                    os.startfile(exe_path)
                    results.append(f"  → Successfully launched {app_name} with os.startfile")
                except Exception as e:
                    results.append(f"  → os.startfile failed: {e}")
                    
                    try:
                        # Method 2: subprocess with shell=False
                        subprocess.Popen([exe_path])
                        results.append(f"  → Successfully launched {app_name} with subprocess")
                    except Exception as e2:
                        results.append(f"  → subprocess failed: {e2}")
            else:
                results.append(f"✗ {app_name} NOT found: {exe_path}")
    else:
        results.append(f"✗ Office directory NOT found: {office_path}")
        
    return "\n".join(results)

if __name__ == "__main__":
    # prints a one-line JSON schema, then listens for tool calls
    mcp.run(transport="stdio")