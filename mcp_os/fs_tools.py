# fs_tools.py - Updated version with Steam integration

from mcp.server.fastmcp import FastMCP  # or just import the existing mcp object
from mcp.types import TextContent
import os, shutil, stat, pathlib, ctypes, uuid, ctypes.wintypes as wt
import json
from datetime import datetime
import subprocess
import winreg

# ─────────────────────────────────────
# 1. Helper: path resolution & checks
# ─────────────────────────────────────
BASE_DIR = pathlib.Path.home() / "OneDrive"        
SAFE_DRIVES = {BASE_DIR.drive}          # Only C: by default


desktop_path = BASE_DIR / "Desktop" 
documents_path = BASE_DIR / "Documents"


def safe_path(raw: str) -> pathlib.Path:
    # Strip whitespace / quotes
    raw = raw.strip().strip('"').strip("'")

    # ── Handle desktop alias ───────────────────────────
    if raw.lower() in {"desktop", "~/desktop", "%desktop%"}:
        return desktop_path

    # If the user passed something *under* the desktop
    if raw.lower().startswith(("desktop/", "~/desktop/")):
        sub = raw.split("/", 2)[-1]               # e.g. 'Projects/test'
        return desktop_path / sub
    
    if raw.lower() in {"documents", "~/documents", "%documents%"}:
        return documents_path
    
    if raw.lower().startswith(("documents/", "~/documents/")):
        sub = raw.split("/", 2)[-1]               # e.g. 'Projects/test'
        return documents_path / sub

    # Existing expansion & sandbox logic below …
    expanded = os.path.expandvars(os.path.expanduser(raw))
    p = pathlib.Path(expanded).resolve()

    # Reject drive changes or parent-escape
    if p.drive not in SAFE_DRIVES or BASE_DIR not in p.parents and p != BASE_DIR:
        raise ValueError(f"Path '{raw}' is outside the allowed area.")

    return p

def result(msg: str) -> str | TextContent:
    """Wrap once so you can switch to TextContent later if desired."""
    return msg

def format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def get_file_info(path: pathlib.Path) -> dict:
    """Get detailed file information"""
    try:
        stat_info = path.stat()
        return {
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": stat_info.st_size,
            "size_human": format_size(stat_info.st_size),
            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            "hidden": path.name.startswith('.') or (os.name == 'nt' and bool(stat_info.st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)),
            "readonly": os.name == 'nt' and bool(stat_info.st_file_attributes & stat.FILE_ATTRIBUTE_READONLY)
        }
    except Exception as e:
        return {
            "name": path.name,
            "type": "unknown",
            "error": str(e)
        }

# ─────────────────────────────────────
# Steam Integration
# ─────────────────────────────────────

def get_steam_path():
    """Get Steam installation path from Windows registry"""
    try:
        # Try to get Steam path from registry
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam")
        steam_path, _ = winreg.QueryValueEx(key, "SteamPath")
        winreg.CloseKey(key)
        return pathlib.Path(steam_path)
    except:
        # Fallback to common installation paths
        common_paths = [
            pathlib.Path("C:/Program Files (x86)/Steam"),
            pathlib.Path("C:/Program Files/Steam"),
            pathlib.Path("D:/Steam"),
            pathlib.Path("E:/Steam")
        ]
        for path in common_paths:
            if path.exists():
                return path
        return None

def get_steam_apps():
    """Get list of installed Steam games"""
    steam_path = get_steam_path()
    if not steam_path:
        return {}
    
    apps = {}
    
    # Parse Steam library folders
    library_folders_path = steam_path / "steamapps" / "libraryfolders.vdf"
    
    if library_folders_path.exists():
        try:
            # Simple VDF parser for library folders
            content = library_folders_path.read_text(encoding='utf-8')
            
            # Look for library paths
            library_paths = [steam_path / "steamapps"]
            
            # Parse additional library folders (simplified)
            import re
            path_pattern = r'"path"\s+"([^"]+)"'
            for match in re.finditer(path_pattern, content):
                lib_path = pathlib.Path(match.group(1).replace("\\\\", "/"))
                if lib_path.exists():
                    library_paths.append(lib_path / "steamapps")
            
            # Look for appmanifest files in each library
            for lib_path in library_paths:
                if not lib_path.exists():
                    continue
                    
                for manifest in lib_path.glob("appmanifest_*.acf"):
                    try:
                        # Parse ACF file (simple approach)
                        acf_content = manifest.read_text(encoding='utf-8')
                        
                        # Extract app ID
                        app_id_match = re.search(r'"appid"\s+"(\d+)"', acf_content)
                        name_match = re.search(r'"name"\s+"([^"]+)"', acf_content)
                        
                        if app_id_match and name_match:
                            app_id = app_id_match.group(1)
                            name = name_match.group(1)
                            apps[name.lower()] = {
                                "id": app_id,
                                "name": name,
                                "manifest": str(manifest)
                            }
                    except:
                        continue
                        
        except Exception as e:
            pass
    
    return apps

# ─────────────────────────────────────
# 2. File-system tools
# ─────────────────────────────────────
def register_fs_tools(mcp):

    @mcp.tool()
    def create_folder(path: str | None = None,
                    name: str | None = None) -> str:
        """Create a directory; if `name` is given it is appended to `path`."""
        try:
            # 1. Choose a base path
            if path is None:
                base = safe_path("~")                # fallback to home
            else:
                base = safe_path(path)

            # 2. Add the optional sub-folder
            target = base / name if name else base

            target.mkdir(parents=True, exist_ok=True)
            return f"__OK__ Folder created at {target}"
        except Exception as e:
            return f"Could not create folder: {e}"

    @mcp.tool()
    def create_file(path: str, content: str = "") -> str:
        """Create (or overwrite) a text file with optional content."""
        try:
            p = safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return result(f"File written to: {p}")
        except Exception as e:
            return result(f"Could not write file: {e}")

    @mcp.tool()
    def append_file(path: str, content: str) -> str:
        """Append text to a file (creates it if absent)."""
        try:
            p = safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(content + "\n")
            return result(f"Appended to {p}")
        except Exception as e:
            return result(f"Could not append: {e}")

    @mcp.tool()
    def read_file(path: str, 
                  encoding: str = "utf-8",
                  max_size_mb: float = 10.0) -> str:
        """Read a text file's contents. 
        
        Args:
            path: Path to the file
            encoding: Text encoding (default: utf-8)
            max_size_mb: Maximum file size in MB to read (default: 10MB)
        """
        try:
            p = safe_path(path)
            
            if not p.exists():
                return f"File not found: {p}"
            
            if not p.is_file():
                return f"Not a file: {p}"
            
            # Check file size
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                return f"File too large ({size_mb:.1f}MB). Maximum allowed: {max_size_mb}MB"
            
            # Try to read as text
            try:
                content = p.read_text(encoding=encoding)
                lines = content.splitlines()
                
                # Provide summary for large files
                if len(lines) > 100:
                    preview = "\n".join(lines[:50])
                    return f"File: {p}\nSize: {format_size(p.stat().st_size)}\nLines: {len(lines)}\n\nFirst 50 lines:\n{preview}\n\n[... {len(lines) - 50} more lines ...]"
                else:
                    return f"File: {p}\nSize: {format_size(p.stat().st_size)}\n\nContent:\n{content}"
                    
            except UnicodeDecodeError:
                # Try common encodings
                for enc in ['latin-1', 'cp1252', 'utf-16']:
                    try:
                        content = p.read_text(encoding=enc)
                        return f"File: {p} (encoding: {enc})\n\nContent:\n{content}"
                    except:
                        continue
                
                return f"Could not decode file {p} as text. It might be a binary file."
                
        except Exception as e:
            return f"Could not read file: {e}"

    @mcp.tool()
    def list_files(path: str = "~",
                   pattern: str = "*",
                   recursive: bool = False,
                   include_hidden: bool = False,
                   sort_by: str = "name",
                   limit: int = 100) -> str:
        """List files and folders in a directory.
        
        Args:
            path: Directory path (default: home)
            pattern: Glob pattern for filtering (e.g., "*.txt")
            recursive: Search subdirectories
            include_hidden: Include hidden files
            sort_by: Sort by 'name', 'size', 'modified', or 'type'
            limit: Maximum number of items to return
        """
        try:
            p = safe_path(path)
            
            if not p.exists():
                return f"Path not found: {p}"
            
            if not p.is_dir():
                # If it's a file, show info about that file
                info = get_file_info(p)
                return f"Single file:\n{json.dumps(info, indent=2)}"
            
            # Collect items
            items = []
            
            if recursive:
                # Use rglob for recursive search
                paths = p.rglob(pattern)
            else:
                # Use glob for non-recursive search
                paths = p.glob(pattern)
            
            for item_path in paths:
                # Skip hidden files if not requested
                if not include_hidden and item_path.name.startswith('.'):
                    continue
                
                info = get_file_info(item_path)
                info['path'] = str(item_path.relative_to(p))
                items.append(info)
                
                if len(items) >= limit:
                    break
            
            # Sort items
            if sort_by == 'size':
                items.sort(key=lambda x: x.get('size', 0), reverse=True)
            elif sort_by == 'modified':
                items.sort(key=lambda x: x.get('modified', ''), reverse=True)
            elif sort_by == 'type':
                items.sort(key=lambda x: (x.get('type', ''), x.get('name', '')))
            else:  # name
                items.sort(key=lambda x: x.get('name', '').lower())
            
            # Format output
            output = [f"Directory: {p}"]
            output.append(f"Items found: {len(items)}")
            
            if items:
                output.append("\n" + "─" * 60)
                
                # Group by type
                dirs = [i for i in items if i.get('type') == 'directory']
                files = [i for i in items if i.get('type') == 'file']
                
                if dirs:
                    output.append(f"\nDirectories ({len(dirs)}):")
                    for d in dirs:
                        output.append(f"  📁 {d['name']}/")
                
                if files:
                    output.append(f"\nFiles ({len(files)}):")
                    for f in files:
                        size_str = f.get('size_human', 'unknown')
                        output.append(f"  📄 {f['name']} ({size_str})")
                
                # Summary statistics
                total_size = sum(i.get('size', 0) for i in files)
                if total_size > 0:
                    output.append(f"\nTotal size: {format_size(total_size)}")
            else:
                output.append("\nNo items found matching the criteria.")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Could not list files: {e}"

    @mcp.tool()
    def search_files(pattern: str,
                     path: str = "~",
                     content: str | None = None,
                     file_type: str | None = None,
                     max_results: int = 50) -> str:
        """Search for files by name pattern and optionally by content.
        
        Args:
            pattern: File name pattern (e.g., "*.py" or "report*")
            path: Starting directory for search
            content: Optional text to search for within files
            file_type: Filter by type: 'text', 'image', 'doc', etc.
            max_results: Maximum number of results to return
        """
        try:
            p = safe_path(path)
            
            if not p.exists() or not p.is_dir():
                return f"Invalid search path: {p}"
            
            # Define file type extensions
            type_extensions = {
                'text': ['.txt', '.md', '.log', '.csv', '.json', '.xml'],
                'code': ['.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go'],
                'doc': ['.doc', '.docx', '.pdf', '.odt', '.rtf'],
                'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
                'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv'],
                'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
            }
            
            results = []
            
            for item_path in p.rglob(pattern):
                if len(results) >= max_results:
                    break
                
                if not item_path.is_file():
                    continue
                
                # Check file type filter
                if file_type and file_type in type_extensions:
                    if not any(item_path.suffix.lower() == ext for ext in type_extensions[file_type]):
                        continue
                
                file_info = get_file_info(item_path)
                file_info['path'] = str(item_path)
                
                # Search content if requested
                if content:
                    try:
                        # Only search in text files
                        if item_path.suffix.lower() in ['.txt', '.md', '.log', '.csv', '.json', '.xml', '.py', '.js', '.java', '.cpp', '.c', '.h']:
                            file_content = item_path.read_text(encoding='utf-8', errors='ignore')
                            if content.lower() in file_content.lower():
                                # Find line with match
                                lines = file_content.splitlines()
                                for i, line in enumerate(lines):
                                    if content.lower() in line.lower():
                                        file_info['match'] = f"Line {i+1}: {line.strip()[:100]}..."
                                        break
                                results.append(file_info)
                    except:
                        pass
                else:
                    results.append(file_info)
            
            # Format output
            output = [f"Search results for '{pattern}' in {p}"]
            if content:
                output.append(f"Containing text: '{content}'")
            output.append(f"Found {len(results)} matches\n")
            
            for r in results:
                output.append(f"📄 {r['path']}")
                output.append(f"   Size: {r['size_human']} | Modified: {r['modified'][:10]}")
                if 'match' in r:
                    output.append(f"   Match: {r['match']}")
                output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Search failed: {e}"

    @mcp.tool()
    def file_info(path: str) -> str:
        """Get detailed information about a file or directory."""
        try:
            p = safe_path(path)
            
            if not p.exists():
                return f"Path not found: {p}"
            
            info = get_file_info(p)
            
            # Add additional details
            output = [f"Information for: {p}"]
            output.append("─" * 50)
            output.append(f"Name: {info['name']}")
            output.append(f"Type: {info['type']}")
            
            if info['type'] == 'file':
                output.append(f"Size: {info['size_human']} ({info['size']:,} bytes)")
                output.append(f"Extension: {p.suffix or 'none'}")
            elif info['type'] == 'directory':
                try:
                    # Count items in directory
                    items = list(p.iterdir())
                    dirs = sum(1 for i in items if i.is_dir())
                    files = sum(1 for i in items if i.is_file())
                    output.append(f"Contains: {dirs} directories, {files} files")
                except:
                    pass
            
            output.append(f"Created: {info['created']}")
            output.append(f"Modified: {info['modified']}")
            
            if os.name == 'nt':
                output.append(f"Hidden: {info.get('hidden', False)}")
                output.append(f"Read-only: {info.get('readonly', False)}")
            
            # Add parent directory
            output.append(f"Parent: {p.parent}")
            
            # Add full path
            output.append(f"Full path: {p.absolute()}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Could not get file info: {e}"

    @mcp.tool()
    def move(path: str, destination: str) -> str:
        """Move/rename a file or folder."""
        try:
            src = safe_path(path)
            dst = safe_path(destination)
            shutil.move(src, dst)
            return result(f"Moved {src} → {dst}")
        except Exception as e:
            return result(f"Could not move: {e}")

    @mcp.tool()
    def delete(path: str) -> str:
        """Delete a file or folder (recursively for folders)."""
        try:
            p = safe_path(path)
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
            return result(f"Deleted {p}")
        except Exception as e:
            return result(f"Could not delete: {e}")

    @mcp.tool()
    def set_attributes(path: str,
                       read_only: bool | None = None,
                       hidden: bool | None = None) -> str:
        """Update Windows file attributes."""
        try:
            p = safe_path(path)
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(p))
            if attrs == -1:
                raise FileNotFoundError

            if read_only is not None:
                if read_only:
                    attrs |= stat.FILE_ATTRIBUTE_READONLY
                else:
                    attrs &= ~stat.FILE_ATTRIBUTE_READONLY
            if hidden is not None:
                if hidden:
                    attrs |= stat.FILE_ATTRIBUTE_HIDDEN
                else:
                    attrs &= ~stat.FILE_ATTRIBUTE_HIDDEN

            ctypes.windll.kernel32.SetFileAttributesW(str(p), attrs)
            return result(f"Attributes updated for {p}")
        except Exception as e:
            return result(f"Could not update attributes: {e}")

    @mcp.tool()
    def open_folder(path: str = "~") -> str:
        """Open a folder in Windows Explorer."""
        try:
            p = safe_path(path)
            
            if not p.exists():
                return f"Path not found: {p}"
                
            # Ensure it's a directory or get parent if it's a file
            if p.is_file():
                p = p.parent
                
            # Use Windows Explorer
            os.startfile(str(p))
            return f"Opened folder in Explorer: {p}"
            
        except Exception as e:
            return f"Could not open folder: {e}"
    
    # ─────────────────────────────────────
    # Steam Tools
    # ─────────────────────────────────────
    
    @mcp.tool()
    def launch_steam_game(game_name: str = None, app_id: str = None) -> str:
        """Launch a Steam game by name or app ID.
        
        Args:
            game_name: Name of the game (e.g., "Counter-Strike 2", "Dota 2")
            app_id: Steam App ID (e.g., "730" for CS:GO)
        """
        try:
            steam_path = get_steam_path()
            if not steam_path:
                return "Steam installation not found. Please ensure Steam is installed."
            
            steam_exe = steam_path / "steam.exe"
            if not steam_exe.exists():
                return f"Steam executable not found at {steam_exe}"
            
            # If app_id provided, use it directly
            if app_id:
                # Launch using steam://rungameid/appid protocol
                subprocess.Popen([str(steam_exe), f"steam://rungameid/{app_id}"])
                return f"Launching Steam game with App ID: {app_id}"
            
            # If game name provided, try to find it
            elif game_name:
                # Get installed games
                games = get_steam_apps()
                
                # Normalize the search name
                search_name = game_name.lower().strip()
                
                # First try exact match
                if search_name in games:
                    app_id = games[search_name]["id"]
                    subprocess.Popen([str(steam_exe), f"steam://rungameid/{app_id}"])
                    return f"Launching {games[search_name]['name']} (App ID: {app_id})"
                
                # Try partial match
                matches = []
                for game_key, game_info in games.items():
                    if search_name in game_key or game_key in search_name:
                        matches.append(game_info)
                
                if len(matches) == 1:
                    app_id = matches[0]["id"]
                    subprocess.Popen([str(steam_exe), f"steam://rungameid/{app_id}"])
                    return f"Launching {matches[0]['name']} (App ID: {app_id})"
                elif len(matches) > 1:
                    game_list = "\n".join([f"  - {m['name']} (ID: {m['id']})" for m in matches[:10]])
                    return f"Multiple games found matching '{game_name}':\n{game_list}\n\nPlease be more specific or use the App ID."
                else:
                    # Provide some popular games as examples
                    return f"Game '{game_name}' not found in your Steam library.\n\nTry using the exact name or App ID. Some examples:\n  - 'Counter-Strike 2' or app_id='730'\n  - 'Dota 2' or app_id='570'\n  - 'Team Fortress 2' or app_id='440'"
            
            else:
                return "Please provide either a game name or app ID"
                
        except Exception as e:
            return f"Error launching Steam game: {e}"
    
    @mcp.tool()
    def list_steam_games(search: str = None, limit: int = 20) -> str:
        """List installed Steam games.
        
        Args:
            search: Optional search term to filter games
            limit: Maximum number of games to return
        """
        try:
            games = get_steam_apps()
            
            if not games:
                return "No Steam games found. Make sure Steam is installed and you have games in your library."
            
            # Filter by search term if provided
            if search:
                search_lower = search.lower()
                filtered = {k: v for k, v in games.items() if search_lower in k}
                games = filtered
            
            # Sort by name
            sorted_games = sorted(games.items(), key=lambda x: x[1]['name'])[:limit]
            
            output = ["Installed Steam Games:"]
            output.append("─" * 50)
            
            for _, game_info in sorted_games:
                output.append(f"🎮 {game_info['name']}")
                output.append(f"   App ID: {game_info['id']}")
                output.append("")
            
            if len(games) > limit:
                output.append(f"... and {len(games) - limit} more games")
            
            output.append("\nUse 'launch_steam_game' with the game name or App ID to launch.")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error listing Steam games: {e}"
    
    @mcp.tool()
    def open_steam(page: str = None) -> str:
        """Open Steam client or a specific Steam page.
        
        Args:
            page: Optional page to open ('store', 'library', 'community', 'profile', 'settings')
        """
        try:
            steam_path = get_steam_path()
            if not steam_path:
                return "Steam installation not found."
            
            steam_exe = steam_path / "steam.exe"
            
            if page:
                page_lower = page.lower()
                page_urls = {
                    'store': 'steam://store',
                    'library': 'steam://open/games',
                    'community': 'steam://open/community',
                    'profile': 'steam://open/profile',
                    'settings': 'steam://open/settings',
                    'downloads': 'steam://open/downloads',
                    'friends': 'steam://open/friends',
                    'screenshots': 'steam://open/screenshots',
                    'servers': 'steam://open/servers',
                    'news': 'steam://open/news'
                }
                
                if page_lower in page_urls:
                    subprocess.Popen([str(steam_exe), page_urls[page_lower]])
                    return f"Opening Steam {page} page"
                else:
                    return f"Unknown Steam page '{page}'. Available pages: {', '.join(page_urls.keys())}"
            else:
                # Just open Steam
                subprocess.Popen([str(steam_exe)])
                return "Opening Steam client"
                
        except Exception as e:
            return f"Error opening Steam: {e}"