# fs_tools.py
from mcp.server.fastmcp import FastMCP  # or just import the existing mcp object
from mcp.types import TextContent
import os, shutil, stat, pathlib, ctypes, uuid, ctypes.wintypes as wt

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

# ─────────────────────────────────────
# 3. Call from your existing server
# ─────────────────────────────────────
# in your existing server file, just add:
#     import fs_tools; fs_tools.register_fs_tools(mcp)
