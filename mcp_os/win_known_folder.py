import ctypes.wintypes as wt
import pathlib

def _win_known_folder(rfid: str) -> pathlib.Path | None:
    """Return a pathlib.Path to a Windows known folder (Desktop, Documents …)."""
    # list of GUIDs: https://learn.microsoft.com/windows/win32/shell/knownfolderid
    KNOWN_FOLDERS = {
        "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
        "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
        "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    }
    try:
        fid = KNOWN_FOLDERS[rfid.lower()]
    except KeyError:
        return None

    buf = wt.LPWSTR()
    if ctypes.windll.shell32.SHGetKnownFolderPath(
        ctypes.c_wchar_p(fid), 0, 0, ctypes.byref(buf)
    ) == 0:
        return pathlib.Path(buf.value)
    return None