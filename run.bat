@echo off
:: ────────────────────────────────────────────────
:: ChatOS launch script – Windows 10 / 11
:: place this file in ...\ChatOS\start_chat_os.bat
:: ────────────────────────────────────────────────

REM ── 1.  Define absolute paths (edit if you moved folders) ──────────────
set "CHATOS_DIR=%~dp0"
set "SERVER_VENV=%CHATOS_DIR%.venv-server"
set "HOST_VENV=%CHATOS_DIR%.venv-host"

REM ── 2.  Launch the MCP server in a new window ─────────────────────────
start "ChatOS-Server" ^
cmd /k ^
"^
    cd /d "%CHATOS_DIR%" ^& ^
    call "%SERVER_VENV%\Scripts\activate.bat" ^& ^
    python mcp_os\server.py ^
"

REM ── 3.  Activate the host venv in *this* window and run the UI ────────
call "%HOST_VENV%\Scripts\activate.bat"
python host\enhanced_chat_host.py

REM ── 4.  Keep the window open after exit so you can read logs ──────────
echo.
echo Chat host has terminated.  Press any key to close this window.
pause > nul