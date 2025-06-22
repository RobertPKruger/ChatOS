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
python host\chat_host.py

REM ── 4.  Keep the window open after exit so you can read logs ──────────
echo.
echo Chat host has terminated.  Press any key to close this window.
pause > nul