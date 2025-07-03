set "CHATOS_DIR=%~dp0"
set "HOST_VENV=%CHATOS_DIR%.venv-host"

REM ── 3.  Activate the host venv in *this* window and run the UI ────────
call "%HOST_VENV%\Scripts\activate.bat"
python host\enhanced_chat_host.py

REM ── 4.  Keep the window open after exit so you can read logs ──────────
echo.
echo Chat host has terminated.  Press any key to close this window.
pause > nul
