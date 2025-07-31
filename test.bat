@echo off
REM Advanced ChatOS Test Launcher
REM Automatically sets up virtual environment if needed

setlocal enabledelayedexpansion

echo 🧪 ChatOS Advanced Test Launcher
echo ===================================

cd /d "%~dp0"

REM Function to check if we're already in a virtual environment
if defined VIRTUAL_ENV (
    echo ✅ Already in virtual environment: %VIRTUAL_ENV%
    goto :run_tests
)

REM Look for existing virtual environments (prioritize .venv-host)
echo 🔍 Checking for virtual environments...

if exist ".venv-host\Scripts\activate.bat" (
    echo ✅ Found .venv-host virtual environment (expected)
    call .venv-host\Scripts\activate.bat
    goto :check_packages
) else if exist ".venv\Scripts\activate.bat" (
    echo ✅ Found .venv virtual environment
    call .venv\Scripts\activate.bat
    goto :check_packages
) else if exist "venv\Scripts\activate.bat" (
    echo ✅ Found venv virtual environment
    call venv\Scripts\activate.bat
    goto :check_packages
) else if exist "chatos_env\Scripts\activate.bat" (
    echo ✅ Found chatos_env virtual environment
    call chatos_env\Scripts\activate.bat
    goto :check_packages
) else if exist "env\Scripts\activate.bat" (
    echo ✅ Found env virtual environment
    call env\Scripts\activate.bat
    goto :check_packages
) else (
    echo ❌ No virtual environment found
    echo 💡 Expected: .venv-host
    goto :create_venv
)

:create_venv
echo.
echo 🤔 No virtual environment found. Would you like to create one?
echo    This will ensure tests run with proper dependencies.
echo.
set /p "create_choice=Create virtual environment? (y/n): "

if /i "!create_choice!"=="y" (
    echo 🚀 Creating .venv-host virtual environment...
    python -m venv .venv-host
    
    if !errorlevel! neq 0 (
        echo ❌ Failed to create virtual environment
        echo 💡 Make sure Python is installed and in your PATH
        goto :global_fallback
    )
    
    echo ✅ Virtual environment created successfully
    echo 🔧 Activating .venv-host virtual environment...
    call .venv-host\Scripts\activate.bat
    
    echo 📦 Installing required packages...
    pip install psutil python-dotenv pytest requests openai sounddevice soundfile
    
    if !errorlevel! neq 0 (
        echo ⚠️  Some packages failed to install, but continuing...
    ) else (
        echo ✅ Packages installed successfully
    )
    
    goto :run_tests
) else (
    echo 🐍 Using global Python environment...
    goto :global_fallback
)

:check_packages
echo 🔍 Checking required packages...
python -c "import psutil, dotenv, pytest" 2>nul

if !errorlevel! neq 0 (
    echo ⚠️  Some required packages are missing
    echo 📦 Installing required packages...
    pip install psutil python-dotenv pytest requests openai sounddevice soundfile
    
    if !errorlevel! neq 0 (
        echo ⚠️  Package installation had issues, but continuing...
    )
)

goto :run_tests

:global_fallback
echo ⚠️  Using global Python - tests may fail if packages are missing
echo 💡 If tests fail, consider creating a virtual environment

:run_tests
echo.
echo 🚀 Running ChatOS tests...
echo ===================================
python run_tests.py

echo.
echo ===================================
echo 🎯 Tests completed!

REM Show environment info
echo.
echo 📊 Environment Summary:
if defined VIRTUAL_ENV (
    echo    ✅ Virtual environment: %VIRTUAL_ENV%
) else (
    echo    🐍 Using global Python
)

python -c "import sys; print(f'    🐍 Python: {sys.executable}')" 2>nul || echo    ❌ Python check failed

echo.
echo Press any key to close...
pause >nul