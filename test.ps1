# ChatOS PowerShell Test Launcher
# Automatically handles virtual environment setup and activation

param(
    [switch]$CreateVenv,
    [switch]$Force,
    [string]$VenvName = ".venv-host"
)

Write-Host "🧪 ChatOS PowerShell Test Launcher" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Change to script directory
Set-Location $PSScriptRoot

# Function to check if we're in a virtual environment
function Test-InVirtualEnv {
    return $env:VIRTUAL_ENV -ne $null
}

# Function to find existing virtual environments (prioritize .venv-host)
function Find-VirtualEnv {
    $venvPaths = @(".venv-host", ".venv", "venv", "chatos_env", "env")
    
    foreach ($path in $venvPaths) {
        $activateScript = Join-Path $path "Scripts\Activate.ps1"
        if (Test-Path $activateScript) {
            return $path
        }
    }
    return $null
}

# Function to test required packages
function Test-RequiredPackages {
    $packages = @("psutil", "dotenv", "pytest", "requests", "openai")
    $missing = @()
    
    foreach ($package in $packages) {
        try {
            $importName = if ($package -eq "dotenv") { "python-dotenv" } else { $package }
            python -c "import $package" 2>$null
            if ($LASTEXITCODE -ne 0) {
                $missing += $importName
            }
        }
        catch {
            $missing += $importName
        }
    }
    
    return $missing
}

# Main logic
try {
    # Check if already in virtual environment
    if (Test-InVirtualEnv) {
        Write-Host "✅ Already in virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
    }
    else {
        # Look for existing virtual environment
        $existingVenv = Find-VirtualEnv
        
        if ($existingVenv) {
            Write-Host "✅ Found virtual environment: $existingVenv" -ForegroundColor Green
            Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
            
            $activateScript = Join-Path $existingVenv "Scripts\Activate.ps1"
            & $activateScript
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Virtual environment activated successfully" -ForegroundColor Green
            }
            else {
                Write-Host "⚠️  Failed to activate virtual environment" -ForegroundColor Yellow
            }
        }
        elseif ($CreateVenv -or $Force) {
            Write-Host "🚀 Creating new virtual environment..." -ForegroundColor Yellow
            python -m venv $VenvName
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Virtual environment created successfully" -ForegroundColor Green
                $activateScript = Join-Path $VenvName "Scripts\Activate.ps1"
                & $activateScript
            }
            else {
                Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
                Write-Host "💡 Make sure Python is installed and in your PATH" -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "❌ No virtual environment found" -ForegroundColor Red
            $response = Read-Host "Create one now? (y/n)"
            
            if ($response -match "^[Yy]") {
                Write-Host "🚀 Creating virtual environment..." -ForegroundColor Yellow
                python -m venv $VenvName
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "✅ Virtual environment created" -ForegroundColor Green
                    $activateScript = Join-Path $VenvName "Scripts\Activate.ps1"
                    & $activateScript
                }
            }
            else {
                Write-Host "🐍 Using global Python environment..." -ForegroundColor Yellow
            }
        }
    }
    
    # Check and install packages
    Write-Host "🔍 Checking required packages..." -ForegroundColor Cyan
    $missingPackages = Test-RequiredPackages
    
    if ($missingPackages.Count -gt 0) {
        Write-Host "⚠️  Missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
        Write-Host "📦 Installing missing packages..." -ForegroundColor Yellow
        
        foreach ($package in $missingPackages) {
            Write-Host "   Installing $package..." -ForegroundColor Gray
            pip install $package
        }
        
        Write-Host "✅ Package installation completed" -ForegroundColor Green
    }
    else {
        Write-Host "✅ All required packages are available" -ForegroundColor Green
    }
    
    # Run tests
    Write-Host ""
    Write-Host "🚀 Running ChatOS tests..." -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Cyan
    
    python run_tests.py
    
    # Summary
    Write-Host ""
    Write-Host "====================================" -ForegroundColor Cyan
    Write-Host "🎯 Tests completed!" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "📊 Environment Summary:" -ForegroundColor Cyan
    if ($env:VIRTUAL_ENV) {
        Write-Host "   ✅ Virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
    }
    else {
        Write-Host "   🐍 Using global Python" -ForegroundColor Yellow
    }
    
    $pythonPath = python -c "import sys; print(sys.executable)" 2>$null
    if ($pythonPath) {
        Write-Host "   🐍 Python: $pythonPath" -ForegroundColor Gray
    }
}
catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Write-Host ""
    Write-Host "Press any key to continue..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}