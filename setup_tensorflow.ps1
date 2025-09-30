# TensorFlow PINN Environment Setup Script (Windows PowerShell)
# This script creates a fresh virtual environment for the TensorFlow implementation

# Enable strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up TensorFlow PINN Environment..." -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Check if Python 3.11 is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    
    # Check if it's Python 3.11+
    if (-not ($pythonVersion -match "Python 3\.1[1-9]")) {
        Write-Host "⚠️  Warning: Python 3.11+ is recommended. Found: $pythonVersion" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Error: Python is required but not found." -ForegroundColor Red
    Write-Host "Please install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Remove existing virtual environment if it exists
if (Test-Path "venv") {
    Write-Host "🗑️  Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

# Create new virtual environment
Write-Host "📦 Creating new virtual environment..." -ForegroundColor Cyan
python -m venv venv

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
Write-Host "📥 Installing TensorFlow PINN requirements..." -ForegroundColor Cyan
pip install -r requirements_tensorflow.txt

Write-Host ""
Write-Host "✅ TensorFlow PINN environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 To use this environment:" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "   python run_pinn_stack.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "🔧 To deactivate:" -ForegroundColor White
Write-Host "   deactivate" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 Note: You may need to enable script execution:" -ForegroundColor Cyan
Write-Host "   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Gray
Write-Host ""