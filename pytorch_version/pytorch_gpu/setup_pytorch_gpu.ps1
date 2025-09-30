# PyTorch GPU PINN Environment Setup Script (Windows PowerShell)
# This script creates a fresh virtual environment for the PyTorch GPU implementation

# Enable strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up PyTorch GPU PINN Environment..." -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

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

# Check for NVIDIA GPU
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($gpuInfo) {
        Write-Host "🎮 NVIDIA GPU detected: $gpuInfo" -ForegroundColor Cyan
        Write-Host "✅ GPU setup will proceed with CUDA support" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Warning: nvidia-smi not found. No NVIDIA GPU detected." -ForegroundColor Red
    Write-Host ""
    Write-Host "🚨 GPU Requirements Not Met!" -ForegroundColor Red
    Write-Host "This setup installs PyTorch with CUDA support but your system may not have:" -ForegroundColor Yellow
    Write-Host "  • NVIDIA GPU with CUDA support" -ForegroundColor Gray
    Write-Host "  • NVIDIA drivers installed" -ForegroundColor Gray
    Write-Host ""
    Write-Host "💡 Consider using the CPU version instead:" -ForegroundColor Cyan
    Write-Host "   cd ..\  # Go back to pytorch_version directory" -ForegroundColor Gray
    Write-Host "   .\setup_pytorch.ps1" -ForegroundColor Gray
    Write-Host "   python run_pinn_pytorch.py" -ForegroundColor Gray
    Write-Host ""
    
    $continue = Read-Host "Continue with GPU setup anyway? [y/N]"
    if ($continue -notmatch "^[Yy]$") {
        Write-Host "👋 Exiting. Please try the CPU version setup." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "⚠️  Continuing with GPU setup (will fall back to CPU during training)..." -ForegroundColor Yellow
}

# Remove existing virtual environment if it exists
if (Test-Path "venv_pytorch_gpu") {
    Write-Host "🗑️  Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv_pytorch_gpu"
}

# Create new virtual environment
Write-Host "📦 Creating new virtual environment..." -ForegroundColor Cyan
python -m venv venv_pytorch_gpu

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Cyan
& ".\venv_pytorch_gpu\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
Write-Host "🔥 Installing PyTorch with CUDA support..." -ForegroundColor Red
Write-Host "   This may take a few minutes..." -ForegroundColor Gray

# Install PyTorch GPU version (CUDA 12.1 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
Write-Host "📥 Installing additional requirements..." -ForegroundColor Cyan
pip install -r requirements_pytorch_gpu.txt

Write-Host ""
Write-Host "✅ PyTorch GPU PINN environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 To use this environment:" -ForegroundColor White
Write-Host "   .\venv_pytorch_gpu\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "   python run_pinn_pytorch_gpu.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "🔧 To deactivate:" -ForegroundColor White
Write-Host "   deactivate" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 GPU Setup Verification:" -ForegroundColor Cyan
Write-Host "   After activation, run: python -c `"import torch; print(f'CUDA available: {torch.cuda.is_available()}')`"" -ForegroundColor Gray
Write-Host ""
Write-Host "🎮 GPU Notes:" -ForegroundColor White
Write-Host "   • Requires NVIDIA GPU with CUDA 12.1+ support" -ForegroundColor Gray
Write-Host "   • Uses float32 for optimal GPU performance" -ForegroundColor Gray
Write-Host "   • Includes GPU memory monitoring during training" -ForegroundColor Gray
Write-Host "   • Falls back to CPU if CUDA unavailable" -ForegroundColor Gray
Write-Host ""