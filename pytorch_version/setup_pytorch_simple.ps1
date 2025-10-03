# PyTorch PINN Environment Setup Script (Windows PowerShell)
# Simple version without fancy formatting

Write-Host "Setting up PyTorch PINN Environment..."
Write-Host "======================================"

# Check Python
$pythonVersion = python --version 2>&1
Write-Host "Python found: $pythonVersion"

# Remove existing virtual environment if it exists
if (Test-Path "venv_pytorch") {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Recurse -Force "venv_pytorch"
}

# Create new virtual environment
Write-Host "Creating new virtual environment..."
python -m venv venv_pytorch

# Check if virtual environment was created
if (-not (Test-Path "venv_pytorch")) {
    Write-Host "ERROR: Virtual environment directory not created!"
    exit 1
}

# Check if activation script exists
if (-not (Test-Path "venv_pytorch\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment Python executable not found!"
    exit 1
}

Write-Host "Virtual environment created successfully"

# Install packages using the virtual environment's Python directly
Write-Host "Installing packages..."
& "venv_pytorch\Scripts\python.exe" -m pip install --upgrade pip
& "venv_pytorch\Scripts\python.exe" -m pip install -r requirements_pytorch.txt

Write-Host ""
Write-Host "Setup complete!"
Write-Host ""
Write-Host "To use this environment:"
Write-Host "   venv_pytorch\Scripts\Activate.ps1"
Write-Host "   python run_pinn_pytorch.py"
Write-Host ""
Write-Host "To deactivate:"
Write-Host "   deactivate"