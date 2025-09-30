@echo off
REM Windows Batch file to run PyTorch GPU PINN setup
echo Running PyTorch GPU PINN Setup for Windows...
echo This will create venv_pytorch_gpu in the current directory
echo.

REM Check if PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: PowerShell not found. Please install PowerShell.
    pause
    exit /b 1
)

REM Run the PowerShell GPU setup script
powershell -ExecutionPolicy Bypass -File "setup_pytorch_gpu.ps1"

echo.
echo Setup complete! The virtual environment is now in venv_pytorch_gpu\
echo To activate: .\venv_pytorch_gpu\Scripts\Activate.ps1
echo Press any key to exit...
pause >nul