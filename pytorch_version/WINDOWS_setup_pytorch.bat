@echo off
REM Windows Batch file to run PyTorch PINN setup
echo Running PyTorch PINN Setup for Windows...
echo.

REM Check if PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: PowerShell not found. Please install PowerShell.
    pause
    exit /b 1
)

REM Run the PowerShell setup script
powershell -ExecutionPolicy Bypass -File "setup_pytorch.ps1"

echo.
echo Setup complete! Press any key to exit...
pause >nul