# Quick activation script for GPU environment (Windows)
Write-Host "🔥 Activating PyTorch GPU environment..." -ForegroundColor Red
& ".\venv_pytorch_gpu\Scripts\Activate.ps1"
Write-Host "✅ GPU environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Ready to run: python run_pinn_pytorch_gpu.py" -ForegroundColor Cyan
Write-Host "🔧 To deactivate: deactivate" -ForegroundColor Yellow