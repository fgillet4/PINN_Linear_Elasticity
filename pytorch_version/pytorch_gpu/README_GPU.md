# PyTorch GPU PINN Setup

High-performance GPU-accelerated Physics-Informed Neural Network implementation for linear elasticity problems.

## 🚀 Quick Start

### Linux/macOS:
```bash
./setup_pytorch_gpu.sh
python run_pinn_pytorch_gpu.py
```

### Windows:
```cmd
WINDOWS_setup_pytorch_gpu.bat
```

## 🎮 GPU Requirements

- **NVIDIA GPU** with CUDA Compute Capability 3.5+
- **CUDA 12.1+** drivers installed
- **8GB+ GPU memory** recommended for larger problems
- **NVIDIA drivers** version 525.60.11 or newer

## 📊 Performance Benefits

Compared to CPU version:
- **5-10x faster training** on modern GPUs
- **Real-time loss monitoring** with GPU memory usage
- **Larger problem scaling** capability
- **float32 precision** for optimal GPU performance

## 🔧 Manual Setup

If you prefer manual installation:

```bash
# Create environment
python3.11 -m venv venv_pytorch_gpu
source venv_pytorch_gpu/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements_pytorch_gpu.txt

# Run GPU PINN
python run_pinn_pytorch_gpu.py
```

## 🧠 Model Features

- **Automatic GPU detection** and fallback to CPU
- **GPU memory monitoring** during training
- **Optimized tensor operations** for CUDA
- **Identical results** to CPU version
- **Enhanced progress reporting** with GPU metrics

## 📈 Output Files

GPU version generates:
- `solution_pytorch_gpu.png` - Displacement fields
- `stress_map_pytorch_gpu.png` - Stress fields
- `error_map_pytorch_gpu.png` - Error analysis
- `loss_history_pytorch_gpu.png` - Training curves
- `solidmechanics_model_pytorch_gpu.pt` - Trained model

## 🐛 Troubleshooting

**CUDA not detected:**
```bash
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory errors:**
- Reduce `N_r` (collocation points) in the script
- Use mixed precision training (modify DTYPE to float16)
- Reduce batch size for larger networks

**Driver issues:**
- Update NVIDIA drivers to latest version
- Reinstall CUDA toolkit
- Check PyTorch CUDA compatibility

## 💡 Optimization Tips

- Use **power-of-2 network sizes** for better GPU utilization
- Enable **TensorFloat-32** on Ampere GPUs for speed boost
- Monitor GPU utilization with `nvidia-smi` during training
- Consider **gradient accumulation** for memory-intensive problems