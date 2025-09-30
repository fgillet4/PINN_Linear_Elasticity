#!/bin/bash

# PyTorch GPU PINN Environment Setup Script  
# This script creates a fresh virtual environment for the PyTorch GPU implementation

set -e  # Exit on any error

echo "🚀 Setting up PyTorch GPU PINN Environment..."
echo "=============================================="

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Error: Python 3.11 is required but not found."
    echo "Please install Python 3.11 first."
    exit 1
fi

echo "✅ Python 3.11 found: $(python3.11 --version)"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    echo "✅ GPU setup will proceed with CUDA support"
else
    echo "❌ Warning: nvidia-smi not found. No NVIDIA GPU detected."
    echo ""
    echo "🚨 GPU Requirements Not Met!"
    echo "This setup installs PyTorch with CUDA support but your system may not have:"
    echo "  • NVIDIA GPU with CUDA support"
    echo "  • NVIDIA drivers installed"
    echo ""
    echo "💡 Consider using the CPU version instead:"
    echo "   cd ../  # Go back to pytorch_version directory"
    echo "   ./setup_pytorch.sh"
    echo "   python run_pinn_pytorch.py"
    echo ""
    read -p "Continue with GPU setup anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "👋 Exiting. Please try the CPU version setup."
        exit 1
    fi
    echo "⚠️  Continuing with GPU setup (will fall back to CPU during training)..."
fi

# Remove existing virtual environment if it exists
if [ -d "venv_pytorch_gpu" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf venv_pytorch_gpu
fi

# Create new virtual environment
echo "📦 Creating new virtual environment..."
python3.11 -m venv venv_pytorch_gpu

# Activate virtual environment
echo "🔧 Activating virtualvironment..."
source venv_pytorch_gpu/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
echo "   This may take a few minutes..."

# Install PyTorch GPU version (CUDA 12.1 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "📥 Installing additional requirements..."
pip install -r requirements_pytorch_gpu.txt

echo ""
echo "✅ PyTorch GPU PINN environment setup complete!"
echo ""
echo "🎯 To use this environment:"
echo "   source venv_pytorch_gpu/bin/activate"
echo "   python run_pinn_pytorch_gpu.py"
echo ""
echo "🔧 To deactivate:"
echo "   deactivate"
echo ""
echo "💡 GPU Setup Verification:"
echo "   After activation, run: python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
echo ""