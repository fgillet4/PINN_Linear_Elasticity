#!/bin/bash

# PyTorch PINN Environment Setup Script  
# This script creates a fresh virtual environment for the PyTorch implementation

set -e  # Exit on any error

echo "🚀 Setting up PyTorch PINN Environment..."
echo "=============================================="

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Error: Python 3.11 is required but not found."
    echo "Please install Python 3.11 first."
    exit 1
fi

echo "✅ Python 3.11 found: $(python3.11 --version)"

# Remove existing virtual environment if it exists
if [ -d "venv_pytorch" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf venv_pytorch
fi

# Create new virtual environment
echo "📦 Creating new virtual environment..."
python3.11 -m venv venv_pytorch

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_pytorch/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing PyTorch PINN requirements..."
pip install -r requirements_pytorch.txt

echo ""
echo "✅ PyTorch PINN environment setup complete!"
echo ""
echo "🎯 To use this environment:"
echo "   source venv_pytorch/bin/activate"
echo "   python run_pinn_pytorch.py"
echo ""
echo "🔧 To deactivate:"
echo "   deactivate"
echo ""