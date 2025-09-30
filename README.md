# PINN-for-solid-mechanics

A Physics-Informed Neural Network (PINN) implementation for solving solid mechanics problems using linear elasticity.

The reference paper is "A deep learning framework for solution and discovery in solid mechanics: linear elasticity"

## 🚀 Quick Start

### For Linux/macOS Users:

**TensorFlow Version:**
```bash
./setup_tensorflow.sh
python run_pinn_stack.py
```

**PyTorch Version:**
```bash
cd pytorch_version
./setup_pytorch.sh  
python run_pinn_pytorch.py
```

### For Windows Users:

**TensorFlow Version:**
```cmd
WINDOWS_setup_tensorflow.bat
```

**PyTorch Version:**
```cmd
cd pytorch_version
WINDOWS_setup_pytorch.bat
```

## 📁 Project Structure

```
PINN-for-solid-mechanics/
├── 🔧 setup_tensorflow.sh          # Linux/macOS TensorFlow setup
├── 🔧 WINDOWS_setup_tensorflow.bat  # Windows TensorFlow setup
├── 📦 requirements_tensorflow.txt   # TensorFlow dependencies
├── 🚀 run_pinn_stack.py             # TensorFlow PINN implementation
└── pytorch_version/
    ├── 🔧 setup_pytorch.sh          # Linux/macOS PyTorch setup  
    ├── 🔧 WINDOWS_setup_pytorch.bat  # Windows PyTorch setup
    ├── 📦 requirements_pytorch.txt   # PyTorch dependencies
    └── 🚀 run_pinn_pytorch.py        # PyTorch PINN implementation
```

## 🎯 Features

- **Dual Implementation**: Both TensorFlow and PyTorch versions available
- **Cross-Platform**: Setup scripts for Linux, macOS, and Windows
- **Identical Results**: Both implementations produce matching results
- **Automated Setup**: One-click environment configuration
- **Python 3.11 Compatible**: Uses latest Python features

## 📊 Output

Both implementations generate:
- Displacement field visualizations (Ux, Uy)
- Stress field visualizations (Sxx, Syy, Sxy)
- Error maps comparing with analytical solutions
- Training loss history plots
- Trained model files
