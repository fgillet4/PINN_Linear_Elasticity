# PINN for Solid Mechanics

A Physics-Informed Neural Network (PINN) implementation for solving solid mechanics problems using linear elasticity.

The reference paper is "A deep learning framework for solution and discovery in solid mechanics: linear elasticity"

---

## 🎬 Training Visualizations

Watch the neural network learn to solve elasticity equations in real-time:

<div align="center">

### Component Error Evolution
<img src="pytorch_version/visualizations/animations/component_error_evolution.gif" width="700px">

*How prediction errors in displacement components (Ux, Uy) decrease as the network trains*

### Weight Evolution Through Training
<img src="pytorch_version/visualizations/animations/weight_evolution.gif" width="700px">

*Neural network weights evolving through parameter space during 10,000 training iterations*

### Weight Distribution Across Layers
<img src="pytorch_version/visualizations/animations/weight_bread_slices.gif" width="700px">

*Cross-sectional view of weight distributions across all 8 hidden layers + output layers*

### Physics Equation Discovery
<img src="pytorch_version/visualizations/animations/physics_equation_discovery.gif" width="700px">

*How the network progressively learns to satisfy PDEs, constitutive relations, and boundary conditions*

</div>

---

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

## 🧪 Experiments & Advanced Features

The PyTorch version includes comprehensive experiments for optimizing PINN training:

### Available Experiments:

1. **L2 Regularization Study** (`pytorch_version/experiment_l2_regularization/`)
   - Tests different weight decay values: [0, 1e-6, 1e-5, 1e-4, 1e-3]
   - Analyzes impact on generalization and physics satisfaction
   - Detailed residual tracking for all 17 loss components
   - [Full Documentation](pytorch_version/experiment_l2_regularization/README.md)

2. **Learning Rate Scheduling** (`pytorch_version/experiment_lr_scheduling/`)
   - Compares 7 different LR schedules: constant, manual, step, exponential, cosine, warm restarts, adaptive
   - Analyzes convergence speed vs final accuracy tradeoffs
   - Tracks learning dynamics throughout training
   - [Full Documentation](pytorch_version/experiment_lr_scheduling/README.md)

3. **Residual Tracking & Analysis** (`pytorch_version/pinn_pytorch_with_residual_tracking.py`)
   - Monitors 17 individual residual components during training
   - PDE residuals (momentum balance in x, y)
   - Constitutive residuals (stress-strain relations)
   - 8 boundary condition residuals
   - Generates comprehensive visualization plots

### Running Experiments:

```bash
cd pytorch_version

# Run baseline with detailed tracking
python pinn_pytorch_with_residual_tracking.py

# Run L2 regularization experiments
cd experiment_l2_regularization
python experiment_l2_regularization.py

# Run learning rate scheduling experiments  
cd ../experiment_lr_scheduling
python experiment_lr_scheduling.py

# Compare all experiments
cd ..
python compare_residuals.py
```

### Experiment Outputs:

Each experiment saves:
- `model_[experiment_name].pt` - Trained model checkpoints
- `residuals_[experiment_name].npy` - Complete residual history (17 components)
- Comparison plots showing performance across configurations
- Summary statistics identifying best performers

The comparison tool automatically:
- Loads all experiment results
- Creates side-by-side visualizations
- Identifies which approach achieves best physics satisfaction
- Analyzes convergence behavior and training stability

**Documentation:** See [EXPERIMENTS_README.md](pytorch_version/EXPERIMENTS_README.md) for complete details on theory, mathematics, and implementation.
