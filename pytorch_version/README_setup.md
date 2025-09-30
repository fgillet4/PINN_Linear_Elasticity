# PyTorch PINN Setup

## Quick Start

Run the automated setup script:
```bash
./setup_pytorch.sh
```

## Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment
python3.11 -m venv venv_pytorch

# Activate environment
source venv_pytorch/bin/activate

# Install dependencies
pip install -r requirements_pytorch.txt

# Run the PINN
python run_pinn_pytorch.py
```

## Requirements

- Python 3.11.x
- The script will install PyTorch 2.8.0 and all dependencies

## Output Files

After running you'll get:
- `solution_pytorch.png` - Displacement field visualization
- `stress_map_pytorch.png` - Stress field visualization  
- `error_map_pytorch.png` - Error comparison with exact solution
- `loss_history_pytorch.png` - Training loss history
- `solidmechanics_model_pytorch.pt` - Trained model

## Notes

- This PyTorch implementation produces identical results to the TensorFlow version
- Uses pyDOE3 instead of pyDOE2 for Python 3.11+ compatibility
- CPU-based implementation (can be easily modified for GPU)