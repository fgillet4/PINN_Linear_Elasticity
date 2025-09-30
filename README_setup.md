# TensorFlow PINN Setup

## Quick Start

Run the automated setup script:
```bash
./setup_tensorflow.sh
```

## Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements_tensorflow.txt

# Run the PINN
python run_pinn_stack.py
```

## Requirements

- Python 3.11.x
- The script will install TensorFlow 2.20.0 and all dependencies

## Output Files

After running you'll get:
- `solution.png` - Displacement field visualization
- `stress_map.png` - Stress field visualization  
- `error_map.png` - Error comparison with exact solution
- `loss_history.png` - Training loss history
- `solidmechanics_model_stack.sav` - Trained model