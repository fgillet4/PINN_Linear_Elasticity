#!/usr/bin/env python3
"""
Training script with weight evolution animation for PINN
Saves network weight visualizations during training and creates an animated GIF
"""

import torch
import torch.nn as nn
import numpy as np
from pyDOE3 import lhs
import matplotlib.pyplot as plt
from time import time
import os
from visualize_network_graph import NetworkVisualizer
from PIL import Image
import glob

# Set device (CPU for now)
device = torch.device('cpu')
print(f"Using device: {device}")

# Set data type
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

# Set constants & model parameters
pi = torch.tensor(np.pi, dtype=DTYPE, device=device)
E = torch.tensor(4e11/3, dtype=DTYPE, device=device)   # Young's modulus
v = torch.tensor(1/3, dtype=DTYPE, device=device)       # Poisson's ratio
E = E/1e11
lmda = E*v/(1-2*v)/(1+v)
mu = E/(2*(1+v))
Q = torch.tensor(4.0, dtype=DTYPE, device=device)

# Set boundary
xmin = 0.
xmax = 1.
ymin = 0.
ymax = 1.

def u_x_ext(x, y):
    return torch.cos(2*pi*x) * torch.sin(pi*y)

def u_y_ext(x, y):
    return torch.sin(pi*x) * Q/4*torch.pow(y, 4)

def f_x_ext(x, y):
    return 1.0*(-4*torch.pow(pi, 2)*torch.cos(2*pi*x)*torch.sin(pi*y)+pi*torch.cos(pi*x)*Q*torch.pow(y, 3)) + 0.5*(-9*torch.pow(pi, 2)*torch.cos(2*pi*x)*torch.sin(pi*y)+pi*torch.cos(pi*x)*Q*torch.pow(y, 3))

def f_y_ext(x, y):
    return lmda*(3*torch.sin(pi*x)*Q*torch.pow(y, 2)-2*torch.pow(pi, 2)*torch.sin(2*pi*x)*torch.cos(pi*y)) + mu*(6*torch.sin(pi*x)*Q*torch.pow(y, 2)-2*torch.pow(pi, 2)*torch.sin(2*pi*x)*torch.cos(pi*y)-torch.pow(pi, 2)*torch.sin(pi*x)*Q*torch.pow(y, 4)/4)

def fun_b_yy(x, y):
    return (lmda+2*mu)*Q*torch.sin(pi*x)

# Set number of data points
N_bound = 50
N_r = 1000

# Lower and upper bounds
lb = torch.tensor([xmin, ymin], dtype=DTYPE, device=device)
ub = torch.tensor([xmax, ymax], dtype=DTYPE, device=device)

# Set random seed for reproducible data generation only
np.random.seed(0)  # Only for consistent boundary point generation

print("Generating boundary points...")

# Generate boundary training data exactly like TensorFlow version
x_up = lhs(1, samples=N_bound, random_state=123)
x_up = xmin + (xmax-xmin)*x_up
y_up = np.full((len(x_up), 1), ymax)
x_up_train = torch.tensor(np.hstack((x_up, y_up)), dtype=DTYPE, device=device)
eux_up_train = torch.zeros((len(x_up), 1), dtype=DTYPE, device=device)  # Zero displacement BC
Syy_up_train = fun_b_yy(torch.tensor(x_up, dtype=DTYPE), torch.tensor(y_up, dtype=DTYPE))

x_lo = lhs(1, samples=N_bound, random_state=123) 
x_lo = xmin + (xmax-xmin)*x_lo
y_lo = np.full((len(x_lo), 1), ymin)
x_lo_train = torch.tensor(np.hstack((x_lo, y_lo)), dtype=DTYPE, device=device)
eux_lo_train = torch.zeros((len(x_lo), 1), dtype=DTYPE, device=device)  # Zero displacement BC
uy_lo_train = torch.zeros((len(x_lo), 1), dtype=DTYPE, device=device)   # Zero displacement BC

y_ri = lhs(1, samples=N_bound, random_state=123)
y_ri = ymin + (ymax-ymin)*y_ri
x_ri = np.full((len(y_ri), 1), xmax)
x_ri_train = torch.tensor(np.hstack((x_ri, y_ri)), dtype=DTYPE, device=device)
uy_ri_train = torch.zeros((len(x_ri), 1), dtype=DTYPE, device=device)   # Zero displacement BC
Sxx_ri_train = torch.zeros((len(x_ri), 1), dtype=DTYPE, device=device)

y_le = lhs(1, samples=N_bound, random_state=123)
y_le = ymin + (ymax-ymin)*y_le
x_le = np.full((len(y_le), 1), xmin)
x_le_train = torch.tensor(np.hstack((x_le, y_le)), dtype=DTYPE, device=device)
uy_le_train = torch.zeros((len(x_le), 1), dtype=DTYPE, device=device)   # Zero displacement BC
Sxx_le_train = torch.zeros((len(x_le), 1), dtype=DTYPE, device=device)

# Combine boundary data using zero displacement BCs like TensorFlow
eux_b_train = torch.cat([eux_up_train, eux_lo_train], 0)
uy_b_train = torch.cat([uy_lo_train, uy_ri_train, uy_le_train], 0)
Sxx_b_train = torch.cat([Sxx_ri_train, Sxx_le_train], 0)
Syy_b_train = Syy_up_train

# Collocation points
grid_pt = lhs(2, N_r)
grid_pt[:, 0] = xmin + (xmax-xmin)*grid_pt[:, 0]
grid_pt[:, 1] = ymin + (ymax-ymin)*grid_pt[:, 1]
X_col_train = torch.tensor(grid_pt, dtype=DTYPE, device=device)

print("Data preparation completed!")

# Neural Network Model (same as original)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        self.register_buffer('lb', lb)
        self.register_buffer('ub', ub)
        
        # Hidden layers exactly like TensorFlow (8 layers, 20 neurons each)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(2, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20)
        ])
        
        # Output layers
        self.output_Ux = nn.Linear(20, 1)
        self.output_Uy = nn.Linear(20, 1)
        self.output_Sxx = nn.Linear(20, 1)
        self.output_Syy = nn.Linear(20, 1)
        self.output_Sxy = nn.Linear(20, 1)
        
        # Initialize weights
        for layer in self.hidden_layers + [self.output_Ux, self.output_Uy, self.output_Sxx, self.output_Syy, self.output_Sxy]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Scale input to [-1, 1]
        x_scaled = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        
        # Forward through hidden layers
        h = x_scaled
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
        
        # Outputs
        Ux = self.output_Ux(h)
        Uy = self.output_Uy(h)
        Sxx = self.output_Sxx(h)
        Syy = self.output_Syy(h)
        Sxy = self.output_Sxy(h)
        
        return Ux, Uy, Sxx, Syy, Sxy

# Create directories for snapshots
os.makedirs('weight_snapshots', exist_ok=True)
os.makedirs('animations', exist_ok=True)

# Set a time-based seed for weight initialization (different each run)
import time as time_module
weight_seed = int(time_module.time()) % 10000
torch.manual_seed(weight_seed)
print(f"🎲 Using weight initialization seed: {weight_seed}")

model = PINN().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Debug: Check initial weight statistics
print("\n🔍 Initial weight statistics:")
for name, param in model.named_parameters():
    if 'weight' in name:
        weight_stats = param.data
        print(f"  {name}: min={weight_stats.min().item():.4f}, max={weight_stats.max().item():.4f}, mean={weight_stats.mean().item():.4f}, std={weight_stats.std().item():.4f}")
        break  # Just show first layer for brevity

# Loss computation (same as original)
def compute_loss():
    # Physics loss
    X_col_req = X_col_train.requires_grad_(True)
    Ux, Uy, Sxx, Syy, Sxy = model(X_col_req)
    
    x = X_col_req[:, 0:1]
    y = X_col_req[:, 1:2]
    
    # Gradients for PDE
    grad_Sxx = torch.autograd.grad(Sxx, X_col_req, grad_outputs=torch.ones_like(Sxx), create_graph=True, retain_graph=True)[0]
    grad_Syy = torch.autograd.grad(Syy, X_col_req, grad_outputs=torch.ones_like(Syy), create_graph=True, retain_graph=True)[0]
    grad_Sxy = torch.autograd.grad(Sxy, X_col_req, grad_outputs=torch.ones_like(Sxy), create_graph=True, retain_graph=True)[0]
    
    dsxxdx = grad_Sxx[:, 0:1]
    dsyydy = grad_Syy[:, 1:2]
    dsxydx = grad_Sxy[:, 0:1]
    dsxydy = grad_Sxy[:, 1:2]
    
    # PDE residuals
    r_x = dsxxdx + dsxydy - f_x_ext(x, y)
    r_y = dsxydx + dsyydy - f_y_ext(x, y)
    phi_r = torch.mean(torch.abs(r_x)) + torch.mean(torch.abs(r_y))
    
    # Constitutive equations
    grad_Ux = torch.autograd.grad(Ux, X_col_req, grad_outputs=torch.ones_like(Ux), create_graph=True, retain_graph=True)[0]
    grad_Uy = torch.autograd.grad(Uy, X_col_req, grad_outputs=torch.ones_like(Uy), create_graph=True, retain_graph=True)[0]
    
    duxdx = grad_Ux[:, 0:1]
    duxdy = grad_Ux[:, 1:2]
    duydx = grad_Uy[:, 0:1]
    duydy = grad_Uy[:, 1:2]
    
    r_const_x = (lmda + 2*mu)*duxdx + lmda*duydy - Sxx
    r_const_y = (lmda + 2*mu)*duydy + lmda*duxdx - Syy
    r_const_xy = 2*mu*0.5*(duxdy + duydx) - Sxy
    phi_r_const = torch.mean(torch.abs(r_const_x)) + torch.mean(torch.abs(r_const_y)) + torch.mean(torch.abs(r_const_xy))
    
    # Boundary conditions
    Ux_up, Uy_up, _, Syy_up, _ = model(x_up_train)
    Ux_lo, Uy_lo, _, _, _ = model(x_lo_train)
    _, Uy_ri, Sxx_ri, _, _ = model(x_ri_train)
    _, Uy_le, Sxx_le, _, _ = model(x_le_train)
    
    r_ux = torch.cat([Ux_up, Ux_lo], 0) - eux_b_train
    r_uy = torch.cat([Uy_lo, Uy_ri, Uy_le], 0) - uy_b_train
    r_Sxx = torch.cat([Sxx_ri, Sxx_le], 0) - Sxx_b_train
    r_Syy = Syy_up - Syy_b_train
    
    phi_r_u = torch.mean(torch.abs(r_ux)) + torch.mean(torch.abs(r_uy))
    phi_r_S = torch.mean(torch.abs(r_Sxx)) + torch.mean(torch.abs(r_Syy))
    
    total_loss = phi_r + phi_r_const + phi_r_u + phi_r_S
    return total_loss

def save_weight_snapshot(epoch, model_state_dict, loss_value):
    """Save a weight visualization snapshot for the current epoch"""
    # Create a temporary model file
    temp_model_path = f'weight_snapshots/temp_model_epoch_{epoch:04d}.pt'
    torch.save(model_state_dict, temp_model_path)
    
    # Create visualizer and generate image
    visualizer = NetworkVisualizer(temp_model_path)
    visualizer.load_model()
    visualizer.analyze_architecture()
    
    # Create weight visualization with epoch info in title
    fig, ax = visualizer.visualize_network_with_weights(
        figsize=(18, 8), 
        weight_threshold=0.02,  # Lower threshold to see more connections during training
        save_path=f'weight_snapshots/weights_epoch_{epoch:04d}.png'
    )
    
    # Update title to include epoch and loss info
    total_params = sum(info['weight_tensor'].numel() + 
                     (info['bias_tensor'].numel() if info['bias_tensor'] is not None else 0)
                     for info in visualizer.layer_info.values())
    
    title_text = f'PINN Weight Evolution - Epoch {epoch}\nLoss: {loss_value:.6e} | {total_params:,} Parameters | Blue=Positive, Red=Negative'
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)
    
    # Add input labels (x, y) to the input neurons
    # Add x and y labels next to input neurons
    input_labels = ['x', 'y']
    input_y_positions = [-0.8, 0.8]  # Approximate positions for 2 input neurons
    input_x = 0  # Input layer x position
    
    for i, (label, y_pos) in enumerate(zip(input_labels, input_y_positions)):
        ax.text(input_x - 0.35, y_pos, label, ha='right', va='center',
               fontsize=10, fontweight='bold', color='darkgreen',
               bbox=dict(boxstyle="circle,pad=0.1", facecolor='lightgreen', alpha=0.8))
    
    # Save with updated title and labels
    plt.savefig(f'weight_snapshots/weights_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Keep temporary model files for 3D visualization
    # os.remove(temp_model_path)  # Comment out to keep for 3D viz
    
    print(f"📸 Saved weight snapshot for epoch {epoch}")

def create_weight_evolution_gif():
    """Create an animated GIF from weight snapshots"""
    print("\n🎬 Creating weight evolution animation...")
    
    # Get all weight snapshot images
    image_files = sorted(glob.glob('weight_snapshots/weights_epoch_*.png'))
    
    if not image_files:
        print("❌ No weight snapshots found")
        return
    
    # Create GIF
    images = []
    for img_file in image_files:
        img = Image.open(img_file)
        images.append(img)
    
    # Save as animated GIF
    if images:
        images[0].save(
            'animations/weight_evolution.gif',
            save_all=True,
            append_images=images[1:],
            duration=200,  # 200ms per frame
            loop=0
        )
        print(f"✅ Weight evolution GIF saved as 'animations/weight_evolution.gif'")
        print(f"📊 Total frames: {len(images)}")
    
    # Create a faster version too
    if len(images) > 20:
        # Take every 5th frame for a faster version
        fast_images = images[::5]
        fast_images[0].save(
            'animations/weight_evolution_fast.gif',
            save_all=True,
            append_images=fast_images[1:],
            duration=150,  # 150ms per frame
            loop=0
        )
        print(f"✅ Fast weight evolution GIF saved as 'animations/weight_evolution_fast.gif'")
        print(f"📊 Fast version frames: {len(fast_images)}")

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

print("Starting training with weight evolution tracking...")
N = 2000  # Reduced for demonstration, but you can use 10000
snapshot_interval = 50  # Save snapshot every 50 epochs
hist = []
t0 = time()

# Save initial weights (epoch 0) - BEFORE any training
print("📸 Saving initial random weights (epoch 0)...")
save_weight_snapshot(0, model.state_dict(), float('inf'))

for i in range(1, N + 1):
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    optimizer.step()
    
    if i == 1:
        loss0 = loss.item()
    
    hist.append(loss.item() / loss0)
    
    if i % 50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i, loss.item()))
    
    # Save weight snapshot at specified intervals
    if i % snapshot_interval == 0 or i == N:
        save_weight_snapshot(i, model.state_dict(), loss.item())
    
    # Learning rate schedule
    if i == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
    elif i == 1500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-4

print('\nComputation time: {} seconds'.format(time() - t0))

# Create the animated GIF
create_weight_evolution_gif()

# Save final model
torch.save(model.state_dict(), 'solidmechanics_model_animated.pt')

print("\n🎉 Training with weight evolution animation completed!")
print("📁 Generated files:")
print("   📂 weight_snapshots/ - Individual weight visualizations for each epoch")
print("   📂 animations/weight_evolution.gif - Animated weight evolution")
print("   📂 animations/weight_evolution_fast.gif - Faster version (every 5th frame)")
print("   📁 solidmechanics_model_animated.pt - Final trained model")

print("\n💡 You can view the GIF in any image viewer or web browser to see how the weights evolve!")