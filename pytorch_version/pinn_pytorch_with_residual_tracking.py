#!/usr/bin/env python3
"""
Enhanced PINN training script with comprehensive residual tracking and visualization
"""

# Import PyTorch and NumPy  
import torch
import torch.nn as nn
import numpy as np
from pyDOE3 import lhs
import matplotlib.pyplot as plt
from time import time

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

# Set random seed for reproducible results
torch.manual_seed(0)
np.random.seed(0)

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

# Neural Network Model
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

model = PINN().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Enhanced loss computation with detailed tracking
def compute_detailed_loss():
    """Compute loss with detailed component tracking"""
    
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
    phi_r_pde_x = torch.mean(torch.abs(r_x))
    phi_r_pde_y = torch.mean(torch.abs(r_y))
    phi_r = phi_r_pde_x + phi_r_pde_y
    
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
    
    phi_r_const_x = torch.mean(torch.abs(r_const_x))
    phi_r_const_y = torch.mean(torch.abs(r_const_y))
    phi_r_const_xy = torch.mean(torch.abs(r_const_xy))
    phi_r_const = phi_r_const_x + phi_r_const_y + phi_r_const_xy
    
    # Boundary conditions
    Ux_up, Uy_up, _, Syy_up, _ = model(x_up_train)
    Ux_lo, Uy_lo, _, _, _ = model(x_lo_train)
    _, Uy_ri, Sxx_ri, _, _ = model(x_ri_train)
    _, Uy_le, Sxx_le, _, _ = model(x_le_train)
    
    r_ux_up = Ux_up - eux_up_train
    r_ux_lo = Ux_lo - eux_lo_train
    r_uy_lo = Uy_lo - uy_lo_train
    r_uy_ri = Uy_ri - uy_ri_train
    r_uy_le = Uy_le - uy_le_train
    r_Sxx_ri = Sxx_ri - Sxx_ri_train
    r_Sxx_le = Sxx_le - Sxx_le_train
    r_Syy_up = Syy_up - Syy_b_train
    
    # Detailed boundary condition losses
    phi_bc_ux_up = torch.mean(torch.abs(r_ux_up))
    phi_bc_ux_lo = torch.mean(torch.abs(r_ux_lo))
    phi_bc_uy_lo = torch.mean(torch.abs(r_uy_lo))
    phi_bc_uy_ri = torch.mean(torch.abs(r_uy_ri))
    phi_bc_uy_le = torch.mean(torch.abs(r_uy_le))
    phi_bc_sxx_ri = torch.mean(torch.abs(r_Sxx_ri))
    phi_bc_sxx_le = torch.mean(torch.abs(r_Sxx_le))
    phi_bc_syy_up = torch.mean(torch.abs(r_Syy_up))
    
    # Combined boundary losses
    phi_r_u = phi_bc_ux_up + phi_bc_ux_lo + phi_bc_uy_lo + phi_bc_uy_ri + phi_bc_uy_le
    phi_r_S = phi_bc_sxx_ri + phi_bc_sxx_le + phi_bc_syy_up
    
    total_loss = phi_r + phi_r_const + phi_r_u + phi_r_S
    
    # Return detailed breakdown
    return {
        'total_loss': total_loss,
        
        # Main categories
        'pde_loss': phi_r,
        'constitutive_loss': phi_r_const,
        'boundary_u_loss': phi_r_u,
        'boundary_s_loss': phi_r_S,
        
        # PDE components
        'pde_x': phi_r_pde_x,
        'pde_y': phi_r_pde_y,
        
        # Constitutive components
        'const_xx': phi_r_const_x,
        'const_yy': phi_r_const_y,
        'const_xy': phi_r_const_xy,
        
        # Boundary components
        'bc_ux_up': phi_bc_ux_up,
        'bc_ux_lo': phi_bc_ux_lo,
        'bc_uy_lo': phi_bc_uy_lo,
        'bc_uy_ri': phi_bc_uy_ri,
        'bc_uy_le': phi_bc_uy_le,
        'bc_sxx_ri': phi_bc_sxx_ri,
        'bc_sxx_le': phi_bc_sxx_le,
        'bc_syy_up': phi_bc_syy_up
    }

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

print("Starting training with detailed residual tracking...")
N = 10000
t0 = time()

# Initialize tracking arrays
epochs = []
residual_history = {
    'total_loss': [],
    'pde_loss': [],
    'constitutive_loss': [],
    'boundary_u_loss': [],
    'boundary_s_loss': [],
    'pde_x': [],
    'pde_y': [],
    'const_xx': [],
    'const_yy': [],
    'const_xy': [],
    'bc_ux_up': [],
    'bc_ux_lo': [],
    'bc_uy_lo': [],
    'bc_uy_ri': [],
    'bc_uy_le': [],
    'bc_sxx_ri': [],
    'bc_sxx_le': [],
    'bc_syy_up': []
}

for i in range(N + 1):
    optimizer.zero_grad()
    loss_components = compute_detailed_loss()
    loss = loss_components['total_loss']
    loss.backward()
    optimizer.step()
    
    # Track residuals every 10 epochs
    if i % 10 == 0:
        epochs.append(i)
        for key in residual_history.keys():
            residual_history[key].append(loss_components[key].item())
    
    if i % 50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i, loss.item()))
    
    # Learning rate schedule
    if i == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
    elif i == 3000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-4

print('\nComputation time: {} seconds'.format(time() - t0))

# Create comprehensive residual plots
print("Creating detailed residual plots...")

# Plot 1: Main component breakdown
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('PINN Training: Residual Component Evolution', fontsize=16, fontweight='bold')

# Main categories
ax = axes[0, 0]
ax.semilogy(epochs, residual_history['total_loss'], 'k-', linewidth=2, label='Total Loss')
ax.semilogy(epochs, residual_history['pde_loss'], 'r-', label='PDE Loss')
ax.semilogy(epochs, residual_history['constitutive_loss'], 'b-', label='Constitutive Loss')
ax.semilogy(epochs, residual_history['boundary_u_loss'], 'g-', label='Boundary U Loss')
ax.semilogy(epochs, residual_history['boundary_s_loss'], 'm-', label='Boundary σ Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Main Loss Components')
ax.legend()
ax.grid(True, alpha=0.3)

# PDE components
ax = axes[0, 1]
ax.semilogy(epochs, residual_history['pde_x'], 'r-', label='PDE X (∂σₓₓ/∂x + ∂σₓᵧ/∂y = fₓ)')
ax.semilogy(epochs, residual_history['pde_y'], 'b-', label='PDE Y (∂σₓᵧ/∂x + ∂σᵧᵧ/∂y = fᵧ)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Residual (log scale)')
ax.set_title('PDE Equation Residuals')
ax.legend()
ax.grid(True, alpha=0.3)

# Constitutive components
ax = axes[1, 0]
ax.semilogy(epochs, residual_history['const_xx'], 'r-', label='σₓₓ = (λ+2μ)εₓₓ + λεᵧᵧ')
ax.semilogy(epochs, residual_history['const_yy'], 'b-', label='σᵧᵧ = (λ+2μ)εᵧᵧ + λεₓₓ')
ax.semilogy(epochs, residual_history['const_xy'], 'g-', label='σₓᵧ = 2μεₓᵧ')
ax.set_xlabel('Epoch')
ax.set_ylabel('Residual (log scale)')
ax.set_title('Constitutive Equation Residuals')
ax.legend()
ax.grid(True, alpha=0.3)

# Boundary condition components
ax = axes[1, 1]
ax.semilogy(epochs, residual_history['bc_ux_up'], label='Uₓ=0 (top)')
ax.semilogy(epochs, residual_history['bc_ux_lo'], label='Uₓ=0 (bottom)')
ax.semilogy(epochs, residual_history['bc_uy_lo'], label='Uᵧ=0 (bottom)')
ax.semilogy(epochs, residual_history['bc_uy_ri'], label='Uᵧ=0 (right)')
ax.semilogy(epochs, residual_history['bc_uy_le'], label='Uᵧ=0 (left)')
ax.semilogy(epochs, residual_history['bc_syy_up'], label='σᵧᵧ prescribed (top)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Residual (log scale)')
ax.set_title('Boundary Condition Residuals')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_residual_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Detailed boundary conditions
fig, ax = plt.subplots(figsize=(12, 8))
ax.semilogy(epochs, residual_history['bc_ux_up'], 'r-', linewidth=2, label='Uₓ=0 (top boundary)')
ax.semilogy(epochs, residual_history['bc_ux_lo'], 'r--', linewidth=2, label='Uₓ=0 (bottom boundary)')
ax.semilogy(epochs, residual_history['bc_uy_lo'], 'b-', linewidth=2, label='Uᵧ=0 (bottom boundary)')
ax.semilogy(epochs, residual_history['bc_uy_ri'], 'b--', linewidth=2, label='Uᵧ=0 (right boundary)')
ax.semilogy(epochs, residual_history['bc_uy_le'], 'b:', linewidth=2, label='Uᵧ=0 (left boundary)')
ax.semilogy(epochs, residual_history['bc_sxx_ri'], 'g-', linewidth=2, label='σₓₓ=0 (right boundary)')
ax.semilogy(epochs, residual_history['bc_sxx_le'], 'g--', linewidth=2, label='σₓₓ=0 (left boundary)')
ax.semilogy(epochs, residual_history['bc_syy_up'], 'm-', linewidth=2, label='σᵧᵧ prescribed (top boundary)')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Boundary Condition Residual (log scale)', fontsize=12)
ax.set_title('Detailed Boundary Condition Evolution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('detailed_boundary_residuals.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Learning phases
fig, ax = plt.subplots(figsize=(12, 8))
ax.semilogy(epochs, residual_history['total_loss'], 'k-', linewidth=3, label='Total Loss', alpha=0.8)

# Add learning phase annotations
ax.axvline(x=1000, color='orange', linestyle='--', alpha=0.7, label='LR: 1e-2 → 1e-3')
ax.axvline(x=3000, color='red', linestyle='--', alpha=0.7, label='LR: 1e-3 → 5e-4')

# Add phase labels
ax.text(500, max(residual_history['total_loss'])/2, 'Phase 1\nRapid Learning\n(lr=1e-2)', 
        ha='center', va='center', fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

ax.text(2000, max(residual_history['total_loss'])/10, 'Phase 2\nOptimization\n(lr=1e-3)', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

ax.text(6500, max(residual_history['total_loss'])/100, 'Phase 3\nFine-tuning\n(lr=5e-4)', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Total Loss (log scale)', fontsize=12)
ax.set_title('PINN Training Phases and Learning Rate Schedule', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_phases.png', dpi=300, bbox_inches='tight')
plt.show()

# Save model
torch.save(model.state_dict(), 'solidmechanics_model_pytorch_detailed.pt')

print("\n🎉 Enhanced PINN training completed!")
print("📊 Generated detailed residual tracking plots:")
print("   📁 comprehensive_residual_evolution.png - 4-panel overview")
print("   📁 detailed_boundary_residuals.png - All boundary conditions")
print("   📁 training_phases.png - Learning phases and schedules")
print("   📁 solidmechanics_model_pytorch_detailed.pt - Trained model")

# Print final residual summary
print(f"\n📈 Final Residual Summary (Epoch {N}):")
print(f"   Total Loss: {residual_history['total_loss'][-1]:.2e}")
print(f"   PDE Loss: {residual_history['pde_loss'][-1]:.2e}")
print(f"   Constitutive Loss: {residual_history['constitutive_loss'][-1]:.2e}")
print(f"   Boundary Loss: {residual_history['boundary_u_loss'][-1] + residual_history['boundary_s_loss'][-1]:.2e}")
print(f"   Best performing equation: {min(residual_history, key=lambda x: residual_history[x][-1] if x != 'total_loss' else float('inf'))}")
print(f"   Worst performing equation: {max(residual_history, key=lambda x: residual_history[x][-1] if x != 'total_loss' else 0)}")
