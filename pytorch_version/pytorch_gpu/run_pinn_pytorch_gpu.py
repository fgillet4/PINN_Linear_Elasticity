# Import PyTorch and NumPy  
import torch
import torch.nn as nn
import numpy as np
from pyDOE3 import lhs
import matplotlib.pyplot as plt
from time import time
import sys

print("🔍 Checking GPU Hardware Compatibility...")
print("=" * 50)

# Comprehensive GPU Hardware Check
def check_gpu_requirements():
    """Check if system meets GPU requirements and provide guidance"""
    
    # Check if CUDA is available in PyTorch
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available:
        print("❌ CUDA Support: Not Available")
        print("\n🚨 GPU Requirements Not Met!")
        print("=" * 50)
        print("This script requires:")
        print("  • NVIDIA GPU with CUDA support")
        print("  • CUDA drivers installed (v525.60.11+)")
        print("  • PyTorch compiled with CUDA support")
        print()
        print("💡 What to do next:")
        print("  1. Check if you have an NVIDIA GPU:")
        print("     Run: nvidia-smi")
        print("  2. Install CUDA drivers from: https://developer.nvidia.com/cuda-downloads")
        print("  3. Or use the CPU version instead:")
        print("     cd ../  # Go back to pytorch_version directory")
        print("     python run_pinn_pytorch.py")
        print()
        print("🔄 The CPU version produces identical results")
        print("   (just takes ~10-15 minutes instead of 2-3 minutes)")
        print()
        
        user_choice = input("Continue anyway? (will fall back to CPU) [y/N]: ").strip().lower()
        if user_choice not in ['y', 'yes']:
            print("👋 Exiting. Please try the CPU version: python ../run_pinn_pytorch.py")
            sys.exit(0)
        else:
            print("⚠️  Continuing with CPU fallback...")
            return torch.device('cpu')
    
    # CUDA is available - check details
    device_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    compute_capability = torch.cuda.get_device_capability(0)
    
    print("✅ CUDA Support: Available")
    print(f"🎮 GPU Found: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    print(f"🔢 CUDA Version: {torch.version.cuda}")
    print(f"🧮 Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    
    # Check memory requirements
    if gpu_memory < 4.0:
        print("⚠️  Warning: GPU has limited memory (<4GB)")
        print("   You may need to reduce problem size if you encounter memory errors")
    
    # Check compute capability
    if compute_capability[0] < 3 or (compute_capability[0] == 3 and compute_capability[1] < 5):
        print("⚠️  Warning: GPU compute capability is below 3.5")
        print("   Some optimizations may not be available")
    
    print("\n🚀 All GPU requirements met! Preparing for GPU acceleration...")
    return torch.device('cuda')

# Run the check
device = check_gpu_requirements()
print(f"✅ Selected device: {device}")
print("=" * 50)

# Clear GPU cache if using CUDA
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print("🧹 GPU cache cleared")

# Set data type
DTYPE = torch.float32  # Use float32 for better GPU performance
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

print("📍 Generating boundary points...")

# Generate boundary training data exactly like CPU version
x_up = lhs(1, samples=N_bound, random_state=123)
x_up = xmin + (xmax-xmin)*x_up
y_up = np.full((len(x_up), 1), ymax)
x_up_train = torch.tensor(np.hstack((x_up, y_up)), dtype=DTYPE, device=device)
eux_up_train = torch.zeros((len(x_up), 1), dtype=DTYPE, device=device)
Syy_up_train = fun_b_yy(torch.tensor(x_up, dtype=DTYPE, device=device), torch.tensor(y_up, dtype=DTYPE, device=device))

x_lo = lhs(1, samples=N_bound, random_state=123) 
x_lo = xmin + (xmax-xmin)*x_lo
y_lo = np.full((len(x_lo), 1), ymin)
x_lo_train = torch.tensor(np.hstack((x_lo, y_lo)), dtype=DTYPE, device=device)
eux_lo_train = torch.zeros((len(x_lo), 1), dtype=DTYPE, device=device)
uy_lo_train = torch.zeros((len(x_lo), 1), dtype=DTYPE, device=device)

y_ri = lhs(1, samples=N_bound, random_state=123)
y_ri = ymin + (ymax-ymin)*y_ri
x_ri = np.full((len(y_ri), 1), xmax)
x_ri_train = torch.tensor(np.hstack((x_ri, y_ri)), dtype=DTYPE, device=device)
uy_ri_train = torch.zeros((len(x_ri), 1), dtype=DTYPE, device=device)
Sxx_ri_train = torch.zeros((len(x_ri), 1), dtype=DTYPE, device=device)

y_le = lhs(1, samples=N_bound, random_state=123)
y_le = ymin + (ymax-ymin)*y_le
x_le = np.full((len(y_le), 1), xmin)
x_le_train = torch.tensor(np.hstack((x_le, y_le)), dtype=DTYPE, device=device)
uy_le_train = torch.zeros((len(x_le), 1), dtype=DTYPE, device=device)
Sxx_le_train = torch.zeros((len(x_le), 1), dtype=DTYPE, device=device)

# Combine boundary data
eux_b_train = torch.cat([eux_up_train, eux_lo_train], 0)
uy_b_train = torch.cat([uy_lo_train, uy_ri_train, uy_le_train], 0)
Sxx_b_train = torch.cat([Sxx_ri_train, Sxx_le_train], 0)
Syy_b_train = Syy_up_train

# Collocation points
grid_pt = lhs(2, N_r)
grid_pt[:, 0] = xmin + (xmax-xmin)*grid_pt[:, 0]
grid_pt[:, 1] = ymin + (ymax-ymin)*grid_pt[:, 1]
X_col_train = torch.tensor(grid_pt, dtype=DTYPE, device=device)

print("✅ Data preparation completed!")

# Neural Network Model - Optimized for GPU
class PINN_GPU(nn.Module):
    def __init__(self, num_layers=8, num_neurons=20):
        super(PINN_GPU, self).__init__()
        
        self.register_buffer('lb', lb)
        self.register_buffer('ub', ub)
        
        # Build layers with GPU-optimized structure
        layers = []
        layers.append(nn.Linear(2, num_neurons))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.Tanh())
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layers
        self.output_Ux = nn.Linear(num_neurons, 1)
        self.output_Uy = nn.Linear(num_neurons, 1)
        self.output_Sxx = nn.Linear(num_neurons, 1)
        self.output_Syy = nn.Linear(num_neurons, 1)
        self.output_Sxy = nn.Linear(num_neurons, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Normalize input to [-1, 1]
        x_scaled = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        
        # Forward through hidden layers
        h = self.hidden_layers(x_scaled)
        
        # Get outputs
        Ux = self.output_Ux(h)
        Uy = self.output_Uy(h)
        Sxx = self.output_Sxx(h)
        Syy = self.output_Syy(h)
        Sxy = self.output_Sxy(h)
        
        return Ux, Uy, Sxx, Syy, Sxy

model = PINN_GPU().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"🧠 Model created with {total_params:,} parameters")

if device.type == 'cuda':
    print(f"📊 GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

# Loss computation - GPU optimized
def compute_loss():
    # Physics loss
    X_col_req = X_col_train.requires_grad_(True)
    Ux, Uy, Sxx, Syy, Sxy = model(X_col_req)
    
    x = X_col_req[:, 0:1]
    y = X_col_req[:, 1:2]
    
    # Gradients for PDE (vectorized for GPU efficiency)
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

# Training setup - GPU optimized
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

print("🚀 Starting GPU training...")
N = 10000
hist = []
t0 = time()

# GPU memory monitoring
if device.type == 'cuda':
    torch.cuda.synchronize()

for i in range(N + 1):
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    optimizer.step()
    
    if i == 0:
        loss0 = loss.item()
    
    hist.append(loss.item() / loss0)
    
    if i % 50 == 0:
        gpu_mem = f" | GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB" if device.type == 'cuda' else ""
        print('It {:05d}: loss = {:10.8e}{}'.format(i, loss.item(), gpu_mem))
    
    # Learning rate schedule
    if i == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
    elif i == 3000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-4

if device.type == 'cuda':
    torch.cuda.synchronize()

print('\n⚡ Computation time: {:.2f} seconds'.format(time() - t0))
if device.type == 'cuda':
    print(f"📊 Peak GPU Memory: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")

# Evaluation
print("📊 Evaluating model...")
N_plot = 600
xspace = np.linspace(0, 1, N_plot + 1)
yspace = np.linspace(0, 1, N_plot + 1)
X, Y = np.meshgrid(xspace, yspace)
Xgrid_tensor = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T, dtype=DTYPE, device=device)

model.eval()
with torch.no_grad():
    Ux_pred, Uy_pred, Sxx_pred, Syy_pred, Sxy_pred = model(Xgrid_tensor)

# Move results to CPU for plotting
Ux = Ux_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Uy = Uy_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Sxx = Sxx_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Syy = Syy_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Sxy = Sxy_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)

# Plot displacement components
U_total = [Ux, Uy]
U_total_name = ['Ux_NN_GPU', 'Uy_NN_GPU']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, U_total[i], cmap='seismic', vmin=-0.8, vmax=0.8)
    ax.set_title(U_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
fig.savefig('solution_pytorch_gpu.png')

# Calculate exact solutions for comparison
ux_ext_flat = u_x_ext(torch.tensor(X.flatten(), dtype=DTYPE, device=device), torch.tensor(Y.flatten(), dtype=DTYPE, device=device))
uy_ext_flat = u_y_ext(torch.tensor(X.flatten(), dtype=DTYPE, device=device), torch.tensor(Y.flatten(), dtype=DTYPE, device=device))

# Reshape exact solutions
Ux_ext = ux_ext_flat.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Uy_ext = uy_ext_flat.cpu().numpy().reshape(N_plot + 1, N_plot + 1)

# Plot errors
error_total = [abs(Ux - Ux_ext), abs(Uy - Uy_ext)]
error_total_name = ['point wise error Ux (GPU)', 'point wise error Uy (GPU)']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, error_total[i], cmap='jet')
    ax.set_title(error_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
fig.savefig('error_map_pytorch_gpu.png')

# Plot stress components
S_total = [Sxx, Syy, Sxy]
S_total_name = ['Sxx_NN_GPU', 'Syy_NN_GPU', 'Sxy_NN_GPU']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
for i, ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, S_total[i], cmap='seismic', vmin=-10, vmax=10)
    ax.set_title(S_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
fig.savefig('stress_map_pytorch_gpu.png')

# Plot loss history
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
ax.semilogy(range(len(hist)), hist, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$')
ax.set_title('Training Loss History (GPU)')
fig.savefig('loss_history_pytorch_gpu.png')

# Save model
torch.save(model.state_dict(), 'solidmechanics_model_pytorch_gpu.pt')

# GPU cleanup
if device.type == 'cuda':
    torch.cuda.empty_cache()

print("🎯 PyTorch GPU PINN training completed successfully!")
print(f"💾 Final GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB" if device.type == 'cuda' else "")