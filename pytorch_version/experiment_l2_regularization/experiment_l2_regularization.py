# L2 Regularization Experiment for PINN
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

# Function to compute L2 regularization
def l2_regularization(model):
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param, p=2) ** 2
    return l2_loss

# Enhanced loss computation with detailed tracking and L2 regularization
def compute_detailed_loss_with_l2(model, l2_lambda):
    """Compute loss with detailed component tracking and L2 regularization"""
    
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
    
    # Physics loss
    physics_loss = phi_r + phi_r_const + phi_r_u + phi_r_S
    
    # L2 regularization loss
    l2_loss = l2_regularization(model)
    
    # Total loss
    total_loss = physics_loss + l2_lambda * l2_loss
    
    # Return detailed breakdown
    return {
        'total_loss': total_loss,
        'physics_loss': physics_loss,
        'l2_loss': l2_loss,
        
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

# Function to run experiment with specific L2 lambda
def run_l2_experiment(l2_lambda, experiment_name, num_epochs=5000):
    print(f"\n=== Running {experiment_name} (L2 lambda: {l2_lambda}) ===")
    
    # Create new model
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Initialize tracking arrays
    epochs = []
    residual_history = {
        'total_loss': [],
        'physics_loss': [],
        'l2_loss': [],
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
    
    t0 = time()
    
    for i in range(num_epochs + 1):
        optimizer.zero_grad()
        loss_components = compute_detailed_loss_with_l2(model, l2_lambda)
        total_loss = loss_components['total_loss']
        total_loss.backward()
        optimizer.step()
        
        # Track residuals every 10 epochs
        if i % 10 == 0:
            epochs.append(i)
            for key in residual_history.keys():
                residual_history[key].append(loss_components[key].item())
        
        if i % 200 == 0:
            print('It {:05d}: total_loss = {:10.8e}, physics_loss = {:10.8e}, l2_loss = {:10.8e}'.format(
                i, loss_components['total_loss'].item(), loss_components['physics_loss'].item(), 
                loss_components['l2_loss'].item()))
        
        # Learning rate schedule
        if i == 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3
        elif i == 3000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-4
    
    training_time = time() - t0
    print(f'Computation time: {training_time:.2f} seconds')
    
    # Save model
    torch.save(model.state_dict(), f'model_{experiment_name}.pt')
    
    # Save residual data
    residual_data = {
        'epochs': np.array(epochs),
        'residuals': {key: np.array(val) for key, val in residual_history.items()},
        'l2_lambda': l2_lambda,
        'experiment_name': experiment_name,
        'training_time': training_time
    }
    np.save(f'residuals_{experiment_name}.npy', residual_data, allow_pickle=True)
    
    return epochs, residual_history, model, training_time

# Run experiments with different L2 regularization values
l2_lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
experiment_names = ['baseline', 'l2_1e-6', 'l2_1e-5', 'l2_1e-4', 'l2_1e-3']

results = {}

for l2_lambda, exp_name in zip(l2_lambdas, experiment_names):
    epochs, residual_history, model, training_time = run_l2_experiment(l2_lambda, exp_name)
    results[exp_name] = {
        'l2_lambda': l2_lambda,
        'epochs': epochs,
        'residuals': residual_history,
        'model': model,
        'training_time': training_time
    }

# Plot comparison of loss histories
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot total loss
for exp_name, result in results.items():
    axes[0].semilogy(result['epochs'], result['residuals']['total_loss'], 
                     label=f"{exp_name} (λ={result['l2_lambda']})")
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Total Loss')
axes[0].set_title('Total Loss Comparison')
axes[0].legend()
axes[0].grid(True)

# Plot physics loss only
for exp_name, result in results.items():
    axes[1].semilogy(result['epochs'], result['residuals']['physics_loss'], 
                     label=f"{exp_name} (λ={result['l2_lambda']})")
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Physics Loss')
axes[1].set_title('Physics Loss Comparison')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('l2_regularization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary table
print("\n=== L2 Regularization Experiment Summary ===")
print(f"{'Experiment':<15} {'L2 Lambda':<12} {'Final Total Loss':<18} {'Final Physics Loss':<20} {'Training Time (s)':<18}")
print("-" * 85)

for exp_name, result in results.items():
    final_total = result['residuals']['total_loss'][-1]
    final_physics = result['residuals']['physics_loss'][-1]
    print(f"{exp_name:<15} {result['l2_lambda']:<12.0e} {final_total:<18.6e} {final_physics:<20.6e} {result['training_time']:<18.2f}")

print("\nL2 Regularization experiments completed!")
print("Results saved:")
print("- Loss comparison plot: l2_regularization_comparison.png")
print("- Model checkpoints: model_[experiment_name].pt")
print("- Residual data: residuals_[experiment_name].npy")