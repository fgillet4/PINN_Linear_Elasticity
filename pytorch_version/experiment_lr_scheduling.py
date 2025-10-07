# Learning Rate Scheduling Experiment for PINN
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

# Loss computation
def compute_loss(model):
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

# Function to run experiment with specific learning rate schedule
def run_lr_experiment(schedule_type, schedule_params, experiment_name, num_epochs=5000):
    print(f"\n=== Running {experiment_name} ===")
    
    # Create new model
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=schedule_params.get('initial_lr', 1e-2))
    
    # Create scheduler based on type
    if schedule_type == 'constant':
        scheduler = None
    elif schedule_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=schedule_params['step_size'], 
            gamma=schedule_params['gamma']
        )
    elif schedule_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=schedule_params['gamma']
        )
    elif schedule_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=schedule_params['T_max']
        )
    elif schedule_type == 'cosine_warm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=schedule_params['T_0'],
            T_mult=schedule_params.get('T_mult', 1)
        )
    elif schedule_type == 'reduce_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=schedule_params['factor'],
            patience=schedule_params['patience']
        )
    elif schedule_type == 'manual':
        scheduler = None  # Will be handled manually
    else:
        scheduler = None
    
    hist_loss = []
    hist_lr = []
    t0 = time()
    
    for i in range(num_epochs + 1):
        optimizer.zero_grad()
        loss = compute_loss(model)
        loss.backward()
        optimizer.step()
        
        if i == 0:
            loss0 = loss.item()
        
        hist_loss.append(loss.item() / loss0)
        hist_lr.append(optimizer.param_groups[0]['lr'])
        
        if i % 200 == 0:
            print('It {:05d}: loss = {:10.8e}, lr = {:10.8e}'.format(
            i, loss.item(), optimizer.param_groups[0]['lr']))
        
        # Apply scheduler
        if schedule_type == 'manual':
            # Original manual schedule from notes
            if i == 1000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-3
            elif i == 3000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5e-4
        elif schedule_type == 'reduce_plateau':
            scheduler.step(loss)
        elif scheduler is not None:
            scheduler.step()
    
    training_time = time() - t0
    print(f'Computation time: {training_time:.2f} seconds')
    
    # Save model
    torch.save(model.state_dict(), f'model_{experiment_name}.pt')
    
    return hist_loss, hist_lr, model, training_time

# Define different learning rate schedules to test
schedules = {
    'constant': {
        'type': 'constant',
        'params': {'initial_lr': 1e-2},
        'name': 'constant_lr'
    },
    'manual': {
        'type': 'manual',
        'params': {'initial_lr': 1e-2},
        'name': 'manual_schedule'
    },
    'step': {
        'type': 'step',
        'params': {'initial_lr': 1e-2, 'step_size': 1000, 'gamma': 0.5},
        'name': 'step_lr'
    },
    'exponential': {
        'type': 'exponential',
        'params': {'initial_lr': 1e-2, 'gamma': 0.9995},
        'name': 'exponential_lr'
    },
    'cosine': {
        'type': 'cosine',
        'params': {'initial_lr': 1e-2, 'T_max': 5000},
        'name': 'cosine_lr'
    },
    'cosine_warm': {
        'type': 'cosine_warm',
        'params': {'initial_lr': 1e-2, 'T_0': 1000, 'T_mult': 2},
        'name': 'cosine_warm_restart'
    },
    'reduce_plateau': {
        'type': 'reduce_plateau',
        'params': {'initial_lr': 1e-2, 'factorience': 500},
        'name': 'reduce_on_plateau'
    }
}

results = {}

# Run experiments
for key, schedule in schedules.items():
    hist_loss, hist_lr, model, training_time = run_lr_experiment(
        schedule['type'], 
        schedule['params'], 
        schedule['name']
    )
    res[key] = {
        'schedule': schedule,
        'hist_loss': hist_loss,
        'hist_lr': hist_lr,
        'model': model,
        'training_time': training_time
    }

# Plot comparison of learning rate schedules
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot loss histories
for key, result in results.items():
    axes[0, 0].semilogy(range(len(result['hist_loss'])), result['hist_loss'], 
                        label=f"{result['schedule']['name']}")
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Normalized Loss')
axes[0, 0].set_title('Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot learning rate schedules
for key, result in results.items():
    axes[0, 1].semilogy(range(len(result['hist_lr'])), result['hist_lr'], 
                        label=f"{result['schedule']['name']}")
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_title('Learning Rate Schedules')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot final 1000 epochs for better detail
for key, result in results.items():
    start_idx = max(0, len(result['hist_loss']) - 1000)
    axes[1, 0].semilogy(range(start_idx, len(result['hist_loss'])), 
                        result['hist_loss'][start_idx:], 
                        label=f"{result['schedule']['name']}")
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Normalized Loss')
axes[1, 0].set_title('Loss Comparison (Final 1000 epochs)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot convergence rate (loss improvement per epoch)
for key, result in results.items():
    loss_diff = np.diff(result['hist_loss'])
    axes[1, 1].plot(range(len(loss_diff)), -loss_diff, 
                    label=f"{result['schedule']['name']}", alpha=0.7)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss Improvement per Epoch')
axes[1, 1].set_title('Convergence Rate')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('lr_scheduling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary table
print("\n=== Learning Rate Scheduling Experiment Summary ===")
print(f"{'Schedule':<20} {'Final Loss':<15} {'Min Loss':<15} {'Training Time (s)':<18}")
print("-" * 70)

for key, result in results.items():
    final_loss = result['hist_loss'][-1] * (result['hist_loss'][0] if len(result['hist_loss']) > 0 else 1)
    min_loss = min(result['hist_loss']) * (result['hist_loss'][0] if len(result['hist_loss']) > 0 else 1)
    print(f"{result['schedule']['name']:<20} {final_loss:<15.6e} {min_loss:<15.6e} {result['training_time']:<18.2f}")

print("\nLearning Rate Scheduling experiments completed!")
print("Results saved:")
print("- Comparison plot: lr_scheduling_comparison.png")
print("- Model checkpoints: model_[schedule_name].pt")