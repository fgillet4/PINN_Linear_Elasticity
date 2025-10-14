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

# Generate validation data
print("Generating validation points...")
N_bound_val = 25
N_r_val = 500

x_up_val = lhs(1, samples=N_bound_val, random_state=456)
x_up_val = xmin + (xmax-xmin)*x_up_val
y_up_val = np.full((len(x_up_val), 1), ymax)
x_up_val_data = torch.tensor(np.hstack((x_up_val, y_up_val)), dtype=DTYPE, device=device)
eux_up_val_data = torch.zeros((len(x_up_val), 1), dtype=DTYPE, device=device)
Syy_up_val_data = fun_b_yy(torch.tensor(x_up_val, dtype=DTYPE), torch.tensor(y_up_val, dtype=DTYPE))

x_lo_val = lhs(1, samples=N_bound_val, random_state=456)
x_lo_val = xmin + (xmax-xmin)*x_lo_val
y_lo_val = np.full((len(x_lo_val), 1), ymin)
x_lo_val_data = torch.tensor(np.hstack((x_lo_val, y_lo_val)), dtype=DTYPE, device=device)
eux_lo_val_data = torch.zeros((len(x_lo_val), 1), dtype=DTYPE, device=device)
uy_lo_val_data = torch.zeros((len(x_lo_val), 1), dtype=DTYPE, device=device)

y_ri_val = lhs(1, samples=N_bound_val, random_state=456)
y_ri_val = ymin + (ymax-ymin)*y_ri_val
x_ri_val = np.full((len(y_ri_val), 1), xmax)
x_ri_val_data = torch.tensor(np.hstack((x_ri_val, y_ri_val)), dtype=DTYPE, device=device)
uy_ri_val_data = torch.zeros((len(x_ri_val), 1), dtype=DTYPE, device=device)
Sxx_ri_val_data = torch.zeros((len(x_ri_val), 1), dtype=DTYPE, device=device)

y_le_val = lhs(1, samples=N_bound_val, random_state=456)
y_le_val = ymin + (ymax-ymin)*y_le_val
x_le_val = np.full((len(y_le_val), 1), xmin)
x_le_val_data = torch.tensor(np.hstack((x_le_val, y_le_val)), dtype=DTYPE, device=device)
uy_le_val_data = torch.zeros((len(x_le_val), 1), dtype=DTYPE, device=device)
Sxx_le_val_data = torch.zeros((len(x_le_val), 1), dtype=DTYPE, device=device)

eux_b_val = torch.cat([eux_up_val_data, eux_lo_val_data], 0)
uy_b_val = torch.cat([uy_lo_val_data, uy_ri_val_data, uy_le_val_data], 0)
Sxx_b_val = torch.cat([Sxx_ri_val_data, Sxx_le_val_data], 0)
Syy_b_val = Syy_up_val_data

grid_pt_val = lhs(2, N_r_val, random_state=789)
grid_pt_val[:, 0] = xmin + (xmax-xmin)*grid_pt_val[:, 0]
grid_pt_val[:, 1] = ymin + (ymax-ymin)*grid_pt_val[:, 1]
X_col_val = torch.tensor(grid_pt_val, dtype=DTYPE, device=device)

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

# Loss computation
def compute_loss(X_col, x_up_data, x_lo_data, x_ri_data, x_le_data, 
                 eux_b, uy_b, Sxx_b, Syy_b, compute_gradients=True):
    # Physics loss
    if compute_gradients:
        X_col_req = X_col.requires_grad_(True)
    else:
        X_col_req = X_col.detach().requires_grad_(True)
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
    Ux_up, Uy_up, _, Syy_up, _ = model(x_up_data)
    Ux_lo, Uy_lo, _, _, _ = model(x_lo_data)
    _, Uy_ri, Sxx_ri, _, _ = model(x_ri_data)
    _, Uy_le, Sxx_le, _, _ = model(x_le_data)
    
    r_ux = torch.cat([Ux_up, Ux_lo], 0) - eux_b
    r_uy = torch.cat([Uy_lo, Uy_ri, Uy_le], 0) - uy_b
    r_Sxx = torch.cat([Sxx_ri, Sxx_le], 0) - Sxx_b
    r_Syy = Syy_up - Syy_b
    
    phi_r_u = torch.mean(torch.abs(r_ux)) + torch.mean(torch.abs(r_uy))
    phi_r_S = torch.mean(torch.abs(r_Sxx)) + torch.mean(torch.abs(r_Syy))
    
    total_loss = phi_r + phi_r_const + phi_r_u + phi_r_S
    return total_loss

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

print("Starting training...")
N = 10000
train_hist = []
val_hist = []
t0 = time()

for i in range(N + 1):
    model.train()
    optimizer.zero_grad()
    loss = compute_loss(X_col_train, x_up_train, x_lo_train, x_ri_train, x_le_train,
                        eux_b_train, uy_b_train, Sxx_b_train, Syy_b_train)
    loss.backward()
    optimizer.step()
    
    if i == 0:
        loss0 = loss.item()
    
    train_hist.append(loss.item() / loss0)
    
    # Compute validation loss
    model.eval()
    val_loss = compute_loss(X_col_val, x_up_val_data, x_lo_val_data, x_ri_val_data, x_le_val_data,
                            eux_b_val, uy_b_val, Sxx_b_val, Syy_b_val, compute_gradients=True)
    val_hist.append(val_loss.item() / loss0)
    
    if i % 50 == 0:
        print('It {:05d}: train_loss = {:10.8e}, val_loss = {:10.8e}'.format(i, loss.item(), val_loss.item()))
    
    # Learning rate schedule
    if i == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
    elif i == 3000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-4

print('\nComputation time: {} seconds'.format(time() - t0))

# Evaluation
print("Evaluating model...")
N_plot = 600
xspace = np.linspace(0, 1, N_plot + 1)
yspace = np.linspace(0, 1, N_plot + 1)
X, Y = np.meshgrid(xspace, yspace)
Xgrid_tensor = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T, dtype=DTYPE, device=device)

model.eval()
with torch.no_grad():
    Ux_pred, Uy_pred, Sxx_pred, Syy_pred, Sxy_pred = model(Xgrid_tensor)

# Reshape for plotting
Ux = Ux_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Uy = Uy_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Sxx = Sxx_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Syy = Syy_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Sxy = Sxy_pred.cpu().numpy().reshape(N_plot + 1, N_plot + 1)

# Plot displacement components first
U_total = [Ux, Uy]
U_total_name = ['Ux_NN_PyTorch', 'Uy_NN_PyTorch']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, U_total[i], cmap='seismic', vmin=-0.8, vmax=0.8)
    ax.set_title(U_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
fig.savefig('results/solution_pytorch_with_validation.png')

# Calculate exact solutions for comparison
ux_ext_flat = u_x_ext(torch.tensor(X.flatten(), dtype=DTYPE), torch.tensor(Y.flatten(), dtype=DTYPE))
uy_ext_flat = u_y_ext(torch.tensor(X.flatten(), dtype=DTYPE), torch.tensor(Y.flatten(), dtype=DTYPE))

# Reshape exact solutions
Ux_ext = ux_ext_flat.cpu().numpy().reshape(N_plot + 1, N_plot + 1)
Uy_ext = uy_ext_flat.cpu().numpy().reshape(N_plot + 1, N_plot + 1)

# Plot errors
error_total = [abs(Ux - Ux_ext), abs(Uy - Uy_ext)]
error_total_name = ['point wise error Ux', 'point wise error Uy']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, error_total[i], cmap='jet')
    ax.set_title(error_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
fig.savefig('results/error_map_pytorch_with_validation.png')

# Plot stress components
S_total = [Sxx, Syy, Sxy]
S_total_name = ['Sxx_NN_PyTorch', 'Syy_NN_PyTorch', 'Sxy_NN_PyTorch']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
for i, ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, S_total[i], cmap='seismic', vmin=-10, vmax=10)
    ax.set_title(S_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
fig.savefig('results/stress_map_pytorch_with_validation.png')

# Plot loss history with training and validation
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
ax.semilogy(range(len(train_hist)), train_hist, 'b-', label='Training Loss')
ax.semilogy(range(len(val_hist)), val_hist, 'r-', label='Validation Loss')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig('results/loss_history_pytorch_with_validation.png')

# Save model
torch.save(model.state_dict(), 'solidmechanics_model_pytorch_with_validation.pt')
print("PyTorch PINN training with validation completed successfully!")
