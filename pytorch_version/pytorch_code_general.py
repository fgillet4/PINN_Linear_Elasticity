import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

#%% Neural network definition
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=4, neurons=50):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, neurons), nn.Tanh()]
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
# Separate NNs for displacement and stress
ux_net  = MLP()
uy_net  = MLP()
sxx_net = MLP()
syy_net = MLP()
sxy_net = MLP()

#%% Collocation points
N_col = 1000
x = torch.rand(N_col, 1, requires_grad=True)
y = torch.rand(N_col, 1, requires_grad=True)
xy = torch.cat([x, y], dim=1)

#%% Helper function for gradients
def gradients(u, x, order=1):
    return autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

#%% Material parameters
lam = torch.tensor(1.0)
mu  = torch.tensor(0.5)

#%% Boundary points (training)
N_bc = 200
xb = torch.rand(N_bc,1)
yb = torch.rand(N_bc,1)

left = torch.cat([torch.zeros_like(yb), yb], dim=1)
bottom = torch.cat([xb, torch.zeros_like(xb)], dim=1)

#%% Validation points
N_val = 500
x_val = torch.rand(N_val, 1, requires_grad=True)
y_val = torch.rand(N_val, 1, requires_grad=True)
xy_val = torch.cat([x_val, y_val], dim=1)

#%% Training Loop
params = list(ux_net.parameters()) + list(uy_net.parameters()) + \
         list(sxx_net.parameters()) + list(syy_net.parameters()) + list(sxy_net.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
loss_history = []

for epoch in range(2000):
    optimizer.zero_grad()
    
    # --- Forward pass ---
    ux = ux_net(xy)
    uy = uy_net(xy)
    
    sxx_nn = sxx_net(xy)
    syy_nn = syy_net(xy)
    sxy_nn = sxy_net(xy)
    
    # --- Compute strains ---
    grads_ux = gradients(ux, xy)
    grads_uy = gradients(uy, xy)
    exx = grads_ux[:,0:1]
    eyy = grads_uy[:,1:2]
    exy = 0.5*(grads_ux[:,1:2] + grads_uy[:,0:1])
    
    # --- Constitutive residual ---
    sxx_pred = lam*(exx+eyy) + 2*mu*exx
    syy_pred = lam*(exx+eyy) + 2*mu*eyy
    sxy_pred = 2*mu*exy
    res_const = ((sxx_nn - sxx_pred)**2 + (syy_nn - syy_pred)**2 + (sxy_nn - sxy_pred)**2).mean()
    
    # --- Equilibrium residual ---
    grads_sxx = gradients(sxx_nn, xy)
    grads_syy = gradients(syy_nn, xy)
    grads_sxy = gradients(sxy_nn, xy)
    fx, fy = 0.0, 0.0
    res_eq1 = grads_sxx[:,0:1] + grads_sxy[:,1:2] + fx
    res_eq2 = grads_sxy[:,0:1] + grads_syy[:,1:2] + fy
    res_equil = (res_eq1**2 + res_eq2**2).mean()
    
    # --- Boundary residuals ---
    res_bc = ((ux_net(left) - 0.0)**2).mean() + ((uy_net(bottom) - 0.0)**2).mean()
    
    # --- Total loss ---
    total_loss = res_const + res_equil + res_bc
    
    # --- Backward ---
    total_loss.backward()  # <-- only once per epoch
    optimizer.step()
    
    loss_history.append(total_loss.item())
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss {total_loss.item():.6f}")

print("Training complete.")

#%% Plot loss
plt.figure()
plt.semilogy(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss (log scale)")
plt.title("Loss vs Epochs")
plt.grid(True)
plt.savefig('loss_history_general.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Collocation points visualization
plt.figure()
plt.scatter(x.detach().numpy(), y.detach().numpy(), s=5, alpha=0.7, c='blue', label='Training points')
plt.scatter(x_val.detach().numpy(), y_val.detach().numpy(), s=5, alpha=0.7, c='red', label='Validation points')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Collocation Points Distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('collocation_points_general.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Compute predicted u_x, u_y and errors
nx, ny = 50, 50
xx = np.linspace(0,1,nx)
yy = np.linspace(0,1,ny)
XX, YY = np.meshgrid(xx, yy)
grid = torch.tensor(np.vstack([XX.ravel(), YY.ravel()]).T, dtype=torch.float32)

with torch.no_grad():
    ux_pred = ux_net(grid).reshape(ny,nx).numpy()
    uy_pred = uy_net(grid).reshape(ny,nx).numpy()

ux_exact = np.cos(2*np.pi*XX) * np.sin(np.pi*YY)
uy_exact = np.sin(np.pi*XX) * (YY**4)/4
ux_err = ux_pred - ux_exact
uy_err = uy_pred - uy_exact

# Plot predictions and errors
fig, axs = plt.subplots(2,2, figsize=(10,8))
im0 = axs[0,0].imshow(ux_pred, extent=[0,1,0,1], origin="lower"); axs[0,0].set_title("Predicted u_x"); plt.colorbar(im0, ax=axs[0,0])
im1 = axs[0,1].imshow(uy_pred, extent=[0,1,0,1], origin="lower"); axs[0,1].set_title("Predicted u_y"); plt.colorbar(im1, ax=axs[0,1])
im2 = axs[1,0].imshow(ux_err, extent=[0,1,0,1], origin="lower"); axs[1,0].set_title("Error in u_x"); plt.colorbar(im2, ax=axs[1,0])
im3 = axs[1,1].imshow(uy_err, extent=[0,1,0,1], origin="lower"); axs[1,1].set_title("Error in u_y"); plt.colorbar(im3, ax=axs[1,1])
plt.tight_layout()
plt.show()
