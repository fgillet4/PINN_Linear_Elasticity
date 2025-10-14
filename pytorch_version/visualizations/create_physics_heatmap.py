#!/usr/bin/env python3
"""
Physics Heat Map Visualizer for PINN
Shows where the network is "thinking" most by tracking gradient flow and comprehensive residual analysis
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import os
import glob
from visualize_network_graph import NetworkVisualizer
from PIL import Image
import matplotlib.patches as mpatches

# Copy the physics constants and functions from the original training script
device = torch.device('cpu')
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

# Physics constants
pi = torch.tensor(np.pi, dtype=DTYPE, device=device)
E = torch.tensor(4e11/3, dtype=DTYPE, device=device) / 1e11   # Young's modulus (normalized)
v = torch.tensor(1/3, dtype=DTYPE, device=device)            # Poisson's ratio
lmda = E*v/(1-2*v)/(1+v)                                    # Lamé parameter
mu = E/(2*(1+v))                                            # Shear modulus
Q = torch.tensor(4.0, dtype=DTYPE, device=device)

# Domain bounds
xmin, xmax = 0., 1.
ymin, ymax = 0., 1.
lb = torch.tensor([xmin, ymin], dtype=DTYPE, device=device)
ub = torch.tensor([xmax, ymax], dtype=DTYPE, device=device)

# Exact solutions for comparison
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

# Reconstruct the PINN model class
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

class PhysicsHeatMapVisualizer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        self.epochs = []
        
    def load_model_at_epoch(self, epoch_file):
        """Load a specific model snapshot"""
        model = PINN().to(device)
        model.load_state_dict(torch.load(epoch_file, map_location=device))
        model.eval()
        return model
    
    def compute_gradient_attention(self, model, grid_points):
        """Compute where the network is 'paying attention' by tracking gradient magnitudes"""
        
        X_test = torch.tensor(grid_points, dtype=DTYPE, device=device, requires_grad=True)
        
        # Forward pass
        Ux, Uy, Sxx, Syy, Sxy = model(X_test)
        
        # Compute total output magnitude (overall network activity)
        total_output = torch.abs(Ux) + torch.abs(Uy) + torch.abs(Sxx) + torch.abs(Syy) + torch.abs(Sxy)
        
        # Compute gradients of total output w.r.t. inputs
        grad_attention = torch.autograd.grad(
            outputs=total_output.sum(), 
            inputs=X_test, 
            create_graph=False, 
            retain_graph=False
        )[0]
        
        # Gradient magnitude shows where network is "thinking" most
        attention_magnitude = torch.norm(grad_attention, dim=1)
        
        return {
            'attention': attention_magnitude.detach(),
            'grad_x': grad_attention[:, 0].detach(),
            'grad_y': grad_attention[:, 1].detach(),
            'outputs': {
                'Ux': Ux.detach().flatten(),
                'Uy': Uy.detach().flatten(), 
                'Sxx': Sxx.detach().flatten(),
                'Syy': Syy.detach().flatten(),
                'Sxy': Sxy.detach().flatten()
            }
        }
    
    def compute_comprehensive_residuals(self, model, grid_points):
        """Compute all physics residuals and boundary condition violations"""
        
        X_test = torch.tensor(grid_points, dtype=DTYPE, device=device, requires_grad=True)
        Ux, Uy, Sxx, Syy, Sxy = model(X_test)
        
        x = X_test[:, 0:1]
        y = X_test[:, 1:2]
        
        # === PHYSICS EQUATION RESIDUALS ===
        
        # Stress gradients for equilibrium equations
        grad_Sxx = torch.autograd.grad(Sxx, X_test, grad_outputs=torch.ones_like(Sxx), create_graph=True, retain_graph=True)[0]
        grad_Syy = torch.autograd.grad(Syy, X_test, grad_outputs=torch.ones_like(Syy), create_graph=True, retain_graph=True)[0]
        grad_Sxy = torch.autograd.grad(Sxy, X_test, grad_outputs=torch.ones_like(Sxy), create_graph=True, retain_graph=True)[0]
        
        dsxxdx = grad_Sxx[:, 0:1]
        dsyydy = grad_Syy[:, 1:2]
        dsxydx = grad_Sxy[:, 0:1]
        dsxydy = grad_Sxy[:, 1:2]
        
        # Equilibrium equations
        pde_x_residual = dsxxdx + dsxydy - f_x_ext(x, y)
        pde_y_residual = dsxydx + dsyydy - f_y_ext(x, y)
        
        # Displacement gradients for constitutive equations
        grad_Ux = torch.autograd.grad(Ux, X_test, grad_outputs=torch.ones_like(Ux), create_graph=True, retain_graph=True)[0]
        grad_Uy = torch.autograd.grad(Uy, X_test, grad_outputs=torch.ones_like(Uy), create_graph=True, retain_graph=True)[0]
        
        duxdx = grad_Ux[:, 0:1]
        duxdy = grad_Ux[:, 1:2]
        duydx = grad_Uy[:, 0:1]
        duydy = grad_Uy[:, 1:2]
        
        # Constitutive equations
        const_xx_residual = (lmda + 2*mu)*duxdx + lmda*duydy - Sxx
        const_yy_residual = (lmda + 2*mu)*duydy + lmda*duxdx - Syy
        const_xy_residual = 2*mu*0.5*(duxdy + duydx) - Sxy
        
        # === BOUNDARY CONDITION VIOLATIONS ===
        
        # Identify boundary points
        tol = 1e-6
        
        # Bottom boundary (y = 0): Ux = 0, Uy = 0
        bottom_mask = torch.abs(y.flatten()) < tol
        bc_bottom_ux = torch.zeros_like(Ux.flatten())
        bc_bottom_uy = torch.zeros_like(Uy.flatten())
        bc_bottom_ux[bottom_mask] = torch.abs(Ux.flatten()[bottom_mask])
        bc_bottom_uy[bottom_mask] = torch.abs(Uy.flatten()[bottom_mask])
        
        # Top boundary (y = 1): Ux = 0, Syy = prescribed
        top_mask = torch.abs(y.flatten() - 1.0) < tol
        bc_top_ux = torch.zeros_like(Ux.flatten())
        bc_top_syy = torch.zeros_like(Syy.flatten())
        bc_top_ux[top_mask] = torch.abs(Ux.flatten()[top_mask])
        if torch.any(top_mask):
            x_top = x.flatten()[top_mask]
            y_top = y.flatten()[top_mask]
            prescribed_syy = fun_b_yy(x_top, y_top)
            bc_top_syy[top_mask] = torch.abs(Syy.flatten()[top_mask] - prescribed_syy)
        
        # Left boundary (x = 0): Uy = 0, Sxx = 0
        left_mask = torch.abs(x.flatten()) < tol
        bc_left_uy = torch.zeros_like(Uy.flatten())
        bc_left_sxx = torch.zeros_like(Sxx.flatten())
        bc_left_uy[left_mask] = torch.abs(Uy.flatten()[left_mask])
        bc_left_sxx[left_mask] = torch.abs(Sxx.flatten()[left_mask])
        
        # Right boundary (x = 1): Uy = 0, Sxx = 0
        right_mask = torch.abs(x.flatten() - 1.0) < tol
        bc_right_uy = torch.zeros_like(Uy.flatten())
        bc_right_sxx = torch.zeros_like(Sxx.flatten())
        bc_right_uy[right_mask] = torch.abs(Uy.flatten()[right_mask])
        bc_right_sxx[right_mask] = torch.abs(Sxx.flatten()[right_mask])
        
        # === SOLUTION ACCURACY ===
        
        # Exact solution comparison
        ux_exact = u_x_ext(x, y)
        uy_exact = u_y_ext(x, y)
        solution_error_x = torch.abs(Ux - ux_exact)
        solution_error_y = torch.abs(Uy - uy_exact)
        
        return {
            # Physics equations
            'pde_x': torch.abs(pde_x_residual).detach().flatten(),
            'pde_y': torch.abs(pde_y_residual).detach().flatten(),
            'constitutive_xx': torch.abs(const_xx_residual).detach().flatten(),
            'constitutive_yy': torch.abs(const_yy_residual).detach().flatten(),
            'constitutive_xy': torch.abs(const_xy_residual).detach().flatten(),
            
            # Boundary conditions
            'bc_bottom_ux': bc_bottom_ux.detach(),
            'bc_bottom_uy': bc_bottom_uy.detach(),
            'bc_top_ux': bc_top_ux.detach(),
            'bc_top_syy': bc_top_syy.detach(),
            'bc_left_uy': bc_left_uy.detach(),
            'bc_left_sxx': bc_left_sxx.detach(),
            'bc_right_uy': bc_right_uy.detach(),
            'bc_right_sxx': bc_right_sxx.detach(),
            
            # Solution accuracy
            'solution_error_x': solution_error_x.detach().flatten(),
            'solution_error_y': solution_error_y.detach().flatten(),
            
            # Combined metrics
            'total_physics': (torch.abs(pde_x_residual) + torch.abs(pde_y_residual) + 
                            torch.abs(const_xx_residual) + torch.abs(const_yy_residual) + 
                            torch.abs(const_xy_residual)).detach().flatten(),
            'total_boundary': (bc_bottom_ux + bc_bottom_uy + bc_top_ux + bc_top_syy + 
                             bc_left_uy + bc_left_sxx + bc_right_uy + bc_right_sxx).detach()
        }
    
    def create_comprehensive_heatmap(self, epoch_file, grid_resolution=60):
        """Create comprehensive heat map showing attention and all residuals"""
        
        epoch = int(epoch_file.split('_')[-1].split('.')[0])
        print(f"🔥 Creating comprehensive heat map for epoch {epoch}")
        
        # Load model
        model = self.load_model_at_epoch(epoch_file)
        
        # Create test grid
        x_test = np.linspace(xmin, xmax, grid_resolution)
        y_test = np.linspace(ymin, ymax, grid_resolution)
        X_mesh, Y_mesh = np.meshgrid(x_test, y_test)
        
        # Flatten for model input
        x_flat = X_mesh.flatten()
        y_flat = Y_mesh.flatten()
        grid_points = np.column_stack([x_flat, y_flat])
        
        # Compute gradient attention
        attention_data = self.compute_gradient_attention(model, grid_points)
        
        # Compute comprehensive residuals
        residual_data = self.compute_comprehensive_residuals(model, grid_points)
        
        # Reshape all data back to grids
        def reshape_to_grid(data):
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy().reshape(grid_resolution, grid_resolution)
            return data.reshape(grid_resolution, grid_resolution)
        
        # Create massive visualization
        fig = plt.figure(figsize=(24, 20))
        fig.suptitle(f'Comprehensive PINN Physics Analysis - Epoch {epoch}\n'
                    f'Neural Attention + Complete Residual Analysis', 
                    fontsize=18, fontweight='bold')
        
        # Define all visualizations
        visualizations = [
            # Row 1: Attention and gradients
            ('attention', 'Network Attention\n|∇(Total Output)|', 'viridis', attention_data['attention']),
            ('grad_x', 'X-Gradient\n∂(Total)/∂x', 'RdBu_r', attention_data['grad_x']),
            ('grad_y', 'Y-Gradient\n∂(Total)/∂y', 'RdBu_r', attention_data['grad_y']),
            ('outputs_ux', 'Prediction Ux', 'coolwarm', attention_data['outputs']['Ux']),
            
            # Row 2: Physics equations
            ('pde_x', 'Equilibrium X\n|∂σₓₓ/∂x + ∂σₓᵧ/∂y - fₓ|', 'Reds', residual_data['pde_x']),
            ('pde_y', 'Equilibrium Y\n|∂σₓᵧ/∂x + ∂σᵧᵧ/∂y - fᵧ|', 'Reds', residual_data['pde_y']),
            ('const_xx', 'Constitutive XX\n|σₓₓ - (λ+2μ)εₓₓ - λεᵧᵧ|', 'Blues', residual_data['constitutive_xx']),
            ('const_yy', 'Constitutive YY\n|σᵧᵧ - (λ+2μ)εᵧᵧ - λεₓₓ|', 'Blues', residual_data['constitutive_yy']),
            
            # Row 3: More physics and boundaries
            ('const_xy', 'Constitutive XY\n|σₓᵧ - 2μεₓᵧ|', 'Blues', residual_data['constitutive_xy']),
            ('bc_violations', 'Boundary Violations\nCombined BC Errors', 'Oranges', residual_data['total_boundary']),
            ('solution_error', 'Solution Error X\n|Ux - Ux_exact|', 'YlOrRd', residual_data['solution_error_x']),
            ('total_physics', 'Total Physics Error\nCombined PDE+Constitutive', 'plasma', residual_data['total_physics']),
            
            # Row 4: Additional outputs
            ('outputs_uy', 'Prediction Uy', 'coolwarm', attention_data['outputs']['Uy']),
            ('outputs_sxx', 'Prediction σₐₓ', 'seismic', attention_data['outputs']['Sxx']),
            ('outputs_syy', 'Prediction σᵧᵧ', 'seismic', attention_data['outputs']['Syy']),
            ('outputs_sxy', 'Prediction σₓᵧ', 'seismic', attention_data['outputs']['Sxy'])
        ]
        
        # Create grid of subplots (4x4)
        for i, (key, title, cmap, data) in enumerate(visualizations):
            if i >= 16:  # Max 16 subplots
                break
                
            row = i // 4
            col = i % 4
            ax = plt.subplot(4, 4, i + 1)
            
            # Reshape data
            plot_data = reshape_to_grid(data)
            
            # Use log scale for residuals (but not for raw outputs)
            if 'residual' in key or 'pde' in key or 'const' in key or 'bc' in key or 'error' in key or 'physics' in key:
                plot_data = np.log10(np.abs(plot_data) + 1e-12)
                label_suffix = ' (log₁₀)'
            else:
                label_suffix = ''
            
            # Create contour plot
            if np.isfinite(plot_data).all() and plot_data.std() > 1e-10:
                num_levels = 20
                contour = ax.contourf(X_mesh, Y_mesh, plot_data, levels=num_levels, cmap=cmap, alpha=0.8)
                
                # Add contour lines for better visibility (use fewer levels)
                contour_levels = np.linspace(plot_data.min(), plot_data.max(), 5)
                ax.contour(X_mesh, Y_mesh, plot_data, levels=contour_levels, colors='black', alpha=0.3, linewidths=0.5)
                
                # Colorbar
                cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
                cbar.set_label(label_suffix, rotation=270, labelpad=10, fontsize=8)
                cbar.ax.tick_params(labelsize=8)
            else:
                # Fallback for problematic data
                ax.imshow(plot_data, cmap=cmap, aspect='auto', extent=[xmin, xmax, ymin, ymax])
            
            # Styling
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_aspect('equal')
            
            # Add domain boundary
            boundary = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                               fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(boundary)
        
        plt.tight_layout()
        return fig
    
    def create_heatmap_animation(self, grid_resolution=50):
        """Create heat map animation"""
        print("🔥 Creating comprehensive physics heat map animation...")
        
        # Find all model snapshots
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found")
            return
        
        os.makedirs('heatmap_frames', exist_ok=True)
        
        # Create frames (subsample for performance)
        step = max(1, len(model_files) // 50)  # Max 50 frames
        selected_files = model_files[::step]
        
        for i, model_file in enumerate(selected_files):
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            try:
                fig = self.create_comprehensive_heatmap(model_file, grid_resolution)
                plt.savefig(f'heatmap_frames/heatmap_frame_{i:04d}.png', 
                           dpi=80, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
            except Exception as e:
                print(f"⚠️ Error processing epoch {epoch}: {e}")
                continue
        
        # Create GIF
        print("🎬 Creating physics heat map GIF...")
        frame_files = sorted(glob.glob('heatmap_frames/heatmap_frame_*.png'))
        
        if frame_files:
            images = [Image.open(f) for f in frame_files]
            images[0].save('animations/physics_heatmap_comprehensive.gif', save_all=True,
                          append_images=images[1:], duration=400, loop=0)
            print("✅ Comprehensive physics heat map GIF created!")
            
            # Fast version
            fast_images = images[::2]
            if len(fast_images) > 1:
                fast_images[0].save('animations/physics_heatmap_fast.gif', save_all=True,
                                  append_images=fast_images[1:], duration=300, loop=0)
                print("✅ Fast physics heat map GIF created!")

def main():
    """Main function"""
    print("🔥 Comprehensive Physics Heat Map Visualizer")
    print("=" * 60)
    
    visualizer = PhysicsHeatMapVisualizer()
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    print("\n🔥 Creating comprehensive physics heat map visualization...")
    visualizer.create_heatmap_animation(grid_resolution=40)
    
    print("\n🎉 Physics heat map visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/physics_heatmap_comprehensive.gif - Complete analysis")
    print("   📂 animations/physics_heatmap_fast.gif - Faster version")
    print("   📂 heatmap_frames/ - Individual frames")
    print("\n🔥 This shows WHERE your PINN is thinking and ALL physics violations!")

if __name__ == "__main__":
    main()