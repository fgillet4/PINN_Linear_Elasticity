#!/usr/bin/env python3
"""
Physics Equation Discovery Visualizer for PINN
Shows how well the network satisfies each physics equation across the domain
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

class PhysicsEquationVisualizer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        self.epochs = []
        
    def load_model_at_epoch(self, epoch_file):
        """Load a specific model snapshot"""
        model = PINN().to(device)
        model.load_state_dict(torch.load(epoch_file, map_location=device))
        model.eval()
        return model
    
    def compute_physics_residuals(self, model, grid_points):
        """Compute how well the model satisfies each physics equation"""
        
        X_test = grid_points.requires_grad_(True)
        Ux, Uy, Sxx, Syy, Sxy = model(X_test)
        
        x = X_test[:, 0:1]
        y = X_test[:, 1:2]
        
        # Compute gradients for PDE residuals
        grad_Sxx = torch.autograd.grad(Sxx, X_test, grad_outputs=torch.ones_like(Sxx), create_graph=True, retain_graph=True)[0]
        grad_Syy = torch.autograd.grad(Syy, X_test, grad_outputs=torch.ones_like(Syy), create_graph=True, retain_graph=True)[0]
        grad_Sxy = torch.autograd.grad(Sxy, X_test, grad_outputs=torch.ones_like(Sxy), create_graph=True, retain_graph=True)[0]
        
        dsxxdx = grad_Sxx[:, 0:1]
        dsyydy = grad_Syy[:, 1:2]
        dsxydx = grad_Sxy[:, 0:1]
        dsxydy = grad_Sxy[:, 1:2]
        
        # PDE residuals (equilibrium equations)
        pde_x_residual = dsxxdx + dsxydy - f_x_ext(x, y)
        pde_y_residual = dsxydx + dsyydy - f_y_ext(x, y)
        
        # Compute gradients for constitutive equations
        grad_Ux = torch.autograd.grad(Ux, X_test, grad_outputs=torch.ones_like(Ux), create_graph=True, retain_graph=True)[0]
        grad_Uy = torch.autograd.grad(Uy, X_test, grad_outputs=torch.ones_like(Uy), create_graph=True, retain_graph=True)[0]
        
        duxdx = grad_Ux[:, 0:1]
        duxdy = grad_Ux[:, 1:2]
        duydx = grad_Uy[:, 0:1]
        duydy = grad_Uy[:, 1:2]
        
        # Constitutive equation residuals
        const_xx_residual = (lmda + 2*mu)*duxdx + lmda*duydy - Sxx
        const_yy_residual = (lmda + 2*mu)*duydy + lmda*duxdx - Syy
        const_xy_residual = 2*mu*0.5*(duxdy + duydx) - Sxy
        
        # Displacement prediction vs exact
        ux_exact = u_x_ext(x, y)
        uy_exact = u_y_ext(x, y)
        displacement_error_x = Ux - ux_exact
        displacement_error_y = Uy - uy_exact
        
        return {
            'pde_x': pde_x_residual.detach(),
            'pde_y': pde_y_residual.detach(),
            'constitutive_xx': const_xx_residual.detach(),
            'constitutive_yy': const_yy_residual.detach(),
            'constitutive_xy': const_xy_residual.detach(),
            'displacement_error_x': displacement_error_x.detach(),
            'displacement_error_y': displacement_error_y.detach()
        }
    
    def create_physics_frame(self, epoch_file, grid_resolution=50):
        """Create a single frame showing physics equation satisfaction"""
        
        epoch = int(epoch_file.split('_')[-1].split('.')[0])
        print(f"⚖️ Analyzing physics equations for epoch {epoch}")
        
        # Load model
        model = self.load_model_at_epoch(epoch_file)
        
        # Create test grid over the domain
        x_test = np.linspace(xmin, xmax, grid_resolution)
        y_test = np.linspace(ymin, ymax, grid_resolution)
        X_mesh, Y_mesh = np.meshgrid(x_test, y_test)
        
        # Flatten for model input
        x_flat = X_mesh.flatten()
        y_flat = Y_mesh.flatten()
        grid_points = torch.tensor(np.column_stack([x_flat, y_flat]), dtype=DTYPE, device=device)
        
        # Compute physics residuals (need gradients enabled)
        residuals = self.compute_physics_residuals(model, grid_points)
        
        # Reshape residuals back to grid
        reshaped_residuals = {}
        for key, values in residuals.items():
            reshaped_residuals[key] = values.cpu().numpy().reshape(grid_resolution, grid_resolution)
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle(f'Physics Equation Discovery - Epoch {epoch}\n'
                    f'Red = Poor Satisfaction, Blue = Good Satisfaction', 
                    fontsize=16, fontweight='bold')
        
        # Define equation types and their properties
        equations = [
            ('pde_x', 'Equilibrium X\n∂σₓₓ/∂x + ∂σₓᵧ/∂y = fₓ', 'Reds'),
            ('pde_y', 'Equilibrium Y\n∂σₓᵧ/∂x + ∂σᵧᵧ/∂y = fᵧ', 'Reds'),
            ('constitutive_xx', 'Constitutive XX\nσₓₓ = (λ+2μ)εₓₓ + λεᵧᵧ', 'Blues'),
            ('constitutive_yy', 'Constitutive YY\nσᵧᵧ = (λ+2μ)εᵧᵧ + λεₓₓ', 'Blues'),
            ('constitutive_xy', 'Constitutive XY\nσₓᵧ = 2μεₓᵧ', 'Blues'),
            ('displacement_error_x', 'Displacement Error X\nUₓ - Uₓ_exact', 'RdYlBu'),
            ('displacement_error_y', 'Displacement Error Y\nUᵧ - Uᵧ_exact', 'RdYlBu')
        ]
        
        # Plot each equation's satisfaction
        for i, (key, title, cmap) in enumerate(equations):
            if i >= 7:  # Only plot first 7
                break
                
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            residual_data = reshaped_residuals[key]
            
            # Use log scale for better visualization of small residuals
            abs_residual = np.abs(residual_data)
            log_residual = np.log10(abs_residual + 1e-10)
            
            # Create contour plot
            levels = np.linspace(log_residual.min(), log_residual.max(), 20)
            contour = ax.contourf(X_mesh, Y_mesh, log_residual, levels=levels, cmap=cmap, alpha=0.8)
            
            # Add contour lines
            ax.contour(X_mesh, Y_mesh, log_residual, levels=levels[::4], colors='black', alpha=0.3, linewidths=0.5)
            
            # Styling
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
            cbar.set_label('log₁₀|residual|', rotation=270, labelpad=15, fontsize=9)
            
            # Add domain boundary
            boundary = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                               fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(boundary)
        
        # Overall physics satisfaction metric
        overall_ax = axes[2, 1]
        
        # Compute overall satisfaction score
        total_residual = np.zeros_like(reshaped_residuals['pde_x'])
        for key in ['pde_x', 'pde_y', 'constitutive_xx', 'constitutive_yy', 'constitutive_xy']:
            total_residual += np.abs(reshaped_residuals[key])
        
        log_total = np.log10(total_residual + 1e-10)
        
        contour = overall_ax.contourf(X_mesh, Y_mesh, log_total, levels=20, cmap='RdYlGn_r', alpha=0.8)
        overall_ax.set_title('Overall Physics\nSatisfaction', fontsize=11, fontweight='bold')
        overall_ax.set_xlabel('X')
        overall_ax.set_ylabel('Y')
        overall_ax.set_aspect('equal')
        
        cbar = plt.colorbar(contour, ax=overall_ax, shrink=0.8)
        cbar.set_label('log₁₀|total residual|', rotation=270, labelpad=15, fontsize=9)
        
        # Statistics panel
        stats_ax = axes[2, 2]
        stats_ax.axis('off')
        
        # Compute statistics
        stats_text = "Physics Equation Statistics\n\n"
        for key, title, _ in equations:
            if key in reshaped_residuals:
                res = reshaped_residuals[key]
                mean_abs = np.mean(np.abs(res))
                max_abs = np.max(np.abs(res))
                stats_text += f"{title.split()[0]}: {mean_abs:.2e} (max: {max_abs:.2e})\n"
        
        # Learning phase detection
        if epoch < 100:
            phase = "🌱 Discovery Phase"
            phase_desc = "Learning basic physics"
        elif epoch < 500:
            phase = "🔥 Rapid Learning"
            phase_desc = "Balancing equations"
        elif epoch < 1000:
            phase = "⚖️ Fine-tuning"
            phase_desc = "Optimizing satisfaction"
        else:
            phase = "🎯 Converged"
            phase_desc = "Physics mastered"
        
        stats_text += f"\n{phase}\n{phase_desc}"
        
        stats_ax.text(0.1, 0.9, stats_text, transform=stats_ax.transAxes, fontsize=10,
                     verticalalignment='top', 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_physics_animation(self):
        """Create animation of physics equation discovery"""
        print("⚖️ Creating physics equation discovery animation...")
        
        # Find all model snapshots
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found")
            return
        
        os.makedirs('physics_frames', exist_ok=True)
        
        # Create frames
        for i, model_file in enumerate(model_files):
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            try:
                fig = self.create_physics_frame(model_file, grid_resolution=40)
                plt.savefig(f'physics_frames/physics_frame_{i:04d}.png', 
                           dpi=100, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
            except Exception as e:
                print(f"⚠️ Error processing epoch {epoch}: {e}")
                continue
        
        # Create GIF
        print("🎬 Creating physics discovery GIF...")
        frame_files = sorted(glob.glob('physics_frames/physics_frame_*.png'))
        
        if frame_files:
            images = [Image.open(f) for f in frame_files]
            images[0].save('animations/physics_equation_discovery.gif', save_all=True,
                          append_images=images[1:], duration=300, loop=0)
            print("✅ Physics discovery GIF created!")
            
            # Fast version
            fast_images = images[::2]
            if len(fast_images) > 1:
                fast_images[0].save('animations/physics_equation_discovery_fast.gif', save_all=True,
                                  append_images=fast_images[1:], duration=200, loop=0)
                print("✅ Fast physics discovery GIF created!")

def main():
    """Main function"""
    print("⚖️ Physics Equation Discovery Visualizer")
    print("=" * 50)
    
    visualizer = PhysicsEquationVisualizer()
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    print("\n⚖️ Creating physics equation discovery visualization...")
    visualizer.create_physics_animation()
    
    print("\n🎉 Physics equation visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/physics_equation_discovery.gif - Full physics analysis")
    print("   📂 animations/physics_equation_discovery_fast.gif - Faster version")
    print("   📂 physics_frames/ - Individual frames")
    print("\n🔬 This shows how your PINN learns to satisfy each physics equation!")

if __name__ == "__main__":
    main()