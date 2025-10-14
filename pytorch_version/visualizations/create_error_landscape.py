#!/usr/bin/env python3
"""
Error Landscape Evolution Visualizer for PINN
Creates 3D surface plots showing how prediction error changes across the physical domain over training
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import os
import glob
from visualize_network_graph import NetworkVisualizer
from PIL import Image
import matplotlib.animation as animation

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

def stress_xx_exact(x, y):
    """Exact stress σxx computed from exact displacements"""
    dux_dx = -2*pi*torch.sin(2*pi*x)*torch.sin(pi*y)
    duy_dy = torch.sin(pi*x)*Q*torch.pow(y, 3)
    return (lmda + 2*mu)*dux_dx + lmda*duy_dy

def stress_yy_exact(x, y):
    """Exact stress σyy computed from exact displacements"""
    dux_dx = -2*pi*torch.sin(2*pi*x)*torch.sin(pi*y)
    duy_dy = torch.sin(pi*x)*Q*torch.pow(y, 3)
    return (lmda + 2*mu)*duy_dy + lmda*dux_dx

def stress_xy_exact(x, y):
    """Exact stress σxy computed from exact displacements"""
    dux_dy = pi*torch.cos(2*pi*x)*torch.cos(pi*y)
    duy_dx = pi*torch.cos(pi*x)*Q/4*torch.pow(y, 4)
    return mu*(dux_dy + duy_dx)

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

class ErrorLandscapeVisualizer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        self.epochs = []
        
    def load_model_at_epoch(self, epoch_file):
        """Load a specific model snapshot"""
        model = PINN().to(device)
        model.load_state_dict(torch.load(epoch_file, map_location=device))
        model.eval()
        return model
    
    def compute_error_landscape(self, model, grid_resolution=50):
        """Compute prediction errors across the domain"""
        
        # Create test grid
        x_test = np.linspace(xmin, xmax, grid_resolution)
        y_test = np.linspace(ymin, ymax, grid_resolution)
        X_mesh, Y_mesh = np.meshgrid(x_test, y_test)
        
        # Flatten for model input
        x_flat = X_mesh.flatten()
        y_flat = Y_mesh.flatten()
        grid_points = torch.tensor(np.column_stack([x_flat, y_flat]), dtype=DTYPE, device=device)
        
        # Model predictions
        with torch.no_grad():
            Ux_pred, Uy_pred, Sxx_pred, Syy_pred, Sxy_pred = model(grid_points)
        
        # Extract coordinates as tensors for exact solutions
        x_tensor = grid_points[:, 0:1]
        y_tensor = grid_points[:, 1:2]
        
        # Exact solutions
        Ux_exact = u_x_ext(x_tensor, y_tensor)
        Uy_exact = u_y_ext(x_tensor, y_tensor)
        Sxx_exact = stress_xx_exact(x_tensor, y_tensor)
        Syy_exact = stress_yy_exact(x_tensor, y_tensor)
        Sxy_exact = stress_xy_exact(x_tensor, y_tensor)
        
        # Compute errors
        error_ux = torch.abs(Ux_pred - Ux_exact).cpu().numpy().reshape(grid_resolution, grid_resolution)
        error_uy = torch.abs(Uy_pred - Uy_exact).cpu().numpy().reshape(grid_resolution, grid_resolution)
        error_sxx = torch.abs(Sxx_pred - Sxx_exact).cpu().numpy().reshape(grid_resolution, grid_resolution)
        error_syy = torch.abs(Syy_pred - Syy_exact).cpu().numpy().reshape(grid_resolution, grid_resolution)
        error_sxy = torch.abs(Sxy_pred - Sxy_exact).cpu().numpy().reshape(grid_resolution, grid_resolution)
        
        # Total error (L2 norm across all quantities)
        total_error = np.sqrt(error_ux**2 + error_uy**2 + error_sxx**2 + error_syy**2 + error_sxy**2)
        
        return {
            'X': X_mesh,
            'Y': Y_mesh,
            'total_error': total_error,
            'error_ux': error_ux,
            'error_uy': error_uy,
            'error_sxx': error_sxx,
            'error_syy': error_syy,
            'error_sxy': error_sxy,
            'max_error': total_error.max(),
            'mean_error': total_error.mean(),
            'predictions': {
                'Ux': Ux_pred.cpu().numpy().reshape(grid_resolution, grid_resolution),
                'Uy': Uy_pred.cpu().numpy().reshape(grid_resolution, grid_resolution),
                'Sxx': Sxx_pred.cpu().numpy().reshape(grid_resolution, grid_resolution),
                'Syy': Syy_pred.cpu().numpy().reshape(grid_resolution, grid_resolution),
                'Sxy': Sxy_pred.cpu().numpy().reshape(grid_resolution, grid_resolution)
            },
            'exact': {
                'Ux': Ux_exact.cpu().numpy().reshape(grid_resolution, grid_resolution),
                'Uy': Uy_exact.cpu().numpy().reshape(grid_resolution, grid_resolution),
                'Sxx': Sxx_exact.cpu().numpy().reshape(grid_resolution, grid_resolution),
                'Syy': Syy_exact.cpu().numpy().reshape(grid_resolution, grid_resolution),
                'Sxy': Sxy_exact.cpu().numpy().reshape(grid_resolution, grid_resolution)
            }
        }
    
    def create_3d_error_surface(self, epoch_file, error_type='total_error', save_individual=True):
        """Create 3D surface plot of error landscape"""
        
        epoch = int(epoch_file.split('_')[-1].split('.')[0])
        print(f"🏔️ Creating 3D error landscape for epoch {epoch}")
        
        # Load model and compute errors
        model = self.load_model_at_epoch(epoch_file)
        error_data = self.compute_error_landscape(model, grid_resolution=40)
        
        X = error_data['X']
        Y = error_data['Y']
        Z = error_data[error_type]
        
        # Create 3D surface plot
        fig = plt.figure(figsize=(16, 12))
        
        # Main 3D surface
        ax1 = fig.add_subplot(2, 2, (1, 2), projection='3d')
        
        # Use log scale for better visualization of error ranges
        Z_log = np.log10(Z + 1e-10)
        
        # Create surface with colormap
        surf = ax1.plot_surface(X, Y, Z_log, cmap='hot', alpha=0.9, 
                               linewidth=0, antialiased=True, rasterized=True)
        
        # Add contour lines on the bottom
        contours = ax1.contour(X, Y, Z_log, levels=10, colors='black', alpha=0.4, linewidths=0.5)
        
        # Styling
        ax1.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax1.set_zlabel('log₁₀(Error)', fontsize=12, fontweight='bold')
        
        # Phase detection for title
        if epoch < 100:
            phase = "🌋 Volcanic Errors"
            phase_desc = "Chaotic high errors"
        elif epoch < 500:
            phase = "🏔️ Mountain Formation"
            phase_desc = "Error peaks forming"
        elif epoch < 1000:
            phase = "⛰️ Valley Carving"
            phase_desc = "Errors localizing"
        else:
            phase = "🏞️ Smooth Plains"
            phase_desc = "Low uniform errors"
        
        title = f'3D Error Landscape Evolution\n{phase} - Epoch {epoch}\n{phase_desc}'
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set viewing angle for best perspective
        ax1.view_init(elev=25, azim=45)
        
        # Add colorbar
        cbar = plt.colorbar(surf, ax=ax1, shrink=0.6, aspect=30)
        cbar.set_label('log₁₀(Prediction Error)', rotation=270, labelpad=20)
        
        # Add 2D contour plot
        ax2 = fig.add_subplot(2, 2, 3)
        contour_2d = ax2.contourf(X, Y, Z_log, levels=20, cmap='hot')
        ax2.contour(X, Y, Z_log, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('2D Error Contours')
        ax2.set_aspect('equal')
        plt.colorbar(contour_2d, ax=ax2)
        
        # Add statistics panel
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.axis('off')
        
        # Compute error statistics
        stats_text = f"Error Landscape Statistics\nEpoch {epoch}\n\n"
        stats_text += f"Max Error: {error_data['max_error']:.2e}\n"
        stats_text += f"Mean Error: {error_data['mean_error']:.2e}\n"
        stats_text += f"Error Std: {Z.std():.2e}\n\n"
        
        # Error hotspots
        hotspot_threshold = np.percentile(Z, 95)
        hotspot_percent = np.sum(Z > hotspot_threshold) / Z.size * 100
        stats_text += f"Error Hotspots: {hotspot_percent:.1f}% of domain\n"
        stats_text += f"Hotspot Threshold: {hotspot_threshold:.2e}\n\n"
        
        # Component breakdown
        stats_text += "Component Errors:\n"
        for component in ['error_ux', 'error_uy', 'error_sxx', 'error_syy', 'error_sxy']:
            comp_error = error_data[component]
            stats_text += f"  {component[6:].upper()}: {comp_error.mean():.2e}\n"
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.9))
        
        plt.tight_layout()
        
        if save_individual:
            plt.savefig(f'landscape_frames/landscape_epoch_{epoch:04d}.png', 
                       dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        return fig, error_data
    
    def create_component_error_landscapes(self, epoch_file):
        """Create comprehensive error landscape showing all components"""
        
        epoch = int(epoch_file.split('_')[-1].split('.')[0])
        print(f"🌍 Creating comprehensive error landscapes for epoch {epoch}")
        
        # Load model and compute errors
        model = self.load_model_at_epoch(epoch_file)
        error_data = self.compute_error_landscape(model, grid_resolution=40)
        
        X = error_data['X']
        Y = error_data['Y']
        
        # Create figure with subplots for each component
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Complete Error Landscape Analysis - Epoch {epoch}', 
                    fontsize=18, fontweight='bold')
        
        components = [
            ('total_error', 'Total Combined Error', 'hot'),
            ('error_ux', 'Displacement Ux Error', 'Reds'),
            ('error_uy', 'Displacement Uy Error', 'Blues'),
            ('error_sxx', 'Stress σxx Error', 'Greens'),
            ('error_syy', 'Stress σyy Error', 'Purples'),
            ('error_sxy', 'Stress σxy Error', 'Oranges')
        ]
        
        for i, (component, title, cmap) in enumerate(components):
            # 3D surface plots
            ax_3d = fig.add_subplot(3, 4, 2*i + 1, projection='3d')
            
            Z = error_data[component]
            Z_log = np.log10(Z + 1e-12)
            
            surf = ax_3d.plot_surface(X, Y, Z_log, cmap=cmap, alpha=0.8, 
                                     linewidth=0, antialiased=True)
            
            ax_3d.set_title(f'{title}\n(3D Surface)', fontsize=10, fontweight='bold')
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('log₁₀(Error)')
            ax_3d.view_init(elev=30, azim=45)
            
            # 2D contour plots
            ax_2d = fig.add_subplot(3, 4, 2*i + 2)
            
            contour = ax_2d.contourf(X, Y, Z_log, levels=15, cmap=cmap)
            ax_2d.contour(X, Y, Z_log, levels=15, colors='black', alpha=0.3, linewidths=0.5)
            ax_2d.set_title(f'{title}\n(2D Contours)', fontsize=10, fontweight='bold')
            ax_2d.set_xlabel('X')
            ax_2d.set_ylabel('Y')
            ax_2d.set_aspect('equal')
            
            # Add small colorbar
            cbar = plt.colorbar(contour, ax=ax_2d, shrink=0.8)
            cbar.set_label('log₁₀(Error)', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def create_error_evolution_animation(self, grid_resolution=30):
        """Create animation showing error landscape evolution"""
        print("🎬 Creating error landscape evolution animation...")
        
        # Find all model snapshots
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found")
            return
        
        os.makedirs('landscape_frames', exist_ok=True)
        os.makedirs('component_frames', exist_ok=True)
        
        # Subsample for performance
        step = max(1, len(model_files) // 40)  # Max 40 frames
        selected_files = model_files[::step]
        
        print(f"📊 Processing {len(selected_files)} epochs for animation...")
        
        # Create individual landscape frames
        for i, model_file in enumerate(selected_files):
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            try:
                # Create 3D error surface
                self.create_3d_error_surface(model_file, error_type='total_error', save_individual=True)
                
                # Create comprehensive component analysis (every 5th frame)
                if i % 5 == 0:
                    fig = self.create_component_error_landscapes(model_file)
                    plt.savefig(f'component_frames/components_epoch_{epoch:04d}.png', 
                               dpi=80, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                
            except Exception as e:
                print(f"⚠️ Error processing epoch {epoch}: {e}")
                continue
        
        # Create GIFs
        self.create_landscape_gifs()
    
    def create_landscape_gifs(self):
        """Create GIFs from generated frames"""
        print("🎬 Creating error landscape GIFs...")
        
        # Main landscape animation
        landscape_files = sorted(glob.glob('landscape_frames/landscape_epoch_*.png'))
        if landscape_files:
            images = [Image.open(f) for f in landscape_files]
            images[0].save('animations/error_landscape_evolution.gif', save_all=True,
                          append_images=images[1:], duration=200, loop=0)
            print("✅ Error landscape evolution GIF created!")
            
            # Fast version
            fast_images = images[::2]
            if len(fast_images) > 1:
                fast_images[0].save('animations/error_landscape_fast.gif', save_all=True,
                                  append_images=fast_images[1:], duration=150, loop=0)
                print("✅ Fast error landscape GIF created!")
        
        # Component analysis animation
        component_files = sorted(glob.glob('component_frames/components_epoch_*.png'))
        if component_files:
            images = [Image.open(f) for f in component_files]
            images[0].save('animations/component_error_evolution.gif', save_all=True,
                          append_images=images[1:], duration=400, loop=0)
            print("✅ Component error evolution GIF created!")
    
    def create_error_summary_plots(self):
        """Create summary analysis of error evolution"""
        print("📊 Creating error evolution summary...")
        
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        if not model_files:
            return
        
        epochs = []
        max_errors = []
        mean_errors = []
        component_errors = {comp: [] for comp in ['error_ux', 'error_uy', 'error_sxx', 'error_syy', 'error_sxy']}
        
        # Sample every 10th model for efficiency
        for model_file in model_files[::10]:
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            try:
                model = self.load_model_at_epoch(model_file)
                error_data = self.compute_error_landscape(model, grid_resolution=30)
                
                epochs.append(epoch)
                max_errors.append(error_data['max_error'])
                mean_errors.append(error_data['mean_error'])
                
                for comp in component_errors.keys():
                    component_errors[comp].append(error_data[comp].mean())
                    
            except Exception as e:
                print(f"⚠️ Error processing epoch {epoch}: {e}")
                continue
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Error Landscape Evolution Summary', fontsize=16, fontweight='bold')
        
        # Total error evolution
        ax = axes[0, 0]
        ax.semilogy(epochs, max_errors, 'r-', linewidth=2, label='Max Error')
        ax.semilogy(epochs, mean_errors, 'b-', linewidth=2, label='Mean Error')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error (log scale)')
        ax.set_title('Total Error Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Component error evolution
        ax = axes[0, 1]
        colors = ['r', 'b', 'g', 'm', 'orange']
        for i, (comp, color) in enumerate(zip(component_errors.keys(), colors)):
            ax.semilogy(epochs, component_errors[comp], color=color, linewidth=2, 
                       label=comp[6:].upper())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Error (log scale)')
        ax.set_title('Component Error Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error reduction rate
        ax = axes[1, 0]
        if len(mean_errors) > 1:
            error_reduction = [(mean_errors[i-1] - mean_errors[i])/mean_errors[i-1] * 100 
                              for i in range(1, len(mean_errors))]
            ax.plot(epochs[1:], error_reduction, 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Error Reduction Rate (%)')
            ax.set_title('Learning Speed (Error Reduction)')
            ax.grid(True, alpha=0.3)
        
        # Final error distribution
        ax = axes[1, 1]
        if component_errors:
            final_errors = [component_errors[comp][-1] for comp in component_errors.keys()]
            components = [comp[6:].upper() for comp in component_errors.keys()]
            bars = ax.bar(components, final_errors, color=['r', 'b', 'g', 'm', 'orange'])
            ax.set_ylabel('Final Mean Error')
            ax.set_title('Final Error by Component')
            ax.set_yscale('log')
            
            # Add value labels on bars
            for bar, error in zip(bars, final_errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{error:.2e}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('error_evolution_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function"""
    print("🏔️ 3D Error Landscape Evolution Visualizer")
    print("=" * 60)
    
    visualizer = ErrorLandscapeVisualizer()
    
    # Create output directories
    os.makedirs('animations', exist_ok=True)
    
    print("\n🏔️ Creating 3D error landscape evolution...")
    visualizer.create_error_evolution_animation(grid_resolution=25)
    
    print("\n📊 Creating error evolution summary...")
    visualizer.create_error_summary_plots()
    
    print("\n🎉 Error landscape visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/error_landscape_evolution.gif - 3D surface evolution")
    print("   📂 animations/error_landscape_fast.gif - Faster version")
    print("   📂 animations/component_error_evolution.gif - All components")
    print("   📂 error_evolution_summary.png - Statistical summary")
    print("   📂 landscape_frames/ - Individual 3D surface frames")
    print("   📂 component_frames/ - Component analysis frames")
    print("\n🏔️ Watch your PINN's error landscape transform from volcanic chaos to smooth plains!")

if __name__ == "__main__":
    main()