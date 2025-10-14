#!/usr/bin/env python3
"""
Gradient Flow Rivers Visualizer for PINN
Shows gradients as flowing water, watch how "gradient rivers" change course during training
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.streamplot import streamplot
import os
import glob
from visualize_network_graph import NetworkVisualizer
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
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

class GradientRiverVisualizer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        
    def load_model_at_epoch(self, epoch_file):
        """Load a specific model snapshot"""
        model = PINN().to(device)
        model.load_state_dict(torch.load(epoch_file, map_location=device))
        model.eval()
        return model
    
    def compute_gradient_field(self, model, grid_resolution=30):
        """Compute gradient field across the domain for river visualization"""
        
        # Create test grid
        x_test = np.linspace(xmin, xmax, grid_resolution)
        y_test = np.linspace(ymin, ymax, grid_resolution)
        X_mesh, Y_mesh = np.meshgrid(x_test, y_test)
        
        # Flatten for model input
        x_flat = X_mesh.flatten()
        y_flat = Y_mesh.flatten()
        grid_points = torch.tensor(np.column_stack([x_flat, y_flat]), dtype=DTYPE, device=device, requires_grad=True)
        
        # Model predictions
        Ux, Uy, Sxx, Syy, Sxy = model(grid_points)
        
        # Compute gradients for each output
        gradients = {}
        
        outputs = {'Ux': Ux, 'Uy': Uy, 'Sxx': Sxx, 'Syy': Syy, 'Sxy': Sxy}
        
        for output_name, output_tensor in outputs.items():
            # Compute gradient of each output w.r.t. inputs
            grad_outputs = torch.autograd.grad(
                outputs=output_tensor.sum(),
                inputs=grid_points,
                create_graph=False,
                retain_graph=True
            )[0]
            
            # Extract x and y components
            grad_x = grad_outputs[:, 0].detach().cpu().numpy().reshape(grid_resolution, grid_resolution)
            grad_y = grad_outputs[:, 1].detach().cpu().numpy().reshape(grid_resolution, grid_resolution)
            
            # Gradient magnitude (river "speed")
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            gradients[output_name] = {
                'grad_x': grad_x,
                'grad_y': grad_y,
                'magnitude': grad_magnitude,
                'output_values': output_tensor.detach().cpu().numpy().reshape(grid_resolution, grid_resolution)
            }
        
        # Compute total gradient field (combined "watershed")
        total_grad_x = sum([gradients[name]['grad_x'] for name in gradients.keys()]) / len(gradients)
        total_grad_y = sum([gradients[name]['grad_y'] for name in gradients.keys()]) / len(gradients)
        total_magnitude = np.sqrt(total_grad_x**2 + total_grad_y**2)
        
        gradients['total'] = {
            'grad_x': total_grad_x,
            'grad_y': total_grad_y,
            'magnitude': total_magnitude,
            'output_values': total_magnitude
        }
        
        return X_mesh, Y_mesh, gradients
    
    def create_river_frame(self, epoch_file, output_type='total', save_individual=True):
        """Create a single gradient river frame"""
        
        epoch = int(epoch_file.split('_')[-1].split('.')[0])
        print(f"🌊 Creating gradient rivers for epoch {epoch} - {output_type}")
        
        # Load model
        model = self.load_model_at_epoch(epoch_file)
        
        # Compute gradient field
        X, Y, gradients = self.compute_gradient_field(model, grid_resolution=25)
        
        if output_type not in gradients:
            print(f"❌ Output type {output_type} not found")
            return None
        
        grad_data = gradients[output_type]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('navy')
        
        # === Main Streamplot (Top-left) ===
        ax = axes[0, 0]
        
        # Smooth the gradient field for better flow visualization
        smooth_grad_x = gaussian_filter(grad_data['grad_x'], sigma=0.8)
        smooth_grad_y = gaussian_filter(grad_data['grad_y'], sigma=0.8)
        smooth_magnitude = np.sqrt(smooth_grad_x**2 + smooth_grad_y**2)
        
        # Create streamplot (rivers)
        if smooth_magnitude.max() > 1e-10:
            # Normalize for better visualization
            norm_factor = smooth_magnitude.max()
            u_norm = smooth_grad_x / norm_factor
            v_norm = smooth_grad_y / norm_factor
            
            # Create streamlines
            streams = ax.streamplot(X, Y, u_norm, v_norm, 
                                  color=smooth_magnitude, 
                                  cmap='cool',
                                  density=2,
                                  linewidth=1 + 3*smooth_magnitude/smooth_magnitude.max(),
                                  arrowsize=1.5,
                                  arrowstyle='->')
            
            # Add background showing gradient magnitude as "water depth"
            contour = ax.contourf(X, Y, smooth_magnitude, levels=15, cmap='Blues', alpha=0.4)
            
        ax.set_facecolor('darkblue')
        ax.set_title(f'Gradient Rivers - {output_type.upper()}\nEpoch {epoch}', 
                    color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position', color='white')
        ax.set_ylabel('Y Position', color='white')
        ax.tick_params(colors='white')
        
        # === Gradient Magnitude (Top-right) ===
        ax = axes[0, 1]
        magnitude_plot = ax.imshow(grad_data['magnitude'], extent=[xmin, xmax, ymin, ymax], 
                                  cmap='hot', origin='lower', alpha=0.8)
        ax.contour(X, Y, grad_data['magnitude'], levels=10, colors='white', alpha=0.5, linewidths=0.8)
        
        ax.set_title(f'Gradient Magnitude\n"River Speed"', color='white', fontweight='bold')
        ax.set_xlabel('X Position', color='white')
        ax.set_ylabel('Y Position', color='white')
        ax.tick_params(colors='white')
        
        cbar = plt.colorbar(magnitude_plot, ax=ax)
        cbar.set_label('Gradient Magnitude', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # === Output Values (Bottom-left) ===
        ax = axes[1, 0]
        if output_type != 'total':
            output_plot = ax.imshow(grad_data['output_values'], extent=[xmin, xmax, ymin, ymax], 
                                   cmap='seismic', origin='lower', alpha=0.8)
            ax.set_title(f'Output Values - {output_type.upper()}\n"River Source"', 
                        color='white', fontweight='bold')
        else:
            output_plot = ax.imshow(grad_data['magnitude'], extent=[xmin, xmax, ymin, ymax], 
                                   cmap='plasma', origin='lower', alpha=0.8)
            ax.set_title(f'Combined Gradient Field\n"Total Watershed"', 
                        color='white', fontweight='bold')
        
        ax.set_xlabel('X Position', color='white')
        ax.set_ylabel('Y Position', color='white')
        ax.tick_params(colors='white')
        
        cbar = plt.colorbar(output_plot, ax=ax)
        cbar.set_label('Output Value' if output_type != 'total' else 'Field Strength', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # === Vector Field (Bottom-right) ===
        ax = axes[1, 1]
        
        # Subsample for quiver plot
        skip = 3
        X_sub = X[::skip, ::skip]
        Y_sub = Y[::skip, ::skip]
        U_sub = grad_data['grad_x'][::skip, ::skip]
        V_sub = grad_data['grad_y'][::skip, ::skip]
        
        # Color by magnitude
        magnitude_sub = grad_data['magnitude'][::skip, ::skip]
        
        # Create quiver plot
        quiver = ax.quiver(X_sub, Y_sub, U_sub, V_sub, magnitude_sub,
                          cmap='cool', alpha=0.8, scale_units='xy', scale=1, width=0.003)
        
        ax.set_facecolor('darkblue')
        ax.set_title(f'Gradient Vector Field\n"River Currents"', color='white', fontweight='bold')
        ax.set_xlabel('X Position', color='white')
        ax.set_ylabel('Y Position', color='white')
        ax.tick_params(colors='white')
        
        # === Overall Styling ===
        
        # Add epoch and phase information
        if epoch < 100:
            phase = "🌊 Torrential Flow"
            phase_desc = "Chaotic gradient currents"
        elif epoch < 500:
            phase = "🏞️ River Formation"
            phase_desc = "Channels beginning to form"
        elif epoch < 1000:
            phase = "🏔️ Watershed Carving"
            phase_desc = "Stable flow patterns"
        else:
            phase = "🌅 Serene Rivers"
            phase_desc = "Smooth gradient flow"
        
        fig.suptitle(f'Gradient Flow Rivers - {output_type.upper()}\n'
                    f'{phase} - Epoch {epoch}\n{phase_desc}', 
                    color='white', fontsize=16, fontweight='bold')
        
        # Style all axes
        for ax in axes.flatten():
            ax.set_facecolor('navy')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_individual:
            plt.savefig(f'river_frames/river_{output_type}_epoch_{epoch:04d}.png', 
                       dpi=100, bbox_inches='tight', facecolor='navy')
            plt.close(fig)
        
        return fig
    
    def create_combined_watershed(self, epoch_file):
        """Create a comprehensive watershed showing all outputs"""
        
        epoch = int(epoch_file.split('_')[-1].split('.')[0])
        print(f"🏞️ Creating complete watershed for epoch {epoch}")
        
        # Load model
        model = self.load_model_at_epoch(epoch_file)
        
        # Compute gradient field
        X, Y, gradients = self.compute_gradient_field(model, grid_resolution=30)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.patch.set_facecolor('navy')
        
        # Plot each output's gradient rivers
        outputs = ['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy', 'total']
        colors = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'cool']
        
        for i, (output_name, cmap) in enumerate(zip(outputs, colors)):
            if i >= 6:
                break
                
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            grad_data = gradients[output_name]
            
            # Smooth gradients
            smooth_grad_x = gaussian_filter(grad_data['grad_x'], sigma=0.5)
            smooth_grad_y = gaussian_filter(grad_data['grad_y'], sigma=0.5)
            smooth_magnitude = np.sqrt(smooth_grad_x**2 + smooth_grad_y**2)
            
            # Create streamplot
            if smooth_magnitude.max() > 1e-10:
                norm_factor = smooth_magnitude.max()
                u_norm = smooth_grad_x / (norm_factor + 1e-10)
                v_norm = smooth_grad_y / (norm_factor + 1e-10)
                
                # Background "water depth"
                ax.contourf(X, Y, smooth_magnitude, levels=12, cmap=cmap, alpha=0.6)
                
                # Streamlines as rivers
                streams = ax.streamplot(X, Y, u_norm, v_norm, 
                                      color='white', 
                                      density=1.5,
                                      linewidth=0.5 + 2*smooth_magnitude/smooth_magnitude.max(),
                                      arrowsize=1.2)
            
            ax.set_facecolor('darkblue')
            ax.set_title(f'{output_name} Gradient Rivers', color='white', fontweight='bold')
            ax.set_xlabel('X', color='white')
            ax.set_ylabel('Y', color='white')
            ax.tick_params(colors='white')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
        
        # Overall title
        if epoch < 100:
            water_phase = "🌊 Flash Flood"
        elif epoch < 500:
            water_phase = "🏞️ River Channels"
        elif epoch < 1000:
            water_phase = "🏔️ Mountain Streams"
        else:
            water_phase = "🌅 Peaceful Waters"
        
        fig.suptitle(f'Complete Gradient Watershed - Epoch {epoch}\n{water_phase}', 
                    color='white', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_3d_river_landscape(self, epoch_file, output_type='Ux'):
        """Create 3D landscape showing gradient rivers as actual terrain"""
        
        epoch = int(epoch_file.split('_')[-1].split('.')[0])
        print(f"🏔️ Creating 3D river landscape for epoch {epoch} - {output_type}")
        
        model = self.load_model_at_epoch(epoch_file)
        X, Y, gradients = self.compute_gradient_field(model, grid_resolution=25)
        
        if output_type not in gradients:
            return None
        
        grad_data = gradients[output_type]
        
        # Create 3D landscape
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use gradient magnitude as terrain height
        Z = grad_data['magnitude']
        Z_smooth = gaussian_filter(Z, sigma=1.0)
        
        # Create surface
        surf = ax.plot_surface(X, Y, Z_smooth, 
                              cmap='terrain', 
                              alpha=0.8, 
                              linewidth=0, 
                              antialiased=True,
                              rasterized=True)
        
        # Add contour lines as "rivers"
        contours = ax.contour(X, Y, Z_smooth, levels=8, colors='blue', alpha=0.7, linewidths=2)
        
        # Add some "water flow" arrows
        skip = 4
        X_sub = X[::skip, ::skip]
        Y_sub = Y[::skip, ::skip]
        Z_sub = Z_smooth[::skip, ::skip]
        U_sub = grad_data['grad_x'][::skip, ::skip]
        V_sub = grad_data['grad_y'][::skip, ::skip]
        
        # Normalize flow vectors
        flow_magnitude = np.sqrt(U_sub**2 + V_sub**2)
        U_norm = U_sub / (flow_magnitude + 1e-10) * 0.05
        V_norm = V_sub / (flow_magnitude + 1e-10) * 0.05
        
        # Add flow arrows on surface
        ax.quiver(X_sub, Y_sub, Z_sub + 0.01, U_norm, V_norm, np.zeros_like(U_norm),
                 length=0.05, alpha=0.6, color='cyan', arrow_length_ratio=0.3)
        
        # Styling
        ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax.set_zlabel('Gradient Magnitude', fontsize=12, fontweight='bold')
        
        # Phase-based title
        if epoch < 100:
            landscape_phase = "🌋 Volcanic Terrain"
        elif epoch < 500:
            landscape_phase = "🏔️ Mountain Building"
        elif epoch < 1000:
            landscape_phase = "🏞️ Valley Formation"
        else:
            landscape_phase = "🌅 Gentle Hills"
        
        ax.set_title(f'3D Gradient River Landscape - {output_type}\n'
                    f'{landscape_phase} - Epoch {epoch}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        # Add colorbar
        cbar = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=30)
        cbar.set_label('Gradient "Elevation"', rotation=270, labelpad=20)
        
        if save_individual:
            plt.savefig(f'river_3d_frames/river_3d_{output_type}_epoch_{epoch:04d}.png', 
                       dpi=100, bbox_inches='tight', facecolor='navy')
            plt.close(fig)
        
        return fig
    
    def create_river_animations(self):
        """Create animated gradient rivers"""
        print("🌊 Creating gradient river animations...")
        
        # Find all model snapshots
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found")
            return
        
        # Create output directories
        os.makedirs('river_frames', exist_ok=True)
        os.makedirs('river_3d_frames', exist_ok=True)
        os.makedirs('watershed_frames', exist_ok=True)
        
        # Subsample for performance
        step = max(1, len(model_files) // 25)  # Max 25 frames
        selected_files = model_files[::step]
        
        print(f"🎬 Processing {len(selected_files)} epochs...")
        
        # Create frames for main outputs
        outputs_to_animate = ['Ux', 'Uy', 'total']
        
        for output_type in outputs_to_animate:
            print(f"\n🌊 Creating {output_type} river animation...")
            
            for frame_idx, model_file in enumerate(selected_files):
                epoch = int(model_file.split('_')[-1].split('.')[0])
                
                try:
                    # Create 2D river frame
                    self.create_river_frame(model_file, output_type, save_individual=True)
                    
                    # Create 3D landscape (every 3rd frame for performance)
                    if frame_idx % 3 == 0:
                        self.create_3d_river_landscape(model_file, output_type)
                        
                except Exception as e:
                    print(f"⚠️ Error processing epoch {epoch}: {e}")
                    continue
        
        # Create watershed frames
        print("\n🏞️ Creating complete watershed frames...")
        for frame_idx, model_file in enumerate(selected_files[::2]):  # Every other frame
            try:
                fig = self.create_combined_watershed(model_file)
                epoch = int(model_file.split('_')[-1].split('.')[0])
                plt.savefig(f'watershed_frames/watershed_epoch_{epoch:04d}.png', 
                           dpi=80, bbox_inches='tight', facecolor='navy')
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ Error creating watershed: {e}")
                continue
        
        # Create GIFs
        self.create_river_gifs(outputs_to_animate)
    
    def create_river_gifs(self, outputs):
        """Create GIFs from river frames"""
        print("🎬 Creating gradient river GIFs...")
        
        os.makedirs('animations', exist_ok=True)
        
        # Individual output rivers
        for output_type in outputs:
            frame_files = sorted(glob.glob(f'river_frames/river_{output_type}_epoch_*.png'))
            
            if frame_files:
                images = [Image.open(f) for f in frame_files]
                images[0].save(f'animations/gradient_rivers_{output_type}.gif', 
                              save_all=True, append_images=images[1:], 
                              duration=200, loop=0)
                print(f"✅ Gradient rivers GIF created for {output_type}")
        
        # Combined watershed
        watershed_files = sorted(glob.glob('watershed_frames/watershed_epoch_*.png'))
        if watershed_files:
            images = [Image.open(f) for f in watershed_files]
            images[0].save('animations/gradient_watershed_complete.gif', 
                          save_all=True, append_images=images[1:], 
                          duration=300, loop=0)
            print("✅ Complete watershed GIF created!")
        
        # 3D landscapes
        for output_type in outputs:
            landscape_files = sorted(glob.glob(f'river_3d_frames/river_3d_{output_type}_epoch_*.png'))
            if landscape_files:
                images = [Image.open(f) for f in landscape_files]
                images[0].save(f'animations/gradient_landscape_3d_{output_type}.gif', 
                              save_all=True, append_images=images[1:], 
                              duration=250, loop=0)
                print(f"✅ 3D gradient landscape GIF created for {output_type}")

def main():
    """Main function"""
    print("🌊 Gradient Flow Rivers Visualizer")
    print("=" * 50)
    
    visualizer = GradientRiverVisualizer()
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    print("\n🌊 Creating gradient river animations...")
    visualizer.create_river_animations()
    
    print("\n🎉 Gradient river visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/gradient_rivers_*.gif - Individual output rivers")
    print("   📂 animations/gradient_watershed_complete.gif - Complete watershed")
    print("   📂 animations/gradient_landscape_3d_*.gif - 3D river landscapes")
    print("   📂 river_frames/ - 2D river frames")
    print("   📂 river_3d_frames/ - 3D landscape frames")
    print("   📂 watershed_frames/ - Complete watershed frames")
    print("\n🌊 Watch how your PINN's gradient rivers carve their path to physics mastery!")

if __name__ == "__main__":
    main()