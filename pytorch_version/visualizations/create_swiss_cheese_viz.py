#!/usr/bin/env python3
"""
Swiss Cheese Volumetric Weight Visualizer
Creates continuous 3D volume with smooth color fields showing weight evolution
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import os
import glob
from visualize_network_graph import NetworkVisualizer
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

class SwissCheeseVisualizer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        self.weight_data = []
        self.epochs = []
        
    def load_weight_snapshots(self):
        """Load all weight snapshots"""
        print("🧀 Loading weight snapshots for Swiss cheese visualization...")
        
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found")
            return False
            
        for model_file in model_files:
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            visualizer = NetworkVisualizer(model_file)
            if visualizer.load_model():
                visualizer.analyze_architecture()
                
                # Extract all weights with 3D positions
                all_weights = []
                positions = []
                layer_z = 0
                
                for layer_name, info in visualizer.layer_info.items():
                    if 'hidden_layers' in layer_name or layer_name.startswith('output'):
                        weights = info['weight_tensor'].detach().numpy()
                        height, width = weights.shape
                        
                        for i in range(height):
                            for j in range(width):
                                positions.append([j, i, layer_z])
                                all_weights.append(weights[i, j])
                        
                        layer_z += 1
                
                self.weight_data.append({
                    'weights': np.array(all_weights),
                    'positions': np.array(positions)
                })
                self.epochs.append(epoch)
                
        print(f"✅ Loaded {len(self.weight_data)} snapshots for Swiss cheese rendering")
        return len(self.weight_data) > 0
    
    def create_volumetric_grid(self, weights, positions, grid_resolution=40):
        """Create a continuous 3D volume from discrete weight points"""
        
        # Define grid bounds
        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)
        
        # Create 3D grid
        x = np.linspace(min_coords[0], max_coords[0], grid_resolution)
        y = np.linspace(min_coords[1], max_coords[1], grid_resolution)
        z = np.linspace(min_coords[2], max_coords[2], grid_resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Interpolate weights to grid points
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Use linear interpolation to create smooth field
        interpolated_weights = griddata(
            positions, weights, grid_points, 
            method='linear', fill_value=0.0
        )
        
        # Reshape back to 3D grid
        weight_volume = interpolated_weights.reshape(X.shape)
        
        # Apply Gaussian smoothing for ultra-smooth Swiss cheese effect
        weight_volume = gaussian_filter(weight_volume, sigma=1.5)
        
        return X, Y, Z, weight_volume
    
    def create_swiss_cheese_frame(self, frame_idx, grid_resolution=35):
        """Create a single Swiss cheese frame"""
        
        epoch = self.epochs[frame_idx]
        weights = self.weight_data[frame_idx]['weights']
        positions = self.weight_data[frame_idx]['positions']
        
        print(f"🧀 Creating Swiss cheese frame {frame_idx+1}/{len(self.epochs)} (epoch {epoch})")
        
        # Create volumetric grid
        X, Y, Z, weight_volume = self.create_volumetric_grid(weights, positions, grid_resolution)
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create multiple isosurfaces for Swiss cheese effect
        weight_levels = np.linspace(-0.8, 0.8, 8)  # Different "cheese density" levels
        
        # Color map for weight values
        cmap = plt.cm.RdBu_r
        norm = Normalize(vmin=-1.0, vmax=1.0)
        
        # Create multiple transparent isosurfaces
        for i, level in enumerate(weight_levels):
            if level == 0:
                continue  # Skip zero level
                
            # Create isosurface using contour3D
            alpha_val = 0.3 if abs(level) < 0.4 else 0.6  # More opaque for stronger weights
            color = cmap(norm(level))
            
            try:
                # Create isosurface
                ax.contour3D(X, Y, Z, weight_volume, levels=[level], 
                           colors=[color], alpha=alpha_val, linewidths=0.5)
            except:
                pass  # Skip if contour can't be generated
        
        # Add some scattered points for reference
        significant_mask = np.abs(weights) > 0.3  # Only show strong weights as points
        if np.any(significant_mask):
            strong_weights = weights[significant_mask]
            strong_positions = positions[significant_mask]
            
            point_colors = cmap(norm(strong_weights))
            ax.scatter(strong_positions[:, 0], strong_positions[:, 1], strong_positions[:, 2],
                      c=point_colors, s=20, alpha=0.8, edgecolors='black', linewidth=0.3)
        
        # Styling
        ax.set_xlabel('Input Neurons →', fontsize=12, fontweight='bold')
        ax.set_ylabel('Output Neurons →', fontsize=12, fontweight='bold')
        ax.set_zlabel('Network Depth →', fontsize=12, fontweight='bold')
        
        # Phase detection for title
        if epoch < 100:
            phase = "🌱 Formation"
        elif epoch < 500:
            phase = "🔥 Melting"
        elif epoch < 1000:
            phase = "⚡ Shaping"
        else:
            phase = "🎯 Aging"
        
        ax.set_title(f'Swiss Cheese Weight Volume\n{phase} - Epoch {epoch:.1f}\n'
                    f'Continuous Weight Field Visualization', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set consistent viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Set limits
        ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
        ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())
        ax.set_zlim(positions[:, 2].min(), positions[:, 2].max())
        
        # Add weight statistics
        stats_text = f"Volume Range: [{weight_volume.min():.3f}, {weight_volume.max():.3f}]\n"
        stats_text += f"Strong Weights: {np.sum(significant_mask)}/{len(weights)}\n"
        stats_text += f"Volume Std: {weight_volume.std():.3f}"
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
        
        return fig, ax
    
    def create_slice_visualization(self, frame_idx, grid_resolution=50):
        """Create Swiss cheese with cross-sectional slices"""
        
        epoch = self.epochs[frame_idx]
        weights = self.weight_data[frame_idx]['weights']
        positions = self.weight_data[frame_idx]['positions']
        
        print(f"🥪 Creating slice visualization for epoch {epoch}")
        
        # Create volumetric grid
        X, Y, Z, weight_volume = self.create_volumetric_grid(weights, positions, grid_resolution)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 3D view
        ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')
        
        # Create multiple transparent slices
        mid_x = X.shape[0] // 2
        mid_y = X.shape[1] // 2
        mid_z = X.shape[2] // 2
        
        # X-slice (YZ plane)
        ax1.contourf(Y[mid_x, :, :], Z[mid_x, :, :], weight_volume[mid_x, :, :], 
                    levels=20, cmap='RdBu_r', alpha=0.8, vmin=-1, vmax=1)
        
        # Y-slice (XZ plane) 
        ax1.contourf(X[:, mid_y, :], weight_volume[:, mid_y, :], Z[:, mid_y, :], 
                    levels=20, cmap='RdBu_r', alpha=0.8, vmin=-1, vmax=1)
        
        # Z-slice (XY plane)
        ax1.contourf(X[:, :, mid_z], Y[:, :, mid_z], weight_volume[:, :, mid_z], 
                    levels=20, cmap='RdBu_r', alpha=0.8, vmin=-1, vmax=1, zdir='z', offset=mid_z)
        
        ax1.set_title(f'3D Swiss Cheese Slices - Epoch {epoch:.1f}', fontweight='bold')
        ax1.view_init(elev=25, azim=45)
        
        # 2D slice views
        slices_data = [
            (weight_volume[mid_x, :, :], 'YZ Cross-section', Y[mid_x, :, :], Z[mid_x, :, :]),
            (weight_volume[:, mid_y, :], 'XZ Cross-section', X[:, mid_y, :], Z[:, mid_y, :]),
            (weight_volume[:, :, mid_z], 'XY Cross-section', X[:, :, mid_z], Y[:, :, mid_z])
        ]
        
        for i, (slice_data, title, x_coords, y_coords) in enumerate(slices_data):
            ax = fig.add_subplot(2, 3, i + 2)
            im = ax.contourf(x_coords, y_coords, slice_data, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(title, fontweight='bold')
            ax.set_aspect('equal')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=fig.get_axes(), shrink=0.8, aspect=30)
        cbar.set_label('Weight Value', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig, ax1
    
    def create_animation(self, style='cheese', grid_resolution=35):
        """Create Swiss cheese animation"""
        print(f"🎬 Creating {style} animation...")
        
        os.makedirs('cheese_frames', exist_ok=True)
        
        # Create interpolated frames for smooth animation
        interpolated_data = []
        interpolated_epochs = []
        
        for i in range(len(self.weight_data) - 1):
            interpolated_data.append(self.weight_data[i])
            interpolated_epochs.append(self.epochs[i])
            
            # Add 2 interpolated frames
            for j in range(1, 3):
                alpha = j / 3
                weights_start = self.weight_data[i]['weights']
                weights_end = self.weight_data[i + 1]['weights']
                epoch_start = self.epochs[i]
                epoch_end = self.epochs[i + 1]
                
                interpolated_weights = (1 - alpha) * weights_start + alpha * weights_end
                interpolated_epoch = (1 - alpha) * epoch_start + alpha * epoch_end
                
                interpolated_data.append({
                    'weights': interpolated_weights,
                    'positions': self.weight_data[i]['positions']
                })
                interpolated_epochs.append(interpolated_epoch)
        
        # Add final frame
        interpolated_data.append(self.weight_data[-1])
        interpolated_epochs.append(self.epochs[-1])
        
        # Temporarily replace data with interpolated
        original_data = self.weight_data
        original_epochs = self.epochs
        self.weight_data = interpolated_data
        self.epochs = interpolated_epochs
        
        # Create frames
        for i in range(len(self.epochs)):
            if style == 'cheese':
                fig, ax = self.create_swiss_cheese_frame(i, grid_resolution)
            else:
                fig, ax = self.create_slice_visualization(i, grid_resolution)
            
            plt.savefig(f'cheese_frames/cheese_frame_{i:04d}.png', 
                       dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # Restore original data
        self.weight_data = original_data
        self.epochs = original_epochs
        
        # Create GIF
        print("🎬 Creating Swiss cheese GIF...")
        frame_files = sorted(glob.glob('cheese_frames/cheese_frame_*.png'))
        
        if frame_files:
            images = [Image.open(f) for f in frame_files]
            images[0].save('animations/swiss_cheese_weights.gif', save_all=True,
                          append_images=images[1:], duration=150, loop=0)
            print("✅ Swiss cheese GIF created!")
            
            # Fast version
            fast_images = images[::2]
            if len(fast_images) > 1:
                fast_images[0].save('animations/swiss_cheese_weights_fast.gif', save_all=True,
                                  append_images=fast_images[1:], duration=120, loop=0)
                print("✅ Fast Swiss cheese GIF created!")

def main():
    """Main function"""
    print("🧀 Swiss Cheese Weight Visualizer")
    print("=" * 50)
    
    visualizer = SwissCheeseVisualizer()
    
    if not visualizer.load_weight_snapshots():
        return
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    print("\n🧀 Creating Swiss cheese visualization...")
    visualizer.create_animation(style='cheese', grid_resolution=30)
    
    print("\n🎉 Swiss cheese visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/swiss_cheese_weights.gif - Volumetric Swiss cheese")
    print("   📂 animations/swiss_cheese_weights_fast.gif - Faster version")
    print("   📂 cheese_frames/ - Individual frames")

if __name__ == "__main__":
    main()