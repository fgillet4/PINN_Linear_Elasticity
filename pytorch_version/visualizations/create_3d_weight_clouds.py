#!/usr/bin/env python3
"""
3D Volumetric Weight Cloud Visualizer for PINN
Creates stunning 3D point clouds of weights with smooth interpolation and cinematic camera movement
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import os
import glob
from visualize_network_graph import NetworkVisualizer
from PIL import Image
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

class Weight3DCloudVisualizer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        self.weight_data = []
        self.epochs = []
        
    def load_weight_snapshots(self):
        """Load all weight snapshots from training"""
        print("📂 Loading weight snapshots for 3D cloud visualization...")
        
        # Find all model snapshot files
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found. Run train_with_weight_animation.py first!")
            return False
            
        for model_file in model_files:
            # Extract epoch number
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            # Load model
            visualizer = NetworkVisualizer(model_file)
            if visualizer.load_model():
                visualizer.analyze_architecture()
                
                # Extract all weight matrices and flatten into 3D point cloud
                all_weights = []
                positions = []
                layer_ids = []
                
                layer_z = 0
                for layer_name, info in visualizer.layer_info.items():
                    if 'hidden_layers' in layer_name or layer_name.startswith('output'):
                        weights = info['weight_tensor'].detach().numpy()
                        height, width = weights.shape
                        
                        # Create 3D coordinates for each weight
                        for i in range(height):
                            for j in range(width):
                                # Position: (input_neuron, output_neuron, layer_depth)
                                positions.append([j, i, layer_z])
                                all_weights.append(weights[i, j])
                                layer_ids.append(layer_z)
                        
                        layer_z += 1
                
                self.weight_data.append({
                    'weights': np.array(all_weights),
                    'positions': np.array(positions),
                    'layer_ids': np.array(layer_ids)
                })
                self.epochs.append(epoch)
                
        print(f"✅ Loaded {len(self.weight_data)} weight snapshots with 3D coordinates")
        print(f"📊 Each snapshot contains {len(self.weight_data[0]['weights'])} weight points")
        return len(self.weight_data) > 0
    
    def interpolate_weights(self, num_interpolated_frames=5):
        """Create smooth interpolation between weight snapshots"""
        print(f"🔄 Creating smooth interpolation with {num_interpolated_frames}x frames...")
        
        interpolated_data = []
        interpolated_epochs = []
        
        for i in range(len(self.weight_data) - 1):
            # Add original frame
            interpolated_data.append(self.weight_data[i])
            interpolated_epochs.append(self.epochs[i])
            
            # Create interpolated frames
            weights_start = self.weight_data[i]['weights']
            weights_end = self.weight_data[i + 1]['weights']
            epoch_start = self.epochs[i]
            epoch_end = self.epochs[i + 1]
            
            for j in range(1, num_interpolated_frames + 1):
                alpha = j / (num_interpolated_frames + 1)
                
                # Linear interpolation of weights
                interpolated_weights = (1 - alpha) * weights_start + alpha * weights_end
                interpolated_epoch = (1 - alpha) * epoch_start + alpha * epoch_end
                
                interpolated_data.append({
                    'weights': interpolated_weights,
                    'positions': self.weight_data[i]['positions'],  # Positions stay the same
                    'layer_ids': self.weight_data[i]['layer_ids']
                })
                interpolated_epochs.append(interpolated_epoch)
        
        # Add final frame
        interpolated_data.append(self.weight_data[-1])
        interpolated_epochs.append(self.epochs[-1])
        
        # Replace original data with interpolated
        self.weight_data = interpolated_data
        self.epochs = interpolated_epochs
        
        print(f"✅ Created {len(self.weight_data)} total frames with smooth interpolation")
    
    def create_3d_weight_cloud(self, save_frames=True, point_size=20, alpha=0.7):
        """Create 3D point cloud visualization of weights"""
        print("☁️ Creating 3D weight cloud visualization...")
        
        if not self.weight_data:
            print("❌ No weight data loaded")
            return
            
        # Create output directory
        os.makedirs('cloud_frames', exist_ok=True)
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get data dimensions
        positions = self.weight_data[0]['positions']
        max_x = positions[:, 0].max()
        max_y = positions[:, 1].max()
        max_z = positions[:, 2].max()
        
        print(f"📊 3D Cloud dimensions: {max_x+1} × {max_y+1} × {max_z+1}")
        
        # Color normalization
        all_weights = np.concatenate([data['weights'] for data in self.weight_data])
        vmin, vmax = np.percentile(all_weights, [5, 95])  # Use percentiles to avoid outliers
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        def animate_frame(frame_idx):
            ax.clear()
            
            epoch = self.epochs[frame_idx]
            weights = self.weight_data[frame_idx]['weights']
            positions = self.weight_data[frame_idx]['positions']
            layer_ids = self.weight_data[frame_idx]['layer_ids']
            
            # Filter significant weights for better visualization
            significant_mask = np.abs(weights) > 0.05
            sig_weights = weights[significant_mask]
            sig_positions = positions[significant_mask]
            sig_layers = layer_ids[significant_mask]
            
            # Create color map
            colors = plt.cm.RdBu_r(norm(sig_weights))
            
            # Create 3D scatter plot
            scatter = ax.scatter(sig_positions[:, 0], sig_positions[:, 1], sig_positions[:, 2],
                               c=colors, s=point_size, alpha=alpha, 
                               edgecolors='black', linewidth=0.1)
            
            # Set labels and title
            ax.set_xlabel('Input Neurons →', fontsize=12, fontweight='bold')
            ax.set_ylabel('Output Neurons →', fontsize=12, fontweight='bold') 
            ax.set_zlabel('Network Depth →', fontsize=12, fontweight='bold')
            
            # Dynamic title based on epoch
            if epoch < 100:
                phase = "🌱 Initialization"
            elif epoch < 500:
                phase = "🔥 Rapid Learning"
            elif epoch < 1000:
                phase = "⚡ Optimization"
            else:
                phase = "🎯 Fine-tuning"
                
            ax.set_title(f'3D PINN Weight Cloud Evolution\n{phase} - Epoch {epoch:.1f}\n'
                        f'Points: {len(sig_weights)} | Weight Range: [{vmin:.2f}, {vmax:.2f}]', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Set consistent axis limits
            ax.set_xlim(0, max_x)
            ax.set_ylim(0, max_y)
            ax.set_zlim(0, max_z)
            
            # Cinematic camera movement
            elevation = 20 + 10 * np.sin(frame_idx * 0.1)  # Slow elevation change
            azimuth = frame_idx * 2  # Slow rotation
            ax.view_init(elev=elevation, azim=azimuth)
            
            # Add layer labels
            for layer_z in range(int(max_z) + 1):
                if layer_z < 8:  # Hidden layers
                    label = f"H{layer_z+1}"
                else:  # Output layers
                    output_names = ['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy']
                    idx = layer_z - 8
                    label = output_names[idx] if idx < len(output_names) else f"O{idx+1}"
                
                ax.text(max_x + 1, max_y/2, layer_z, label, fontsize=8, fontweight='bold')
            
            # Add weight statistics
            stats_text = f"Active Weights: {len(sig_weights)}/{len(weights)}\n"
            stats_text += f"Mean: {sig_weights.mean():.3f}\n"
            stats_text += f"Std: {sig_weights.std():.3f}"
            
            ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            
            # Save frame
            if save_frames:
                plt.savefig(f'cloud_frames/cloud_frame_{frame_idx:04d}.png', 
                           dpi=100, bbox_inches='tight', facecolor='black')
            
            return [scatter]
        
        # Create animation
        print(f"🎬 Creating 3D cloud animation with {len(self.epochs)} frames...")
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(self.epochs),
                                     interval=100, blit=False, repeat=True)
        
        return anim
    
    def create_volumetric_representation(self, voxel_resolution=50):
        """Create true volumetric representation using voxel grids"""
        print(f"🧊 Creating volumetric voxel representation ({voxel_resolution}³)...")
        
        if not self.weight_data:
            return
            
        os.makedirs('voxel_frames', exist_ok=True)
        
        # Get dimensions
        positions = self.weight_data[0]['positions']
        max_x, max_y, max_z = positions.max(axis=0)
        
        # Create frames
        for frame_idx, epoch in enumerate(self.epochs):
            print(f"🎮 Creating voxel frame {frame_idx+1}/{len(self.epochs)}")
            
            weights = self.weight_data[frame_idx]['weights']
            positions = self.weight_data[frame_idx]['positions']
            
            # Create 3D voxel grid
            voxel_grid = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution))
            
            # Map weights to voxel coordinates
            scale_x = (voxel_resolution - 1) / max_x if max_x > 0 else 1
            scale_y = (voxel_resolution - 1) / max_y if max_y > 0 else 1
            scale_z = (voxel_resolution - 1) / max_z if max_z > 0 else 1
            
            for i, (pos, weight) in enumerate(zip(positions, weights)):
                vx = int(pos[0] * scale_x)
                vy = int(pos[1] * scale_y)
                vz = int(pos[2] * scale_z)
                
                if 0 <= vx < voxel_resolution and 0 <= vy < voxel_resolution and 0 <= vz < voxel_resolution:
                    voxel_grid[vx, vy, vz] = weight
            
            # Create 3D visualization of voxels
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Only show significant voxels
            threshold = np.abs(voxel_grid).max() * 0.1
            significant_mask = np.abs(voxel_grid) > threshold
            
            # Get coordinates of significant voxels
            vx, vy, vz = np.where(significant_mask)
            colors = plt.cm.RdBu_r((voxel_grid[significant_mask] + 1) / 2)  # Normalize to [0,1]
            
            # Create 3D scatter with larger points for voxel effect
            ax.scatter(vx, vy, vz, c=colors, s=100, alpha=0.6, marker='s')
            
            ax.set_title(f'3D Voxel Weight Volume - Epoch {epoch:.1f}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Dimension')
            ax.set_ylabel('Y Dimension')
            ax.set_zlabel('Z Dimension')
            
            # Cinematic rotation
            ax.view_init(elev=20, azim=frame_idx * 3)
            
            plt.savefig(f'voxel_frames/voxel_frame_{frame_idx:04d}.png', 
                       dpi=100, bbox_inches='tight', facecolor='black')
            plt.close(fig)
    
    def create_cloud_gif(self):
        """Create GIF from cloud frames"""
        print("🎬 Creating 3D cloud GIF...")
        
        frame_files = sorted(glob.glob('cloud_frames/cloud_frame_*.png'))
        
        if not frame_files:
            print("❌ No cloud frames found")
            return
            
        # Create GIF
        images = [Image.open(f) for f in frame_files]
        if images:
            images[0].save('animations/weight_3d_cloud.gif', save_all=True,
                          append_images=images[1:], duration=100, loop=0)
            print("✅ 3D weight cloud GIF saved as 'weight_3d_cloud.gif'")
            
            # Create fast version
            fast_images = images[::3]
            if len(fast_images) > 1:
                fast_images[0].save('animations/weight_3d_cloud_fast.gif', save_all=True,
                                  append_images=fast_images[1:], duration=80, loop=0)
                print("✅ Fast 3D cloud GIF saved as 'weight_3d_cloud_fast.gif'")

def main():
    """Main function to create 3D cloud visualizations"""
    print("☁️ 3D Weight Cloud Visualizer")
    print("=" * 50)
    
    # Create visualizer
    visualizer = Weight3DCloudVisualizer()
    
    # Load weight snapshots
    if not visualizer.load_weight_snapshots():
        return
    
    # Create smooth interpolation
    visualizer.interpolate_weights(num_interpolated_frames=3)
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    print("\n☁️ Creating 3D point cloud visualization...")
    anim = visualizer.create_3d_weight_cloud(save_frames=True, point_size=30, alpha=0.8)
    
    print("\n🧊 Creating volumetric voxel representation...")
    visualizer.create_volumetric_representation(voxel_resolution=30)
    
    print("\n🎬 Creating final GIFs...")
    visualizer.create_cloud_gif()
    
    # Create voxel GIF
    voxel_files = sorted(glob.glob('voxel_frames/voxel_frame_*.png'))
    if voxel_files:
        images = [Image.open(f) for f in voxel_files]
        images[0].save('animations/weight_3d_voxels.gif', save_all=True,
                      append_images=images[1:], duration=120, loop=0)
        print("✅ 3D voxel GIF saved as 'weight_3d_voxels.gif'")
    
    print("\n🎉 3D Cloud visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/weight_3d_cloud.gif - 3D point cloud with rotation")
    print("   📂 animations/weight_3d_cloud_fast.gif - Faster version")
    print("   📂 animations/weight_3d_voxels.gif - Volumetric voxel representation")
    print("   📂 cloud_frames/ - Individual cloud frames")
    print("   📂 voxel_frames/ - Individual voxel frames")

if __name__ == "__main__":
    main()