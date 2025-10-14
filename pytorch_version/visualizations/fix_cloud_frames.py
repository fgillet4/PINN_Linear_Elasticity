#!/usr/bin/env python3
"""
Quick fix to generate the missing cloud frames
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

def create_cloud_frames_fixed():
    """Create the missing cloud frames"""
    print("🔧 Fixing cloud frame generation...")
    
    # Load weight snapshots
    model_files = sorted(glob.glob('weight_snapshots/temp_model_epoch_*.pt'))
    
    if not model_files:
        print("❌ No model snapshots found")
        return
        
    weight_data = []
    epochs = []
    
    # Load data
    for model_file in model_files:
        epoch = int(model_file.split('_')[-1].split('.')[0])
        
        visualizer = NetworkVisualizer(model_file)
        if visualizer.load_model():
            visualizer.analyze_architecture()
            
            # Extract all weights as 3D points
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
            
            weight_data.append({
                'weights': np.array(all_weights),
                'positions': np.array(positions)
            })
            epochs.append(epoch)
    
    print(f"✅ Loaded {len(weight_data)} snapshots")
    
    # Create interpolated frames
    interpolated_data = []
    interpolated_epochs = []
    
    for i in range(len(weight_data) - 1):
        interpolated_data.append(weight_data[i])
        interpolated_epochs.append(epochs[i])
        
        # Add 3 interpolated frames
        for j in range(1, 4):
            alpha = j / 4
            weights_start = weight_data[i]['weights']
            weights_end = weight_data[i + 1]['weights']
            epoch_start = epochs[i]
            epoch_end = epochs[i + 1]
            
            interpolated_weights = (1 - alpha) * weights_start + alpha * weights_end
            interpolated_epoch = (1 - alpha) * epoch_start + alpha * epoch_end
            
            interpolated_data.append({
                'weights': interpolated_weights,
                'positions': weight_data[i]['positions']
            })
            interpolated_epochs.append(interpolated_epoch)
    
    # Add final frame
    interpolated_data.append(weight_data[-1])
    interpolated_epochs.append(epochs[-1])
    
    print(f"✅ Created {len(interpolated_data)} interpolated frames")
    
    # Create cloud frames
    os.makedirs('cloud_frames', exist_ok=True)
    
    # Get data bounds
    all_weights = np.concatenate([data['weights'] for data in interpolated_data])
    vmin, vmax = np.percentile(all_weights, [5, 95])
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    positions = interpolated_data[0]['positions']
    max_x, max_y, max_z = positions.max(axis=0)
    
    for frame_idx, epoch in enumerate(interpolated_epochs):
        print(f"☁️ Creating cloud frame {frame_idx+1}/{len(interpolated_epochs)}")
        
        weights = interpolated_data[frame_idx]['weights']
        positions = interpolated_data[frame_idx]['positions']
        
        # Filter significant weights
        significant_mask = np.abs(weights) > 0.05
        sig_weights = weights[significant_mask]
        sig_positions = positions[significant_mask]
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color mapping
        colors = plt.cm.RdBu_r(norm(sig_weights))
        
        # Create scatter plot
        ax.scatter(sig_positions[:, 0], sig_positions[:, 1], sig_positions[:, 2],
                  c=colors, s=30, alpha=0.8, edgecolors='black', linewidth=0.1)
        
        # Labels and title
        ax.set_xlabel('Input Neurons →', fontsize=12, fontweight='bold')
        ax.set_ylabel('Output Neurons →', fontsize=12, fontweight='bold')
        ax.set_zlabel('Network Depth →', fontsize=12, fontweight='bold')
        
        # Phase detection
        if epoch < 100:
            phase = "🌱 Initialization"
        elif epoch < 500:
            phase = "🔥 Rapid Learning"
        elif epoch < 1000:
            phase = "⚡ Optimization"
        else:
            phase = "🎯 Fine-tuning"
        
        ax.set_title(f'3D PINN Weight Cloud Evolution\n{phase} - Epoch {epoch:.1f}\n'
                    f'Active Points: {len(sig_weights)}/{len(weights)}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set limits
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        ax.set_zlim(0, max_z)
        
        # Fixed camera angle - no more dizziness!
        ax.view_init(elev=25, azim=45)  # Nice 3D perspective showing all axes
        
        # Add statistics
        stats_text = f"Active Weights: {len(sig_weights)}/{len(weights)}\n"
        stats_text += f"Mean: {sig_weights.mean():.3f}\n"
        stats_text += f"Std: {sig_weights.std():.3f}"
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Save frame
        plt.savefig(f'cloud_frames/cloud_frame_{frame_idx:04d}.png', 
                   dpi=100, bbox_inches='tight', facecolor='black')
        plt.close(fig)
    
    # Create GIF
    print("🎬 Creating cloud GIF...")
    frame_files = sorted(glob.glob('cloud_frames/cloud_frame_*.png'))
    
    if frame_files:
        images = [Image.open(f) for f in frame_files]
        images[0].save('animations/weight_3d_cloud.gif', save_all=True,
                      append_images=images[1:], duration=100, loop=0)
        print("✅ 3D cloud GIF created!")
        
        # Fast version
        fast_images = images[::3]
        if len(fast_images) > 1:
            fast_images[0].save('animations/weight_3d_cloud_fast.gif', save_all=True,
                              append_images=fast_images[1:], duration=80, loop=0)
            print("✅ Fast 3D cloud GIF created!")
    
    print(f"\n🎉 Fixed! Created {len(frame_files)} cloud frames")

if __name__ == "__main__":
    create_cloud_frames_fixed()