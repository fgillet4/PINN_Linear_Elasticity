#!/usr/bin/env python3
"""
3D Weight Evolution Visualizer for PINN
Creates a "loaf of bread" style 3D visualization showing weight matrix slices changing over time
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

class Weight3DVisualizer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        self.weight_data = []
        self.epochs = []
        
    def load_weight_snapshots(self):
        """Load all weight snapshots from training"""
        print("📂 Loading weight snapshots...")
        
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
                
                # Extract weight matrices for each layer
                layer_weights = {}
                for layer_name, info in visualizer.layer_info.items():
                    if 'hidden_layers' in layer_name:
                        weights = info['weight_tensor'].detach().numpy()
                        layer_weights[layer_name] = weights
                
                self.weight_data.append(layer_weights)
                self.epochs.append(epoch)
                
        print(f"✅ Loaded {len(self.weight_data)} weight snapshots")
        return len(self.weight_data) > 0
    
    def create_3d_weight_loaf(self, layer_name='hidden_layers.0', save_frames=True):
        """Create 3D visualization of weight matrices as bread slices"""
        print(f"🍞 Creating 3D weight loaf for {layer_name}...")
        
        if not self.weight_data:
            print("❌ No weight data loaded")
            return
            
        # Create output directory
        os.makedirs('3d_frames', exist_ok=True)
        
        # Get weight dimensions from first snapshot
        first_weights = self.weight_data[0][layer_name]
        height, width = first_weights.shape  # e.g., (20, 20) for hidden layers
        
        print(f"📊 Weight matrix shape: {height} × {width}")
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids
        x = np.arange(width)   # Input neurons
        y = np.arange(height)  # Output neurons
        X, Y = np.meshgrid(x, y)
        
        # Animation function
        def animate_frame(frame_idx):
            ax.clear()
            
            epoch = self.epochs[frame_idx]
            weights = self.weight_data[frame_idx][layer_name]
            
            # Create multiple slices at different Z positions
            z_positions = np.linspace(0, 10, 20)  # 20 slices spread across Z
            
            # Color map for weights
            vmin, vmax = -1.0, 1.0  # Fixed color scale for consistency
            norm = Normalize(vmin=vmin, vmax=vmax)
            
            # Plot multiple slices to create "loaf" effect
            for i, z_pos in enumerate(z_positions):
                # Create slight variations in the weight matrix for each slice
                slice_weights = weights + 0.1 * np.sin(i * 0.5) * np.random.normal(0, 0.05, weights.shape)
                
                # Create the surface plot
                surf = ax.plot_surface(X, Y, np.full_like(X, z_pos), 
                                     facecolors=plt.cm.RdBu_r(norm(slice_weights)),
                                     alpha=0.7, antialiased=True)
            
            # Set labels and title
            ax.set_xlabel('Input Neurons')
            ax.set_ylabel('Output Neurons') 
            ax.set_zlabel('Layer Depth')
            ax.set_title(f'3D Weight Evolution - {layer_name}\nEpoch {epoch}', 
                        fontsize=14, fontweight='bold')
            
            # Set consistent axis limits
            ax.set_xlim(0, width-1)
            ax.set_ylim(0, height-1)
            ax.set_zlim(0, 10)
            
            # Add colorbar (only on first frame)
            if frame_idx == 0:
                mappable = plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r')
                mappable.set_array([])
                cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, aspect=30)
                cbar.set_label('Weight Value', rotation=270, labelpad=20)
            
            # Save frame if requested
            if save_frames:
                plt.savefig(f'3d_frames/frame_{frame_idx:04d}.png', 
                           dpi=100, bbox_inches='tight')
            
            return []
        
        # Create animation
        print(f"🎬 Creating animation with {len(self.epochs)} frames...")
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(self.epochs),
                                     interval=200, blit=False, repeat=True)
        
        # Save as MP4 (if ffmpeg available) or GIF
        try:
            anim.save('animations/weight_3d_loaf.mp4', writer='ffmpeg', fps=5)
            print("✅ 3D weight loaf animation saved as 'weight_3d_loaf.mp4'")
        except:
            print("⚠️ ffmpeg not available, saving individual frames only")
            
        # Create GIF from saved frames
        if save_frames:
            self.create_gif_from_frames()
            
        return anim
    
    def create_true_bread_slices(self, save_path='animations/weight_bread_slices.gif'):
        """Create true bread slice visualization - each hidden layer as a slice"""
        print("🍞 Creating true bread slice visualization...")
        
        if not self.weight_data:
            print("❌ No weight data loaded")
            return
            
        os.makedirs('bread_frames', exist_ok=True)
        
        # Get all hidden layer names
        hidden_layers = [name for name in self.weight_data[0].keys() if 'hidden_layers' in name]
        hidden_layers.sort(key=lambda x: int(x.split('.')[1]))
        
        print(f"🧅 Found {len(hidden_layers)} hidden layers for bread slices")
        
        frames = []
        
        for frame_idx, epoch in enumerate(self.epochs):
            print(f"🖼️ Creating frame {frame_idx+1}/{len(self.epochs)} (epoch {epoch})")
            
            # Create figure with subplots for each layer slice
            n_layers = len(hidden_layers)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 grid for 8 layers
            axes = axes.flatten()
            
            # Plot each layer as a heatmap slice
            for i, layer_name in enumerate(hidden_layers):
                weights = self.weight_data[frame_idx][layer_name]
                
                # Create heatmap
                im = axes[i].imshow(weights, cmap='RdBu_r', vmin=-1.0, vmax=1.0, 
                                   aspect='auto', interpolation='bilinear')
                axes[i].set_title(f'Layer {i+1}\n{layer_name}', fontsize=10, fontweight='bold')
                axes[i].set_xlabel('Input Neurons')
                axes[i].set_ylabel('Output Neurons')
                
                # Add grid
                axes[i].grid(True, alpha=0.3)
            
            # Add overall title
            fig.suptitle(f'PINN Weight "Bread Slices" - Epoch {epoch}\nEach slice = Hidden Layer Weight Matrix', 
                        fontsize=16, fontweight='bold')
            
            # Adjust layout to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.85, 0.96])  # Leave space on right for colorbar
            
            # Add colorbar positioned cleanly on the right
            cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Weight Value', rotation=270, labelpad=15)
            
            # Save frame
            frame_path = f'bread_frames/bread_frame_{frame_idx:04d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # Store for GIF creation
            frames.append(Image.open(frame_path))
        
        # Create GIF
        if frames:
            frames[0].save(save_path, save_all=True, append_images=frames[1:],
                          duration=300, loop=0)
            print(f"✅ Bread slices GIF saved as '{save_path}'")
            
            # Create fast version
            fast_frames = frames[::3]  # Every 3rd frame
            if len(fast_frames) > 1:
                fast_path = save_path.replace('.gif', '_fast.gif')
                fast_frames[0].save(fast_path, save_all=True, append_images=fast_frames[1:],
                                  duration=200, loop=0)
                print(f"✅ Fast bread slices GIF saved as '{fast_path}'")
        
        return frames
    
    def create_gif_from_frames(self):
        """Create GIF from saved 3D frames"""
        frame_files = sorted(glob.glob('3d_frames/frame_*.png'))
        
        if not frame_files:
            print("❌ No 3D frames found")
            return
            
        # Create GIF
        images = [Image.open(f) for f in frame_files]
        if images:
            images[0].save('animations/weight_3d_loaf.gif', save_all=True,
                          append_images=images[1:], duration=200, loop=0)
            print("✅ 3D weight loaf GIF saved as 'weight_3d_loaf.gif'")

def main():
    """Main function to create 3D weight visualizations"""
    print("🚀 3D Weight Evolution Visualizer")
    print("=" * 50)
    
    # Create visualizer
    visualizer = Weight3DVisualizer()
    
    # Load weight snapshots
    if not visualizer.load_weight_snapshots():
        print("❌ Failed to load weight snapshots")
        print("💡 Run 'python3 train_with_weight_animation.py' first to generate snapshots")
        return
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    # Create true bread slice visualization (recommended)
    print("\n🍞 Creating bread slice visualization...")
    visualizer.create_true_bread_slices()
    
    # Optionally create 3D loaf (more experimental)
    # visualizer.create_3d_weight_loaf()
    
    print("\n🎉 3D Weight visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/weight_bread_slices.gif - Layer-by-layer weight evolution")
    print("   📂 animations/weight_bread_slices_fast.gif - Faster version")
    print("   📂 bread_frames/ - Individual frames")

if __name__ == "__main__":
    main()