#!/usr/bin/env python3
"""
Weight Flow Visualizer for PINN
Creates flowing, stream-like visualization showing how weights connect and flow through layers
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

class WeightFlowVisualizer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        self.weight_data = []
        self.epochs = []
        
    def load_weight_snapshots(self):
        """Load all weight snapshots"""
        print("🌊 Loading weight snapshots for flow visualization...")
        
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found")
            return False
            
        for model_file in model_files:
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            visualizer = NetworkVisualizer(model_file)
            if visualizer.load_model():
                visualizer.analyze_architecture()
                
                # Store layer-by-layer weight matrices
                layer_weights = {}
                layer_order = []
                
                for layer_name, info in visualizer.layer_info.items():
                    if 'hidden_layers' in layer_name or layer_name.startswith('output'):
                        weights = info['weight_tensor'].detach().numpy()
                        layer_weights[layer_name] = weights
                        if 'hidden_layers' in layer_name:
                            layer_order.append(layer_name)
                
                # Sort hidden layers properly
                layer_order.sort(key=lambda x: int(x.split('.')[1]))
                
                # Add output layers
                output_layers = [name for name in layer_weights.keys() if 'output' in name]
                layer_order.extend(sorted(output_layers))
                
                self.weight_data.append({
                    'weights': layer_weights,
                    'layer_order': layer_order
                })
                self.epochs.append(epoch)
                
        print(f"✅ Loaded {len(self.weight_data)} snapshots for flow visualization")
        return len(self.weight_data) > 0
    
    def create_flow_field(self, layer_weights, layer_order, resolution=40):
        "reate a 3D flow field showing weight connections between layers"""
        
        # Create 3D grid spanning the network
        max_neurons = max([weights.shape[0] for weights in layer_weights.values()] + 
                         [weights.shape[1] for weights in layer_weights.values()])
        
        x = np.linspace(0, max_neurons-1, resolution)
        y = np.linspace(0, max_neurons-1, resolution)
        z = np.linspace(0, len(layer_order)-1, resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create flow vectors and magnitudes
        flow_magnitude = np.zeros_like(X)
        flow_u = np.zeros_like(X)  # X-direction flow
        flow_v = np.zeros_like(X)  # Y-direction flow
        flow_w = np.zeros_like(X)  # Z-direction flow (layer-to-layer)
        
        # Fill in the flow field based on actual weights
        for i, layer_name in enumerate(layer_order[:-1]):  # All except last layer
            current_weights = layer_weights[layer_name]
            
            # Get layer dimensions
            out_neurons, in_neurons = current_weights.shape
            
            # Create flow from this layer to next layer
            for out_idx in range(min(out_neurons, resolution)):
                for in_idx in range(min(in_neurons, resolution)):
                    weight_val = current_weights[out_idx, in_idx]
                    
                    # Map to grid coordinates
                    z_idx = int(i * resolution / len(layer_order))
                    x_idx = int(in_idx * resolution / max_neurons)
                    y_idx = int(out_idx * resolution / max_neurons)
                    
                    if (z_idx < resolution-1 and x_idx < resolution and y_idx < resolution):
                        # Set flow magnitude
                        flow_magnitude[x_idx, y_idx, z_idx] = abs(weight_val)
                        
                        # Create flow toward next layer (Z-direction)
                        flow_w[x_idx, y_idx, z_idx] = weight_val * 0.5
                        
                        # Add some lateral flow based on weight patterns
                        flow_u[x_idx, y_idx, z_idx] = weight_val * 0.1 * np.sin(in_idx)
                        flow_v[x_idx, y_idx, z_idx] = weight_val * 0.1 * np.cos(out_idx)
        
        # Smooth the flow field for better visualization
        flow_magnitude = gaussian_filter(flow_magnitude, sigma=1.0)
        flow_u = gaussian_filter(flow_u, sigma=1.0)
        flow_v = gaussian_filter(flow_v, sigma=1.0)
        flow_w = gaussian_filter(flow_w, sigma=1.0)
        
        return X, Y, Z, flow_magnitude, flow_u, flow_v, flow_w
    
    def create_streamlines(self, X, Y, Z, flow_u, flow_v, flow_w, num_streams=30):
        """Create streamlines showing weight flow paths"""
        
        streamlines = []
        
        # Start streamlines from input layer (z=0)
        z_start = 0
        
        for i in range(num_streams):
            # Random starting points in input layer
            x_start = np.random.uniform(0, X.max())
            y_start = np.random.uniform(0, Y.max())
            
            # Trace streamline through the network
            stream_x = [x_start]
            stream_y = [y_start]
            stream_z = [z_start]
            
            x, y, z = x_start, y_start, z_start
            
            for step in range(100):  # Max 100 steps per streamline
                # Interpolate flow at current position
                try:
                    # Simple nearest neighbor for flow direction
                    xi = int(np.clip(x * X.shape[0] / X.max(), 0, X.shape[0]-1))
                    yi = int(np.clip(y * Y.shape[1] / Y.max(), 0, Y.shape[1]-1))
                    zi = int(np.clip(z * Z.shape[2] / Z.max(), 0, Z.shape[2]-1))
                    
                    dx = flow_u[xi, yi, zi] * 0.1
                    dy = flow_v[xi, yi, zi] * 0.1
                    dz = flow_w[xi, yi, zi] * 0.1 + 0.05  # Always move forward through layers
                    
                    # Update position
                    x += dx
                    y += dy
                    z += dz
                    
                    # Stop if we've gone through all layers
                    if z >= Z.max():
                        break
                        
                    stream_x.append(x)
                    stream_y.append(y)
                    stream_z.append(z)
                    
                except:
                    break
            
            if len(stream_x) > 3:  # Only keep streams with enough points
                streamlines.append((np.array(stream_x), np.array(stream_y), np.array(stream_z)))
        
        return streamlines
    
    def create_flow_frame(self, frame_idx, resolution=25, num_streams=40):
        """Create a single flow visualization frame"""
        
        epoch = self.epochs[frame_idx]
        layer_weights = self.weight_data[frame_idx]['weights']
        layer_order = self.weight_data[frame_idx]['layer_order']
        
        print(f"🌊 Creating flow frame {frame_idx+1}/{len(self.epochs)} (epoch {epoch})")
        
        # Create flow field
        X, Y, Z, flow_magnitude, flow_u, flow_v, flow_w = self.create_flow_field(
            layer_weights, layer_order, resolution
        )
        
        # Create streamlines
        streamlines = self.create_streamlines(X, Y, Z, flow_u, flow_v, flow_w, num_streams)
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot volume using scatter points with varying opacity
        threshold = np.percentile(flow_magnitude, 70)  # Only show significant flows
        significant_mask = flow_magnitude > threshold
        
        if np.any(significant_mask):
            xs, ys, zs = X[significant_mask], Y[significant_mask], Z[significant_mask]
            magnitudes = flow_magnitude[significant_mask]
            
            # Color by flow magnitude
            colors = plt.cm.viridis(magnitudes / magnitudes.max())
            
            # Size by magnitude
            sizes = 20 + 100 * (magnitudes / magnitudes.max())
            
            ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.6, edgecolors='none')
        
        # Plot streamlines
        cmap = plt.cm.coolwarm
        for i, (sx, sy, sz) in enumerate(streamlines):
            if len(sx) > 1:
                # Color streamline by flow direction/magnitude
                colors = cmap(i / len(streamlines))
                ax.plot(sx, sy, sz, color=colors, alpha=0.8, linewidth=2)
                
                # Add arrow at the end to show direction
                if len(sx) > 3:
                    ax.scatter(sx[-1], sy[-1], sz[-1], color=colors, s=50, marker='>', alpha=1.0)
        
        # Add layer planes for reference
        for i, layer_name in enumerate(layer_order):
            z_pos = i * Z.max() / (len(layer_order) - 1)
            
            # Create a semi-transparent plane
            xx, yy = np.meshgrid(np.linspace(0, X.max(), 10), np.linspace(0, Y.max(), 10))
            zz = np.full_like(xx, z_pos)
            
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
            
            # Label the layer
            layer_label = f"H{i+1}" if 'hidden' in layer_name else layer_name.split('_')[1]
            ax.text(X.max()/2, Y.max() + 2, z_pos, layer_label, fontsize=10, fontweight='bold')
        
        # Styling
        ax.set_xlabel('Input Dimension →', fontsize=12, fontweight='bold')
        ax.set_ylabel('Output Dimension →', fontsize=12, fontweight='bold')
        ax.set_zlabel('Network Depth →', fontsize=12, fontweight='bold')
        
        # Phase detection
        if epoch < 100:
            phase = "🌱 Initial Flow"
        elif epoch < 500:
            phase = "🌊 Turbulent Learning"
        elif epoch < 1000:
            phase = "💫 Organizing Streams"
        else:
            phase = "🎯 Steady Flow"
        
        ax.set_title(f'Neural Weight Flow Patterns\n{phase} - Epoch {epoch:.1f}\n'
                    f'Information Streams Through Network Layers', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Fixed viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Set limits
        ax.set_xlim(0, X.max())
        ax.set_ylim(0, Y.max())
        ax.set_zlim(0, Z.max())
        
        # Add flow statistics
        total_flow = np.sum(flow_magnitude)
        max_flow = np.max(flow_magnitude)
        active_streams = len(streamlines)
        
        stats_text = f"Total Flow: {total_flow:.2f}\n"
        stats_text += f"Max Flow: {max_flow:.3f}\n"
        stats_text += f"Active Streams: {active_streams}"
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.9))
        
        return fig, ax
    
    def create_flow_animation(self, resolution=20, num_streams=30):
        """Create flow animation"""
        print("🎬 Creating weight flow animation...")
        
        os.makedirs('flow_frames', exist_ok=True)
        
        # Create interpolated frames for smoother animation
        interpolated_data = []
        interpolated_epochs = []
        
        for i in range(len(self.weight_data) - 1):
            interpolated_data.append(self.weight_data[i])
            interpolated_epochs.append(self.epochs[i])
            
            # Add 2 interpolated frames
            for j in range(1, 3):
                alpha = j / 3
                
                # Interpolate weights for each layer
                interpolated_weights = {}
                for layer_name in self.weight_data[i]['weights'].keys():
                    weights_start = self.weight_data[i]['weights'][layer_name]
                    weights_end = self.weight_data[i + 1]['weights'][layer_name]
                    interpolated_weights[layer_name] = (1 - alpha) * weights_start + alpha * weights_end
                
                epoch_start = self.epochs[i]
                epoch_end = self.epochs[i + 1]
                interpolated_epoch = (1 - alpha) * epoch_start + alpha * epoch_end
                
                interpolated_data.append({
                    'weights': interpolated_weights,
                    'layer_order': self.weight_data[i]['layer_order']
                })
                interpolated_epochs.append(interpolated_epoch)
        
        # Add final frame
        interpolated_data.append(self.weight_data[-1])
        interpolated_epochs.append(self.epochs[-1])
        
        # Temporarily replace data
        original_data = self.weight_data
        original_epochs = self.epochs
        self.weight_data = interpolated_data
        self.epochs = interpolated_epochs
        
        # Create frames
        for i in range(len(self.epochs)):
            fig, ax = self.create_flow_frame(i, resolution, num_streams)
            plt.savefig(f'flow_frames/flow_frame_{i:04d}.png', 
                       dpi=100, bbox_inches='tight', facecolor='black')
            plt.close(fig)
        
        # Restore original data
        self.weight_data = original_data
        self.epochs = original_epochs
        
        # Create GIF
        print("🎬 Creating flow GIF...")
        frame_files = sorted(glob.glob('flow_frames/flow_frame_*.png'))
        
        if frame_files:
            images = [Image.open(f) for f in frame_files]
            images[0].save('animations/weight_flow_streams.gif', save_all=True,
                          append_images=images[1:], duration=120, loop=0)
            print("✅ Weight flow GIF created!")
            
            # Fast version
            fast_images = images[::2]
            if len(fast_images) > 1:
                fast_images[0].save('animations/weight_flow_streams_fast.gif', save_all=True,
                                  append_images=fast_images[1:], duration=100, loop=0)
                print("✅ Fast weight flow GIF created!")

def main():
    """Main function"""
    print("🌊 Neural Weight Flow Visualizer")
    print("=" * 50)
    
    visualizer = WeightFlowVisualizer()
    
    if not visualizer.load_weight_snapshots():
        return
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    print("\n🌊 Creating weight flow visualization...")
    visualizer.create_flow_animation(resolution=20, num_streams=25)
    
    print("\n🎉 Weight flow visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/weight_flow_streams.gif - Neural weight flow patterns")
    print("   📂 animations/weight_flow_streams_fast.gif - Faster version")
    print("   📂 flow_frames/ - Individual frames")

if __name__ == "__main__":
    main()