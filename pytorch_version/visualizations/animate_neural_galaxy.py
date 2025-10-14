#!/usr/bin/env python3
"""
Animated Neural Galaxy Evolution for PINN
Creates stunning animated galaxies showing how neural activations evolve during training
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import glob
from visualize_network_graph import NetworkVisualizer
from PIL import Image
import matplotlib.patches as mpatches

# Copy the physics constants and PINN model from original script
device = torch.device('cpu')
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

# Domain bounds
xmin, xmax = 0., 1.
ymin, ymax = 0., 1.
lb = torch.tensor([xmin, ymin], dtype=DTYPE, device=device)
ub = torch.tensor([xmax, ymax], dtype=DTYPE, device=device)

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
        
        # Forward through hidden layers with activation tracking
        h = x_scaled
        activations = [h]  # Store input
        
        for layer in self.hidden_layers:
            h = layer(h)
            activations.append(h)  # Store pre-activation
            h = torch.tanh(h)
            activations.append(h)  # Store post-activation
        
        # Outputs
        Ux = self.output_Ux(h)
        Uy = self.output_Uy(h)
        Sxx = self.output_Sxx(h)
        Syy = self.output_Syy(h)
        Sxy = self.output_Sxy(h)
        
        return Ux, Uy, Sxx, Syy, Sxy, activations

class NeuralGalaxyAnimator:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        
    def load_model_at_epoch(self, epoch_file):
        """Load a specific model snapshot"""
        model = PINN().to(device)
        model.load_state_dict(torch.load(epoch_file, map_location=device))
        model.eval()
        return model
    
    def get_layer_activations(self, model, input_sample):
        """Get activations for all layers with a specific input"""
        
        with torch.no_grad():
            # Forward pass with activation tracking
            Ux, Uy, Sxx, Syy, Sxy, activations = model(input_sample)
            
            # Extract hidden layer activations (post-tanh)
            hidden_activations = []
            for i in range(0, len(activations) - 1, 2):  # Skip pre-activation, take post-tanh
                if i + 1 < len(activations):
                    hidden_activations.append(activations[i + 1])
            
            # Add output activations
            outputs = torch.cat([Ux, Uy, Sxx, Syy, Sxy], dim=1)
            hidden_activations.append(outputs)
            
            return hidden_activations
    
    def create_galaxy_spiral_positions(self, n_neurons, galaxy_type='spiral'):
        """Create different galaxy formations"""
        
        if n_neurons == 1:
            return np.array([0]), np.array([0])
        
        if galaxy_type == 'spiral':
            # Spiral galaxy
            angles = np.linspace(0, 6*np.pi, n_neurons, endpoint=False)
            radii = 0.3 + 1.2 * angles / (6*np.pi)
            x_pos = radii * np.cos(angles)
            y_pos = radii * np.sin(angles)
            
        elif galaxy_type == 'elliptical':
            # Elliptical galaxy
            angles = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)
            a, b = 1.5, 0.8  # Ellipse parameters
            x_pos = a * np.cos(angles)
            y_pos = b * np.sin(angles)
            
        elif galaxy_type == 'cluster':
            # Globular cluster
            # Create multiple small clusters
            n_clusters = max(1, n_neurons // 5)
            cluster_angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
            cluster_centers_x = 0.8 * np.cos(cluster_angles)
            cluster_centers_y = 0.8 * np.sin(cluster_angles)
            
            x_pos = []
            y_pos = []
            
            for i in range(n_neurons):
                cluster_idx = i % n_clusters
                # Add small random offset around cluster center
                offset_angle = np.random.uniform(0, 2*np.pi)
                offset_radius = np.random.uniform(0, 0.3)
                
                x_pos.append(cluster_centers_x[cluster_idx] + offset_radius * np.cos(offset_angle))
                y_pos.append(cluster_centers_y[cluster_idx] + offset_radius * np.sin(offset_angle))
            
            x_pos = np.array(x_pos)
            y_pos = np.array(y_pos)
        
        return x_pos, y_pos
    
    def create_galaxy_frame(self, epoch_file, input_sample, galaxy_types=None):
        """Create a single galaxy frame for a specific epoch"""
        
        epoch = int(epoch_file.split('_')[-1].split('.')[0])
        print(f"🌌 Creating neural galaxy for epoch {epoch}")
        
        # Load model
        model = self.load_model_at_epoch(epoch_file)
        
        # Get activations
        activations = self.get_layer_activations(model, input_sample)
        
        # Default galaxy types for each layer
        if galaxy_types is None:
            galaxy_types = ['spiral'] * 8 + ['cluster']  # 8 hidden + 1 output
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        
        # Plot first 8 hidden layers + output layer
        for layer_idx in range(min(9, len(activations))):
            ax = axes[layer_idx]
            
            layer_activations = activations[layer_idx].numpy().flatten()
            n_neurons = len(layer_activations)
            
            # Get galaxy positions
            galaxy_type = galaxy_types[layer_idx] if layer_idx < len(galaxy_types) else 'spiral'
            x_pos, y_pos = self.create_galaxy_spiral_positions(n_neurons, galaxy_type)
            
            # Normalize activations for brightness
            act_normalized = np.abs(layer_activations)
            if act_normalized.max() > 0:
                act_normalized = act_normalized / act_normalized.max()
            
            # Star sizes and colors based on activation strength
            sizes = 50 + 400 * act_normalized if n_neurons > 1 else [300]
            colors = act_normalized if n_neurons > 1 else [1.0]
            
            # Create the galaxy
            scatter = ax.scatter(x_pos, y_pos, 
                               s=sizes,
                               c=colors,
                               cmap='plasma',  # Galaxy-like colors
                               alpha=0.9,
                               edgecolors='white',
                               linewidth=0.8)
            
            # Add gravitational field connections between strong neurons
            if n_neurons > 1:
                # Connect strongest neurons with "gravitational bonds"
                strong_threshold = np.percentile(act_normalized, 70)
                strong_neurons = np.where(act_normalized > strong_threshold)[0]
                
                for i in range(len(strong_neurons)):
                    for j in range(i + 1, min(i + 4, len(strong_neurons))):  # Limit connections
                        idx1, idx2 = strong_neurons[i], strong_neurons[j]
                        
                        # Connection strength based on both activations
                        connection_strength = (act_normalized[idx1] + act_normalized[idx2]) / 2
                        
                        # Draw gravitational field line
                        ax.plot([x_pos[idx1], x_pos[idx2]], [y_pos[idx1], y_pos[idx2]], 
                               'cyan', alpha=0.3 + 0.4 * connection_strength, 
                               linewidth=0.5 + 2 * connection_strength)
            
            # Galaxy styling
            ax.set_facecolor('black')
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Layer titles with cosmic theme
            if layer_idx < 8:
                layer_title = f'Hidden Galaxy {layer_idx + 1}\n{n_neurons} Stars'
                if epoch < 100:
                    cosmic_phase = "🌑 Dark Nebula"
                elif epoch < 500:
                    cosmic_phase = "💫 Star Formation"
                elif epoch < 1000:
                    cosmic_phase = "⭐ Stellar Ignition"
                else:
                    cosmic_phase = "🌟 Mature Galaxy"
            else:
                layer_title = f'Output Constellation\n{n_neurons} Stars'
                cosmic_phase = "🎯 Final Destination"
            
            ax.set_title(f'{layer_title}\n{cosmic_phase}', 
                        color='white', fontsize=11, fontweight='bold')
            
            # Add activation statistics
            if n_neurons > 1:
                stats_text = f"Max: {act_normalized.max():.3f}\nMean: {act_normalized.mean():.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       color='white', fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='navy', alpha=0.7))
        
        # Overall cosmic title
        input_str = f"[{input_sample[0,0]:.2f}, {input_sample[0,1]:.2f}]"
        
        if epoch < 100:
            cosmic_era = "🌌 Big Bang Era"
        elif epoch < 500:
            cosmic_era = "💫 Galaxy Formation"
        elif epoch < 1000:
            cosmic_era = "⭐ Stellar Evolution"
        else:
            cosmic_era = "🌟 Cosmic Harmony"
        
        fig.suptitle(f'Neural Galaxy Evolution - Epoch {epoch}\n'
                    f'{cosmic_era} | Input: {input_str}', 
                    color='white', fontsize=16, fontweight='bold')
        
        # Set black background
        fig.patch.set_facecolor('black')
        
        return fig
    
    def create_galaxy_animation(self, input_samples=None, galaxy_types=None):
        """Create animated neural galaxies"""
        print("🌌 Creating animated neural galaxy evolution...")
        
        # Default input samples if none provided
        if input_samples is None:
            input_samples = [
                torch.tensor([[0.5, 0.5]], dtype=DTYPE, device=device),  # Center
                torch.tensor([[0.2, 0.8]], dtype=DTYPE, device=device),  # Top-left
                torch.tensor([[0.8, 0.2]], dtype=DTYPE, device=device)   # Bottom-right
            ]
        
        # Find all model snapshots
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found")
            return
        
        # Create output directories
        for i in range(len(input_samples)):
            os.makedirs(f'galaxy_frames_input_{i+1}', exist_ok=True)
        
        # Subsample for animation performance
        step = max(1, len(model_files) // 30)  # Max 30 frames
        selected_files = model_files[::step]
        
        print(f"🎬 Processing {len(selected_files)} epochs for {len(input_samples)} input samples...")
        
        # Create frames for each input sample
        for input_idx, input_sample in enumerate(input_samples):
            print(f"\n🌟 Creating galaxy animation for input sample {input_idx + 1}")
            
            for frame_idx, model_file in enumerate(selected_files):
                epoch = int(model_file.split('_')[-1].split('.')[0])
                
                try:
                    fig = self.create_galaxy_frame(model_file, input_sample, galaxy_types)
                    
                    plt.savefig(f'galaxy_frames_input_{input_idx + 1}/galaxy_frame_{frame_idx:04d}.png', 
                               dpi=100, bbox_inches='tight', facecolor='black')
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"⚠️ Error processing epoch {epoch} for input {input_idx + 1}: {e}")
                    continue
        
        # Create GIFs for each input
        self.create_galaxy_gifs(len(input_samples))
    
    def create_galaxy_gifs(self, num_inputs):
        """Create GIFs from galaxy frames"""
        print("🎬 Creating neural galaxy GIFs...")
        
        os.makedirs('animations', exist_ok=True)
        
        for input_idx in range(num_inputs):
            frame_files = sorted(glob.glob(f'galaxy_frames_input_{input_idx + 1}/galaxy_frame_*.png'))
            
            if frame_files:
                images = [Image.open(f) for f in frame_files]
                
                # Regular speed
                images[0].save(f'animations/neural_galaxy_input_{input_idx + 1}.gif', 
                              save_all=True, append_images=images[1:], 
                              duration=250, loop=0)
                print(f"✅ Neural galaxy GIF created for input {input_idx + 1}")
                
                # Fast version
                fast_images = images[::2]
                if len(fast_images) > 1:
                    fast_images[0].save(f'animations/neural_galaxy_input_{input_idx + 1}_fast.gif', 
                                      save_all=True, append_images=fast_images[1:], 
                                      duration=150, loop=0)
                    print(f"✅ Fast neural galaxy GIF created for input {input_idx + 1}")
    
    def create_combined_galaxy_evolution(self):
        """Create a combined view showing multiple galaxies evolving side by side"""
        print("🌌 Creating combined galaxy evolution...")
        
        # Input samples
        input_samples = [
            torch.tensor([[0.3, 0.3]], dtype=DTYPE, device=device),  # Interior point
            torch.tensor([[0.0, 0.5]], dtype=DTYPE, device=device),  # Boundary point
            torch.tensor([[0.7, 0.9]], dtype=DTYPE, device=device)   # Corner region
        ]
        
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        if not model_files:
            return
        
        os.makedirs('combined_galaxy_frames', exist_ok=True)
        
        # Subsample
        step = max(1, len(model_files) // 25)
        selected_files = model_files[::step]
        
        for frame_idx, model_file in enumerate(selected_files):
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            try:
                model = self.load_model_at_epoch(model_file)
                
                # Create combined figure
                fig, axes = plt.subplots(3, 3, figsize=(18, 18))
                fig.patch.set_facecolor('black')
                
                # Show 3 different layers for 3 different inputs
                layers_to_show = [1, 4, 7]  # Early, middle, late hidden layers
                
                for input_idx, input_sample in enumerate(input_samples):
                    activations = self.get_layer_activations(model, input_sample)
                    
                    for layer_plot_idx, layer_idx in enumerate(layers_to_show):
                        if layer_idx < len(activations):
                            ax = axes[input_idx, layer_plot_idx]
                            
                            layer_activations = activations[layer_idx].numpy().flatten()
                            n_neurons = len(layer_activations)
                            
                            # Different galaxy types for variety
                            galaxy_types = ['spiral', 'elliptical', 'cluster']
                            galaxy_type = galaxy_types[layer_plot_idx]
                            
                            x_pos, y_pos = self.create_galaxy_spiral_positions(n_neurons, galaxy_type)
                            
                            # Normalize activations
                            act_normalized = np.abs(layer_activations)
                            if act_normalized.max() > 0:
                                act_normalized = act_normalized / act_normalized.max()
                            
                            # Plot galaxy
                            sizes = 30 + 200 * act_normalized
                            scatter = ax.scatter(x_pos, y_pos, 
                                               s=sizes,
                                               c=act_normalized,
                                               cmap='plasma',
                                               alpha=0.9,
                                               edgecolors='white',
                                               linewidth=0.5)
                            
                            # Add connections between bright stars
                            bright_threshold = np.percentile(act_normalized, 80)
                            bright_neurons = np.where(act_normalized > bright_threshold)[0]
                            
                            for i in range(len(bright_neurons)):
                                for j in range(i + 1, min(i + 3, len(bright_neurons))):
                                    idx1, idx2 = bright_neurons[i], bright_neurons[j]
                                    strength = (act_normalized[idx1] + act_normalized[idx2]) / 2
                                    
                                    ax.plot([x_pos[idx1], x_pos[idx2]], [y_pos[idx1], y_pos[idx2]], 
                                           'cyan', alpha=0.2 + 0.5 * strength, 
                                           linewidth=0.3 + 1.5 * strength)
                            
                            # Styling
                            ax.set_facecolor('black')
                            ax.set_xlim(-2.2, 2.2)
                            ax.set_ylim(-2.2, 2.2)
                            ax.set_aspect('equal')
                            ax.axis('off')
                            
                            # Titles
                            input_str = f"[{input_sample[0,0]:.1f}, {input_sample[0,1]:.1f}]"
                            layer_name = f"Layer {layer_idx + 1}" if layer_idx < 8 else "Output"
                            
                            ax.set_title(f'{layer_name} | {galaxy_type.title()}\nInput: {input_str}', 
                                        color='white', fontsize=10, fontweight='bold')
                
                # Overall title with cosmic theme
                if epoch < 100:
                    cosmic_era = "🌑 Primordial Darkness"
                elif epoch < 500:
                    cosmic_era = "💫 Cosmic Dawn"
                elif epoch < 1000:
                    cosmic_era = "⭐ Galaxy Formation"
                else:
                    cosmic_era = "🌟 Stellar Maturity"
                
                fig.suptitle(f'Neural Galaxy Universe - Epoch {epoch}\n{cosmic_era}', 
                            color='white', fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f'combined_galaxy_frames/combined_galaxy_{frame_idx:04d}.png', 
                           dpi=100, bbox_inches='tight', facecolor='black')
                plt.close(fig)
                
            except Exception as e:
                print(f"⚠️ Error processing epoch {epoch}: {e}")
                continue
        
        # Create combined GIF
        combined_files = sorted(glob.glob('combined_galaxy_frames/combined_galaxy_*.png'))
        if combined_files:
            images = [Image.open(f) for f in combined_files]
            images[0].save('animations/neural_galaxy_universe.gif', save_all=True,
                          append_images=images[1:], duration=300, loop=0)
            print("✅ Combined neural galaxy universe GIF created!")

def main():
    """Main function"""
    print("🌌 Animated Neural Galaxy Visualizer")
    print("=" * 50)
    
    animator = NeuralGalaxyAnimator()
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    print("\n🌌 Creating individual galaxy animations...")
    animator.create_galaxy_animation()
    
    print("\n🌌 Creating combined galaxy universe...")
    animator.create_combined_galaxy_evolution()
    
    print("\n🎉 Neural galaxy animation complete!")
    print("📁 Generated files:")
    print("   📂 animations/neural_galaxy_input_*.gif - Individual input galaxies")
    print("   📂 animations/neural_galaxy_input_*_fast.gif - Fast versions")
    print("   📂 animations/neural_galaxy_universe.gif - Combined universe view")
    print("   📂 galaxy_frames_input_*/ - Individual frames")
    print("   📂 combined_galaxy_frames/ - Universe frames")
    print("\n🌌 Watch your PINN's neurons evolve like a cosmic symphony!")

if __name__ == "__main__":
    main()