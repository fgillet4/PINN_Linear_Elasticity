#!/usr/bin/env python3
"""
Neural Microscopy Time-Lapse Visualizer for PINN
Zoom into individual neurons like biological cells, watch their "growth" and "connections"
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import Normalize
import os
import glob
from visualize_network_graph import NetworkVisualizer
from PIL import Image
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

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
        activations = [h.clone()]
        
        for layer in self.hidden_layers:
            h = layer(h)
            activations.append(h.clone())  # Pre-activation
            h = torch.tanh(h)
            activations.append(h.clone())  # Post-activation
        
        # Outputs
        Ux = self.output_Ux(h)
        Uy = self.output_Uy(h)
        Sxx = self.output_Sxx(h)
        Syy = self.output_Syy(h)
        Sxy = self.output_Sxy(h)
        
        return Ux, Uy, Sxx, Syy, Sxy, activations

class NeuralMicroscopyVisualizer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        self.weight_evolution = {}
        self.activation_evolution = {}
        self.epochs = []
        
    def load_model_at_epoch(self, epoch_file):
        """Load a specific model snapshot"""
        model = PINN().to(device)
        model.load_state_dict(torch.load(epoch_file, map_location=device))
        model.eval()
        return model
    
    def get_neuron_characteristics(self, weights_in, weights_out, bias, activation):
        """Analyze individual neuron characteristics like a biological cell"""
        
        # Cell size based on total connection strength
        total_input_strength = np.sum(np.abs(weights_in))
        total_output_strength = np.sum(np.abs(weights_out)) if weights_out is not None else 0
        cell_size = np.sqrt(total_input_strength + total_output_strength)
        
        # Cell activity (brightness) based on activation
        cell_activity = float(np.abs(activation))
        
        # Cell health based on weight distribution
        input_diversity = np.std(weights_in) if len(weights_in) > 1 else 0
        cell_health = min(1.0, input_diversity * 10)  # Scale for visualization
        
        # Cell specialization based on weight pattern
        specialization = np.max(np.abs(weights_in)) / (np.mean(np.abs(weights_in)) + 1e-10)
        
        # Connection strength to other neurons
        connection_strengths = np.abs(weights_out) if weights_out is not None else np.array([])
        
        return {
            'size': cell_size,
            'activity': cell_activity,
            'health': cell_health,
            'specialization': specialization,
            'bias': float(bias),
            'connections': connection_strengths,
            'input_weights': weights_in,
            'output_weights': weights_out
        }
    
    def create_cellular_layer_view(self, model, layer_idx=1, test_input=None):
        """Create detailed cellular view of a specific layer"""
        
        if test_input is None:
            test_input = torch.tensor([[0.5, 0.5]], dtype=DTYPE, device=device)
        
        # Get activations
        with torch.no_grad():
            Ux, Uy, Sxx, Syy, Sxy, activations = model(test_input)
        
        # Extract layer information
        if layer_idx >= len(model.hidden_layers):
            return None
            
        current_layer = model.hidden_layers[layer_idx]
        weights_in = current_layer.weight.detach().numpy()  # Shape: [out_neurons, in_neurons]
        bias = current_layer.bias.detach().numpy()
        
        # Get next layer weights if available
        weights_out = None
        if layer_idx + 1 < len(model.hidden_layers):
            next_layer = model.hidden_layers[layer_idx + 1]
            next_weights = next_layer.weight.detach().numpy()  # Shape: [out_next, in_current]
            weights_out = next_weights.T  # Transpose to [in_current, out_next]
        elif layer_idx + 1 == len(model.hidden_layers):
            # Connect to output layers - shape: [20, 5]
            output_weights = []
            for output_layer in [model.output_Ux, model.output_Uy, model.output_Sxx, model.output_Syy, model.output_Sxy]:
                output_weights.append(output_layer.weight.detach().numpy().flatten())
            weights_out = np.column_stack(output_weights)  # Shape: [20, 5]
        
        # Get layer activations (post-tanh)
        layer_activations = activations[2 + layer_idx * 2].numpy().flatten()  # Skip input, get post-tanh
        
        # Analyze each neuron as a biological cell
        cells = []
        for neuron_idx in range(weights_in.shape[0]):
            neuron_weights_in = weights_in[neuron_idx, :]
            neuron_weights_out = weights_out[neuron_idx, :] if weights_out is not None else None
            neuron_bias = bias[neuron_idx]
            neuron_activation = layer_activations[neuron_idx]
            
            cell_char = self.get_neuron_characteristics(
                neuron_weights_in, neuron_weights_out, neuron_bias, neuron_activation
            )
            cells.append(cell_char)
        
        return cells, weights_in, weights_out
    
    def create_microscopy_frame(self, epoch_file, layer_idx=1, zoom_neurons=None):
        """Create a single microscopy frame showing cellular neuron view"""
        
        epoch = int(epoch_file.split('_')[-1].split('.')[0])
        print(f"🔬 Creating neural microscopy frame for epoch {epoch} - Layer {layer_idx + 1}")
        
        # Load model
        model = self.load_model_at_epoch(epoch_file)
        
        # Test inputs - different locations to see cellular response
        test_inputs = [
            torch.tensor([[0.3, 0.3]], dtype=DTYPE, device=device),  # Interior
            torch.tensor([[0.0, 0.5]], dtype=DTYPE, device=device),  # Boundary
            torch.tensor([[0.8, 0.8]], dtype=DTYPE, device=device)   # Corner
        ]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.patch.set_facecolor('black')
        
        # Analyze layer for different inputs
        for input_idx, test_input in enumerate(test_inputs):
            cells, weights_in, weights_out = self.create_cellular_layer_view(model, layer_idx, test_input)
            
            if cells is None:
                continue
            
            # === Cellular Colony View (Top row) ===
            ax = axes[0, input_idx]
            
            # Arrange neurons in a tissue-like pattern
            n_neurons = len(cells)
            cols = int(np.ceil(np.sqrt(n_neurons)))
            rows = int(np.ceil(n_neurons / cols))
            
            # Create cellular positions
            positions = []
            for i in range(n_neurons):
                row = i // cols
                col = i % cols
                x = col + 0.3 * np.random.normal()  # Add slight randomness for organic look
                y = row + 0.3 * np.random.normal()
                positions.append((x, y))
            
            # Draw each neuron as a biological cell
            for i, (cell, pos) in enumerate(zip(cells, positions)):
                x, y = pos
                
                # Cell body size based on connection strength
                cell_radius = 0.1 + 0.3 * min(1.0, cell['size'] / 10)
                
                # Cell color based on activation (like fluorescent staining)
                activity_color = plt.cm.viridis(min(1.0, cell['activity'] * 2))
                
                # Cell membrane
                cell_circle = Circle((x, y), cell_radius, 
                                   facecolor=activity_color, 
                                   edgecolor='white', 
                                   linewidth=1.5,
                                   alpha=0.8)
                ax.add_patch(cell_circle)
                
                # Nucleus (bias)
                nucleus_size = cell_radius * 0.4
                nucleus_color = 'red' if cell['bias'] > 0 else 'blue'
                nucleus_alpha = min(1.0, abs(cell['bias']) * 5)
                
                nucleus = Circle((x, y), nucleus_size,
                               facecolor=nucleus_color,
                               alpha=nucleus_alpha,
                               edgecolor='white',
                               linewidth=0.5)
                ax.add_patch(nucleus)
                
                # Cell extensions (dendrites/axons) based on strongest connections
                if zoom_neurons is None or i in zoom_neurons:
                    # Show strongest input connections as dendrites
                    strong_inputs = np.argsort(np.abs(cell['input_weights']))[-3:]
                    for input_idx in strong_inputs:
                        weight_strength = abs(cell['input_weights'][input_idx])
                        if weight_strength > 0.1:  # Only show significant connections
                            
                            # Dendrite direction (random but consistent)
                            np.random.seed(i * 100 + input_idx)  # Consistent randomness
                            angle = np.random.uniform(0, 2*np.pi)
                            dendrite_length = cell_radius + 0.2 * weight_strength
                            
                            end_x = x + dendrite_length * np.cos(angle)
                            end_y = y + dendrite_length * np.sin(angle)
                            
                            # Dendrite color
                            dendrite_color = 'lime' if cell['input_weights'][input_idx] > 0 else 'red'
                            alpha = min(1.0, weight_strength * 3)
                            
                            ax.plot([x, end_x], [y, end_y], 
                                   color=dendrite_color, 
                                   alpha=alpha, 
                                   linewidth=1 + 2*weight_strength)
                            
                            # Dendrite tip
                            ax.scatter(end_x, end_y, s=20*weight_strength, 
                                     color=dendrite_color, alpha=alpha)
                
                # Cell ID label
                ax.text(x, y, str(i), ha='center', va='center', 
                       fontsize=6, color='white', fontweight='bold')
            
            ax.set_facecolor('black')
            ax.set_xlim(-1, cols)
            ax.set_ylim(-1, rows)
            ax.set_aspect('equal')
            ax.axis('off')
            
            input_str = f"[{test_input[0,0]:.1f}, {test_input[0,1]:.1f}]"
            ax.set_title(f'Neural Tissue - Input {input_str}\nLayer {layer_idx + 1} Cellular Colony', 
                        color='white', fontweight='bold', fontsize=11)
            
            # === Detailed Microscopy View (Bottom row) ===
            ax = axes[1, input_idx]
            
            # Focus on a few specific neurons for detailed view
            focus_neurons = zoom_neurons if zoom_neurons else [0, 5, 10, 15]  # Default focus neurons
            focus_neurons = [n for n in focus_neurons if n < len(cells)]
            
            # Limit focus neurons to available cells
            max_neurons_to_show = min(4, len(cells))
            focus_neurons = focus_neurons[:max_neurons_to_show]
            
            # Create detailed cellular structures
            spacing = 3
            for plot_idx, neuron_idx in enumerate(focus_neurons[:4]):  # Max 4 detailed neurons
                if plot_idx >= 4:
                    break
                    
                cell = cells[neuron_idx]
                base_x = plot_idx * spacing
                base_y = 0
                
                # === Main Cell Body ===
                
                # Cell size based on total strength
                cell_radius = 0.3 + 0.5 * min(1.0, cell['size'] / 5)
                
                # Cell membrane with health-based transparency
                membrane_color = plt.cm.viridis(cell['activity'])
                
                # Main cell body
                cell_body = Circle((base_x, base_y), cell_radius,
                                 facecolor=membrane_color,
                                 edgecolor='white',
                                 linewidth=2,
                                 alpha=0.7 + 0.3 * cell['health'])
                ax.add_patch(cell_body)
                
                # === Organelles (Internal Structures) ===
                
                # Nucleus (bias) with size proportional to bias strength
                nucleus_radius = cell_radius * (0.2 + 0.3 * min(1.0, abs(cell['bias'])))
                nucleus_color = 'gold' if cell['bias'] > 0 else 'purple'
                
                nucleus = Circle((base_x, base_y), nucleus_radius,
                               facecolor=nucleus_color,
                               alpha=0.8,
                               edgecolor='white',
                               linewidth=1)
                ax.add_patch(nucleus)
                
                # Mitochondria (activation strength indicators)
                n_mitochondria = min(8, int(cell['activity'] * 10) + 1)
                for mito_idx in range(n_mitochondria):
                    angle = 2 * np.pi * mito_idx / n_mitochondria
                    mito_distance = cell_radius * 0.6
                    mito_x = base_x + mito_distance * np.cos(angle)
                    mito_y = base_y + mito_distance * np.sin(angle)
                    
                    mito_size = 0.05 + 0.1 * cell['activity']
                    mitochondria = Circle((mito_x, mito_y), mito_size,
                                        facecolor='orange',
                                        alpha=0.6,
                                        edgecolor='yellow',
                                        linewidth=0.5)
                    ax.add_patch(mitochondria)
                
                # === Synaptic Connections ===
                
                # Input dendrites (incoming connections)
                strong_inputs = np.argsort(np.abs(cell['input_weights']))[-5:]  # Top 5 inputs
                
                for dendrite_idx, input_idx in enumerate(strong_inputs):
                    weight_val = cell['input_weights'][input_idx]
                    weight_strength = abs(weight_val)
                    
                    if weight_strength > 0.05:  # Only show significant connections
                        # Dendrite angle
                        angle = 2 * np.pi * dendrite_idx / len(strong_inputs) + np.pi
                        dendrite_length = cell_radius + 0.3 + 0.4 * weight_strength
                        
                        end_x = base_x + dendrite_length * np.cos(angle)
                        end_y = base_y + dendrite_length * np.sin(angle)
                        
                        # Dendrite properties
                        dendrite_color = 'lime' if weight_val > 0 else 'red'
                        dendrite_width = 1 + 4 * weight_strength
                        dendrite_alpha = 0.6 + 0.4 * weight_strength
                        
                        # Draw dendrite with branches
                        ax.plot([base_x + cell_radius * np.cos(angle), end_x], 
                               [base_y + cell_radius * np.sin(angle), end_y],
                               color=dendrite_color,
                               linewidth=dendrite_width,
                               alpha=dendrite_alpha)
                        
                        # Synaptic terminal
                        terminal_size = 20 + 50 * weight_strength
                        ax.scatter(end_x, end_y, s=terminal_size,
                                 color=dendrite_color, alpha=dendrite_alpha,
                                 marker='o', edgecolors='white', linewidth=1)
                        
                        # Add small branches for realism
                        for branch in range(2):
                            branch_angle = angle + 0.5 * (branch - 0.5)
                            branch_length = 0.15 * weight_strength
                            branch_x = end_x + branch_length * np.cos(branch_angle)
                            branch_y = end_y + branch_length * np.sin(branch_angle)
                            
                            ax.plot([end_x, branch_x], [end_y, branch_y],
                                   color=dendrite_color, alpha=dendrite_alpha * 0.7,
                                   linewidth=dendrite_width * 0.5)
                
                # Output axons (outgoing connections)
                if cell['connections'].size > 0:
                    strong_outputs = np.argsort(cell['connections'])[-3:]  # Top 3 outputs
                    
                    for axon_idx, output_idx in enumerate(strong_outputs):
                        weight_strength = cell['connections'][output_idx]
                        
                        if weight_strength > 0.05:
                            # Axon angle
                            angle = 2 * np.pi * axon_idx / len(strong_outputs)
                            axon_length = cell_radius + 0.5 + 0.6 * weight_strength
                            
                            end_x = base_x + axon_length * np.cos(angle)
                            end_y = base_y + axon_length * np.sin(angle)
                            
                            # Axon properties
                            axon_color = 'cyan'
                            axon_width = 2 + 3 * weight_strength
                            axon_alpha = 0.7 + 0.3 * weight_strength
                            
                            # Draw axon
                            ax.plot([base_x + cell_radius * np.cos(angle), end_x], 
                                   [base_y + cell_radius * np.sin(angle), end_y],
                                   color=axon_color,
                                   linewidth=axon_width,
                                   alpha=axon_alpha)
                            
                            # Axon terminal
                            ax.scatter(end_x, end_y, s=30 + 70 * weight_strength,
                                     color=axon_color, alpha=axon_alpha,
                                     marker='s', edgecolors='white', linewidth=1)
                
                # Cell information label
                info_text = f"Cell {neuron_idx}\nSize: {cell['size']:.2f}\nActivity: {cell['activity']:.2f}\nHealth: {cell['health']:.2f}"
                ax.text(base_x, base_y - cell_radius - 0.8, info_text,
                       ha='center', va='top', color='white', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='darkblue', alpha=0.7))
            
            ax.set_facecolor('black')
            ax.set_xlim(-1, len(focus_neurons) * spacing)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.axis('off')
            
            ax.set_title(f'High-Res Cellular Microscopy\nSelected Neurons - Detailed Structure', 
                        color='white', fontweight='bold', fontsize=11)
        
        # === Overall Styling ===
        
        # Biological development phase
        if epoch < 100:
            bio_phase = "🧬 Cell Division"
            bio_desc = "Rapid cellular growth"
        elif epoch < 500:
            bio_phase = "🌱 Differentiation"
            bio_desc = "Cells specializing"
        elif epoch < 1000:
            bio_phase = "🧠 Tissue Formation"
            bio_desc = "Neural circuits forming"
        else:
            bio_phase = "🏛️ Mature Tissue"
            bio_desc = "Stable neural architecture"
        
        fig.suptitle(f'Neural Microscopy Time-Lapse - Layer {layer_idx + 1}\n'
                    f'{bio_phase} - Epoch {epoch}\n{bio_desc}', 
                    color='white', fontsize=16, fontweight='bold')
        
        return fig
    
    def create_growth_time_lapse(self, focus_layer=1, focus_neurons=[0, 5, 10, 15]):
        """Create time-lapse animation of neural growth"""
        print(f"🎬 Creating neural growth time-lapse for layer {focus_layer + 1}...")
        
        # Find all model snapshots
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found")
            return
        
        os.makedirs('microscopy_frames', exist_ok=True)
        
        # Subsample for smooth animation
        step = max(1, len(model_files) // 25)  # Max 25 frames
        selected_files = model_files[::step]
        
        print(f"🔬 Processing {len(selected_files)} epochs for microscopy...")
        
        for frame_idx, model_file in enumerate(selected_files):
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            try:
                fig = self.create_microscopy_frame(model_file, focus_layer, focus_neurons)
                
                plt.savefig(f'microscopy_frames/microscopy_layer_{focus_layer}_epoch_{epoch:04d}.png', 
                           dpi=100, bbox_inches='tight', facecolor='black')
                plt.close(fig)
                
            except Exception as e:
                print(f"⚠️ Error processing epoch {epoch}: {e}")
                continue
        
        # Create GIF
        self.create_microscopy_gif(focus_layer)
    
    def create_microscopy_gif(self, layer_idx):
        """Create GIF from microscopy frames"""
        print("🎬 Creating neural microscopy GIF...")
        
        frame_files = sorted(glob.glob(f'microscopy_frames/microscopy_layer_{layer_idx}_epoch_*.png'))
        
        if frame_files:
            images = [Image.open(f) for f in frame_files]
            
            os.makedirs('animations', exist_ok=True)
            
            # Regular speed
            images[0].save(f'animations/neural_microscopy_layer_{layer_idx}.gif', 
                          save_all=True, append_images=images[1:], 
                          duration=250, loop=0)
            print(f"✅ Neural microscopy GIF created for layer {layer_idx}")
            
            # Fast version
            fast_images = images[::2]
            if len(fast_images) > 1:
                fast_images[0].save(f'animations/neural_microscopy_layer_{layer_idx}_fast.gif', 
                                  save_all=True, append_images=fast_images[1:], 
                                  duration=150, loop=0)
                print(f"✅ Fast neural microscopy GIF created for layer {layer_idx}")

def main():
    """Main function"""
    print("🔬 Neural Microscopy Time-Lapse Visualizer")
    print("=" * 50)
    
    microscope = NeuralMicroscopyVisualizer()
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    print("\n🔬 Creating neural microscopy time-lapse...")
    
    # Create microscopy for multiple layers
    layers_to_study = [0, 3, 6]  # First, middle, second-to-last hidden layers (avoid layer 7)
    
    for layer_idx in layers_to_study:
        print(f"\n🧬 Studying cellular development in layer {layer_idx + 1}...")
        # Use safer neuron indices
        safe_focus_neurons = [0, 5, 10, 15]  # Remove neuron 19 to avoid bounds issues
        microscope.create_growth_time_lapse(focus_layer=layer_idx, 
                                           focus_neurons=safe_focus_neurons)
    
    print("\n🎉 Neural microscopy visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/neural_microscopy_layer_*.gif - Cellular time-lapse")
    print("   📂 animations/neural_microscopy_layer_*_fast.gif - Faster versions")
    print("   📂 microscopy_frames/ - Individual microscopy frames")
    print("\n🔬 Watch your neurons grow like living cells under a microscope!")

if __name__ == "__main__":
    main()