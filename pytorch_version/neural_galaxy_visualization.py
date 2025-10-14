# Neural Galaxy Visualization for PINN Networks
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#%% Neural Network definition (matching pytorch_code_general.py)
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=4, neurons=50):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, neurons), nn.Tanh()]
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#%% Neural Galaxy Visualization Function
def visualize_neural_galaxy(model, input_sample, save_name="neural_galaxy"):
    """
    Visualize neural network as a galaxy where:
    - Each neuron is a star with brightness proportional to activation
    - Connections shown as gravitational fields (lines with opacity based on weight strength)
    """
    model.eval()
    with torch.no_grad():
        # Get activations for each layer
        activations = []
        weights = []
        x = input_sample
        
        # Extract weights and compute activations
        linear_layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
        
        for i, layer in enumerate(model.net):
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.detach().numpy())
                x = layer(x)
                activations.append(x.clone())
                if i < len(model.net) - 1 and isinstance(model.net[i+1], nn.Tanh):
                    x = torch.tanh(x)
            elif isinstance(layer, nn.Tanh):
                x = layer(x)
        
        # Create galaxy visualization
        n_layers = len(activations)
        fig, axes = plt.subplots(1, min(n_layers, 4), figsize=(5*min(n_layers, 4), 5))
        
        if n_layers == 1:
            axes = [axes]
        elif n_layers > 1 and n_layers <= 4:
            if n_layers < 4:
                axes = axes[:n_layers]
        
        for layer_idx, activation in enumerate(activations[:4]):
            ax = axes[layer_idx] if n_layers > 1 else axes[0]
            
            # Get layer activations
            layer_activations = activation.numpy().flatten()
            
            # Create neuron positions in a spiral galaxy pattern
            n_neurons = len(layer_activations)
            
            if n_neurons == 1:
                # Special case for output layer
                x_pos = np.array([0])
                y_pos = np.array([0])
            else:
                # Create spiral galaxy structure
                angles = np.linspace(0, 4*np.pi, n_neurons, endpoint=False)
                spiral_factor = 0.5
                radii = 0.5 + spiral_factor * angles / (4*np.pi)
                
                x_pos = radii * np.cos(angles)
                y_pos = radii * np.sin(angles)
            
            # Normalize activations for brightness (0-1)
            act_normalized = np.abs(layer_activations)
            if act_normalized.max() > 0:
                act_normalized = act_normalized / act_normalized.max()
            
            # Plot neurons as stars
            sizes = 100 + 400*act_normalized if n_neurons > 1 else [300]
            colors = act_normalized if n_neurons > 1 else [1.0]
            
            scatter = ax.scatter(x_pos, y_pos, 
                               s=sizes,                     # Size based on activation
                               c=colors,                    # Color based on activation
                               cmap='plasma',               # Galaxy-like colormap
                               alpha=0.8,
                               edgecolors='white',
                               linewidth=0.5)
            
            # Add connections as gravitational fields
            if layer_idx < len(weights) - 1:
                next_weights = weights[layer_idx + 1]
                
                # Create positions for next layer
                next_n_neurons = next_weights.shape[0]
                if next_n_neurons == 1:
                    next_x_pos = np.array([0])
                    next_y_pos = np.array([0])
                else:
                    next_angles = np.linspace(0, 4*np.pi, next_n_neurons, endpoint=False)
                    next_radii = 0.5 + spiral_factor * next_angles / (4*np.pi)
                    next_x_pos = next_radii * np.cos(next_angles)
                    next_y_pos = next_radii * np.sin(next_angles)
                
                # Sample connections to avoid clutter
                max_connections = min(30, n_neurons)
                connection_indices = np.random.choice(n_neurons, max_connections, replace=False) if n_neurons > max_connections else range(n_neurons)
                
                for i in connection_indices:
                    for j in range(min(8, next_n_neurons)):  # Limit outgoing connections
                        # Calculate connection strength
                        weight_strength = abs(next_weights[j, i])
                        max_weight = np.abs(next_weights).max()
                        if max_weight > 0:
                            weight_strength = weight_strength / max_weight
                        
                        # Draw connection as gravitational field
                        alpha = 0.05 + 0.3 * weight_strength
                        linewidth = 0.3 + 1.0 * weight_strength
                        
                        ax.plot([x_pos[i], next_x_pos[j]], [y_pos[i], next_y_pos[j]], 
                               'cyan', alpha=alpha, linewidth=linewidth)
            
            # Styling for galaxy appearance
            ax.set_facecolor('black')
            
            # Set limits based on data
            if n_neurons > 1:
                margin = 0.5
                ax.set_xlim(x_pos.min() - margin, x_pos.max() + margin)
                ax.set_ylim(y_pos.min() - margin, y_pos.max() + margin)
            else:
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
            
            ax.set_aspect('equal')
            ax.set_title(f'Layer {layer_idx+1} ({n_neurons} neurons)', color='white', fontsize=12)
            ax.axis('off')
            
            # Add colorbar for activation strength
            if n_neurons > 1:
                cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Activation Strength', color='white', fontsize=10)
                cbar.ax.yaxis.set_tick_params(color='white', labelsize=8)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Overall styling
        fig.patch.set_facecolor('black')
        plt.suptitle(f'Neural Galaxy: {save_name}', color='white', fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()

#%% Create and train simple networks for demonstration
print("Creating demonstration networks...")

# Create networks
ux_net = MLP(input_dim=2, output_dim=1, hidden_layers=3, neurons=20)
uy_net = MLP(input_dim=2, output_dim=1, hidden_layers=3, neurons=20)
sxx_net = MLP(input_dim=2, output_dim=1, hidden_layers=3, neurons=20)

# Simple training data
torch.manual_seed(42)
x_train = torch.rand(100, 2)
y_train = torch.sin(x_train[:, 0:1]) * torch.cos(x_train[:, 1:2])

# Quick training for demonstration
optimizer_ux = torch.optim.Adam(ux_net.parameters(), lr=1e-3)
optimizer_uy = torch.optim.Adam(uy_net.parameters(), lr=1e-3)
optimizer_sxx = torch.optim.Adam(sxx_net.parameters(), lr=1e-3)

print("Training networks briefly for demonstration...")
for epoch in range(100):
    # Train ux_net
    pred_ux = ux_net(x_train)
    loss_ux = ((pred_ux - y_train)**2).mean()
    optimizer_ux.zero_grad()
    loss_ux.backward()
    optimizer_ux.step()
    
    # Train uy_net with different target
    pred_uy = uy_net(x_train)
    loss_uy = ((pred_uy - y_train * 0.5)**2).mean()
    optimizer_uy.zero_grad()
    loss_uy.backward()
    optimizer_uy.step()
    
    # Train sxx_net with different target
    pred_sxx = sxx_net(x_train)
    loss_sxx = ((pred_sxx - y_train * 2)**2).mean()
    optimizer_sxx.zero_grad()
    loss_sxx.backward()
    optimizer_sxx.step()

print("Training complete. Creating Neural Galaxy visualizations...")

# Create sample inputs for galaxy visualization
sample_inputs = [
    torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    torch.tensor([[0.2, 0.8]], dtype=torch.float32),
    torch.tensor([[0.9, 0.1]], dtype=torch.float32)
]

# Visualize each network as a galaxy with different inputs
for i, sample_input in enumerate(sample_inputs):
    print(f"Creating galaxies for input sample {i+1}: {sample_input.numpy().flatten()}")
    
    visualize_neural_galaxy(ux_net, sample_input, save_name=f"neural_galaxy_ux_sample{i+1}")
    visualize_neural_galaxy(uy_net, sample_input, save_name=f"neural_galaxy_uy_sample{i+1}")
    visualize_neural_galaxy(sxx_net, sample_input, save_name=f"neural_galaxy_sxx_sample{i+1}")

#%% Create animated galaxy evolution (showing how activations change)
def create_galaxy_evolution(model, input_samples, save_name="galaxy_evolution"):
    """Create multiple galaxy snapshots showing how neural activations change with different inputs"""
    
    fig, axes = plt.subplots(2, len(input_samples), figsize=(5*len(input_samples), 10))
    if len(input_samples) == 1:
        axes = axes.reshape(-1, 1)
    
    for sample_idx, sample_input in enumerate(input_samples):
        model.eval()
        with torch.no_grad():
            # Get activations
            activations = []
            x = sample_input
            
            for layer in model.net:
                if isinstance(layer, nn.Linear):
                    x = layer(x)
                    activations.append(x.clone())
                    if hasattr(model.net, '__getitem__'):
                        next_idx = list(model.net).index(layer) + 1
                        if next_idx < len(model.net) and isinstance(model.net[next_idx], nn.Tanh):
                            x = torch.tanh(x)
                elif isinstance(layer, nn.Tanh):
                    x = layer(x)
            
            # Plot first and last hidden layers
            for row_idx, layer_idx in enumerate([0, -2]):  # First hidden and last hidden layer
                if layer_idx < len(activations):
                    ax = axes[row_idx, sample_idx]
                    
                    layer_activations = activations[layer_idx].numpy().flatten()
                    n_neurons = len(layer_activations)
                    
                    # Create positions
                    angles = np.linspace(0, 4*np.pi, n_neurons, endpoint=False)
                    radii = 0.5 + 0.5 * angles / (4*np.pi)
                    x_pos = radii * np.cos(angles)
                    y_pos = radii * np.sin(angles)
                    
                    # Normalize activations
                    act_normalized = np.abs(layer_activations)
                    if act_normalized.max() > 0:
                        act_normalized = act_normalized / act_normalized.max()
                    
                    # Plot
                    scatter = ax.scatter(x_pos, y_pos, 
                                       s=100 + 300*act_normalized,
                                       c=act_normalized,
                                       cmap='plasma',
                                       alpha=0.8,
                                       edgecolors='white',
                                       linewidth=0.5)
                    
                    ax.set_facecolor('black')
                    ax.set_xlim(-2, 2)
                    ax.set_ylim(-2, 2)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    
                    # Title
                    layer_name = "First Hidden" if row_idx == 0 else "Last Hidden"
                    input_str = f"[{sample_input[0,0]:.1f}, {sample_input[0,1]:.1f}]"
                    ax.set_title(f'{layer_name} Layer\nInput: {input_str}', 
                               color='white', fontsize=10)
    
    fig.patch.set_facecolor('black')
    plt.suptitle('Neural Galaxy Evolution Across Inputs', color='white', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()

# Create evolution visualization
print("Creating galaxy evolution visualization...")
create_galaxy_evolution(ux_net, sample_inputs, save_name="neural_galaxy_evolution")

print("\nNeural Galaxy visualizations completed!")
print("Generated files:")
print("- neural_galaxy_ux_sample*.png")
print("- neural_galaxy_uy_sample*.png") 
print("- neural_galaxy_sxx_sample*.png")
print("- neural_galaxy_evolution.png")