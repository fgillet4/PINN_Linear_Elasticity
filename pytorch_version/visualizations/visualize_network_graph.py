#!/usr/bin/env python3
"""
Neural Network Graph Visualizer for PyTorch PINN Models

This script loads a PyTorch .pt model file and creates a visual representation
of the network architecture as a directed graph, including weights and connections.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from collections import defaultdict


class NetworkVisualizer:
    def __init__(self, model_path):
        """Initialize the visualizer with a model path"""
        self.model_path = model_path
        self.model_dict = None
        self.graph = nx.DiGraph()
        self.layer_info = {}
        self.pos = {}
        
    def load_model(self):
        """Load the PyTorch model state dict"""
        try:
            self.model_dict = torch.load(self.model_path, map_location='cpu')
            print(f"✅ Successfully loaded model from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def analyze_architecture(self):
        """Analyze the model architecture from state dict"""
        if not self.model_dict:
            print("❌ No model loaded")
            return
        
        print("\n🔍 Analyzing Network Architecture:")
        print("=" * 50)
        
        # Group parameters by layer
        layers = defaultdict(dict)
        for param_name, param_tensor in self.model_dict.items():
            if '.' in param_name:
                layer_name, param_type = param_name.rsplit('.', 1)
                layers[layer_name][param_type] = param_tensor
            else:
                layers[param_name]['full'] = param_tensor
        
        # Analyze each layer
        layer_order = []
        for layer_name in sorted(layers.keys()):
            layer_params = layers[layer_name]
            
            if 'weight' in layer_params:
                weight_shape = layer_params['weight'].shape
                bias_shape = layer_params.get('bias', torch.tensor([])).shape
                
                # Extract layer information
                if len(weight_shape) == 2:  # Fully connected layer
                    input_size, output_size = weight_shape[1], weight_shape[0]
                    layer_type = 'Linear'
                else:
                    input_size, output_size = "Unknown", "Unknown"
                    layer_type = 'Unknown'
                
                self.layer_info[layer_name] = {
                    'type': layer_type,
                    'input_size': input_size,
                    'output_size': output_size,
                    'weight_shape': weight_shape,
                    'bias_shape': bias_shape,
                    'weight_tensor': layer_params['weight'],
                    'bias_tensor': layer_params.get('bias', None)
                }
                
                layer_order.append(layer_name)
                
                print(f"📋 {layer_name}:")
                print(f"   Type: {layer_type}")
                print(f"   Input → Output: {input_size} → {output_size}")
                print(f"   Weight shape: {weight_shape}")
                print(f"   Bias shape: {bias_shape}")
                print(f"   Parameters: {layer_params['weight'].numel() + (layer_params.get('bias', torch.tensor([])).numel())}")
                
        return layer_order
    
    def build_graph(self, show_weights=True, weight_threshold=0.1):
        """Build NetworkX graph representation"""
        if not self.layer_info:
            print("❌ No layer information available")
            return
        
        print(f"\n🏗️  Building Network Graph (weight_threshold={weight_threshold})...")
        
        self.graph.clear()
        node_id = 0
        layer_nodes = {}
        
        # Sort layers by their natural order
        def layer_sort_key(layer_name):
            if 'hidden_layers' in layer_name:
                # Extract number from 'hidden_layers.X'
                return (0, int(layer_name.split('.')[1]))
            elif 'output' in layer_name:
                # Output layers come last
                return (1, layer_name)
            else:
                # Other layers come first
                return (-1, layer_name)
        
        sorted_layers = sorted(self.layer_info.keys(), key=layer_sort_key)
        
        # Create input layer nodes
        first_layer = sorted_layers[0] if sorted_layers else None
        if first_layer:
            input_size = self.layer_info[first_layer]['input_size']
            input_nodes = []
            for i in range(min(input_size, 10)):  # Limit visualization nodes
                self.graph.add_node(node_id, layer='input', neuron=i, layer_name='Input')
                input_nodes.append(node_id)
                node_id += 1
            layer_nodes['input'] = input_nodes
        
        # Create hidden and output layer nodes
        for layer_name in sorted_layers:
            layer_data = self.layer_info[layer_name]
            output_size = layer_data['output_size']
            
            # Determine layer type for visualization
            if 'hidden' in layer_name:
                layer_type = 'hidden'
            elif 'output' in layer_name:
                layer_type = 'output'
            else:
                layer_type = 'other'
            
            # Create nodes for this layer
            layer_node_list = []
            max_nodes = min(output_size, 15)  # Limit nodes for visualization
            for i in range(max_nodes):
                self.graph.add_node(node_id, layer=layer_type, neuron=i, layer_name=layer_name)
                layer_node_list.append(node_id)
                node_id += 1
            
            layer_nodes[layer_name] = layer_node_list
        
        # Add edges with weights
        prev_layer_nodes = layer_nodes['input']
        prev_layer_name = 'input'
        
        for layer_name in sorted_layers:
            current_nodes = layer_nodes[layer_name]
            weight_tensor = self.layer_info[layer_name]['weight_tensor']
            
            # Add edges between previous layer and current layer
            for i, curr_node in enumerate(current_nodes):
                for j, prev_node in enumerate(prev_layer_nodes):
                    if j < weight_tensor.shape[1] and i < weight_tensor.shape[0]:
                        weight_val = float(weight_tensor[i, j])
                        
                        # Only show significant weights
                        if show_weights and abs(weight_val) > weight_threshold:
                            self.graph.add_edge(prev_node, curr_node, 
                                              weight=weight_val,
                                              abs_weight=abs(weight_val))
                        elif not show_weights:
                            self.graph.add_edge(prev_node, curr_node, weight=1.0)
            
            prev_layer_nodes = current_nodes
            prev_layer_name = layer_name
        
        print(f"✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def calculate_layout(self):
        """Calculate node positions for left-to-right visualization"""
        if not self.graph.nodes():
            return
        
        self.pos = {}
        
        # Get all unique layer names in order
        layer_order = []
        layer_nodes = defaultdict(list)
        
        # Group nodes by their actual layer
        for node, data in self.graph.nodes(data=True):
            layer_name = data['layer_name']
            layer_nodes[layer_name].append(node)
            if layer_name not in layer_order:
                layer_order.append(layer_name)
        
        # Sort layers properly
        def layer_sort_key(layer_name):
            if layer_name == 'Input':
                return (-1, 0)
            elif 'hidden_layers' in layer_name:
                return (0, int(layer_name.split('.')[1]))
            elif 'output' in layer_name:
                return (1, layer_name)
            else:
                return (2, layer_name)
        
        layer_order = sorted(layer_order, key=layer_sort_key)
        
        # Calculate positions - left to right layout
        x_spacing = 3.0  # Horizontal spacing between layers
        
        for layer_idx, layer_name in enumerate(layer_order):
            nodes = layer_nodes[layer_name]
            x = layer_idx * x_spacing
            
            # Vertical spacing for nodes in the same layer
            if len(nodes) == 1:
                y_positions = [0]
            else:
                y_spacing = 2.0
                y_start = -(len(nodes) - 1) * y_spacing / 2
                y_positions = [y_start + i * y_spacing for i in range(len(nodes))]
            
            for node, y in zip(nodes, y_positions):
                self.pos[node] = (x, y)
    
    def visualize_network_simple(self, figsize=(16, 8), save_path='network_graph_simple.png'):
        """Create a clean left-to-right network visualization with circles and lines only"""
        print(f"\n🎨 Creating simple left-to-right visualization...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define layer structure from model
        layers = [
            {'name': 'Input', 'size': 2, 'color': '#4CAF50'},
            {'name': 'Hidden 1', 'size': 20, 'color': '#2196F3'}, 
            {'name': 'Hidden 2', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 3', 'size': 20, 'color': '#2196F3'}, 
            {'name': 'Hidden 4', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 5', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 6', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 7', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 8', 'size': 20, 'color': '#2196F3'},
            {'name': 'Output', 'size': 5, 'color': '#FF9800'}
        ]
        
        # Calculate positions
        x_spacing = 2.0
        node_positions = []
        
        for layer_idx, layer in enumerate(layers):
            x = layer_idx * x_spacing
            size = layer['size']
            
            if size == 1:
                y_positions = [0]
            else:
                y_spacing = 0.8
                y_start = -(size - 1) * y_spacing / 2
                y_positions = [y_start + i * y_spacing for i in range(size)]
            
            layer_positions = []
            for y in y_positions:
                layer_positions.append((x, y))
            
            node_positions.append({
                'positions': layer_positions,
                'color': layer['color'],
                'name': layer['name']
            })
        
        # Draw connections (lines) first so they appear behind nodes
        for i in range(len(node_positions) - 1):
            current_layer = node_positions[i]
            next_layer = node_positions[i + 1]
            
            for curr_pos in current_layer['positions']:
                for next_pos in next_layer['positions']:
                    ax.plot([curr_pos[0], next_pos[0]], [curr_pos[1], next_pos[1]], 
                           'gray', alpha=0.3, linewidth=0.5)
        
        # Draw nodes (circles)
        for layer_data in node_positions:
            positions = layer_data['positions']
            color = layer_data['color']
            
            for pos in positions:
                circle = plt.Circle(pos, 0.15, color=color, alpha=0.8, zorder=3)
                ax.add_patch(circle)
        
        # Add layer labels
        for i, layer_data in enumerate(node_positions):
            positions = layer_data['positions']
            if positions:
                x = positions[0][0]
                max_y = max([pos[1] for pos in positions])
                min_y = min([pos[1] for pos in positions])
                
                # Position label above the layer
                label_y = max_y + 0.5
                ax.text(x, label_y, layer_data['name'], ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
                
                # Add individual output node labels
                if layer_data['name'] == 'Output':
                    output_labels = ['Ux', 'Uy', 'Sxx', 'Sxy', 'Syy']  # Based on alphabetical order of output layers
                    for idx, (pos, label) in enumerate(zip(positions, output_labels)):
                        if idx < len(output_labels):
                            ax.text(pos[0] + 0.25, pos[1], label, ha='left', va='center',
                                   fontsize=8, fontweight='bold', color='darkred')
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, (len(layers) - 1) * x_spacing + 0.5)
        
        # Calculate y limits based on all positions
        all_y = []
        for layer_data in node_positions:
            all_y.extend([pos[1] for pos in layer_data['positions']])
        
        if all_y:
            y_margin = 1.0
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        # Remove axis ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add title with network info
        total_params = sum(info['weight_tensor'].numel() + 
                         (info['bias_tensor'].numel() if info['bias_tensor'] is not None else 0)
                         for info in self.layer_info.values())
        
        title_text = f'PINN Network Architecture\nPhysics-Informed Neural Network for Linear Elasticity\n8 Hidden Layers × 20 Neurons | {total_params:,} Parameters | Outputs: Ux, Uy, Sxx, Syy, Sxy'
        ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)
        
        # Save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Simple network visualization saved as '{save_path}'")
        
        return fig, ax
    
    def visualize_network_with_weights(self, figsize=(18, 8), weight_threshold=0.05, save_path='pinn_network_with_weights.png'):
        """Create network visualization with weight-colored edges"""
        print(f"\n🎨 Creating network visualization with weight-colored edges...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define layer structure from model
        layers = [
            {'name': 'Input', 'size': 2, 'color': '#4CAF50'},
            {'name': 'Hidden 1', 'size': 20, 'color': '#2196F3'}, 
            {'name': 'Hidden 2', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 3', 'size': 20, 'color': '#2196F3'}, 
            {'name': 'Hidden 4', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 5', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 6', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 7', 'size': 20, 'color': '#2196F3'},
            {'name': 'Hidden 8', 'size': 20, 'color': '#2196F3'},
            {'name': 'Output', 'size': 5, 'color': '#FF9800'}
        ]
        
        # Calculate positions
        x_spacing = 2.0
        node_positions = []
        
        for layer_idx, layer in enumerate(layers):
            x = layer_idx * x_spacing
            size = layer['size']
            
            if size == 1:
                y_positions = [0]
            else:
                y_spacing = 0.8
                y_start = -(size - 1) * y_spacing / 2
                y_positions = [y_start + i * y_spacing for i in range(size)]
            
            layer_positions = []
            for y in y_positions:
                layer_positions.append((x, y))
            
            node_positions.append({
                'positions': layer_positions,
                'color': layer['color'],
                'name': layer['name']
            })
        
        # Get layer names and organize them properly
        hidden_layers = [name for name in self.layer_info.keys() if 'hidden_layers' in name]
        hidden_layers_sorted = sorted(hidden_layers, key=lambda x: int(x.split('.')[1]))
        output_layers = [name for name in self.layer_info.keys() if 'output' in name]
        output_layers_sorted = sorted(output_layers)  # Alphabetical order
        
        # Draw connections with weight-based colors
        for i in range(len(node_positions) - 1):
            current_layer_pos = node_positions[i]
            next_layer_pos = node_positions[i + 1]
            
            # Determine which layer weights to use
            if i == 0:
                # Input to first hidden layer
                layer_name = hidden_layers_sorted[0]
                weight_tensor = self.layer_info[layer_name]['weight_tensor']
                weights = weight_tensor.detach().numpy()
                
                # Normalize weights for color mapping
                max_abs_weight = np.abs(weights).max()
                
                for next_idx, next_pos in enumerate(next_layer_pos['positions']):
                    for curr_idx, curr_pos in enumerate(current_layer_pos['positions']):
                        if (next_idx < weights.shape[0] and curr_idx < weights.shape[1]):
                            weight_val = weights[next_idx, curr_idx]
                            abs_weight = abs(weight_val)
                            
                            # Only draw significant weights
                            if abs_weight > weight_threshold:
                                # Color based on weight value: red for negative, blue for positive
                                if weight_val > 0:
                                    color = plt.cm.Blues(abs_weight / max_abs_weight)
                                else:
                                    color = plt.cm.Reds(abs_weight / max_abs_weight)
                                
                                # Line width based on absolute weight
                                line_width = (abs_weight / max_abs_weight) * 2 + 0.2
                                
                                ax.plot([curr_pos[0], next_pos[0]], [curr_pos[1], next_pos[1]], 
                                       color=color, alpha=0.7, linewidth=line_width)
                                       
            elif i < len(hidden_layers_sorted):
                # Hidden layer to hidden layer
                layer_name = hidden_layers_sorted[i]
                weight_tensor = self.layer_info[layer_name]['weight_tensor']
                weights = weight_tensor.detach().numpy()
                
                # Normalize weights for color mapping
                max_abs_weight = np.abs(weights).max()
                
                for next_idx, next_pos in enumerate(next_layer_pos['positions']):
                    for curr_idx, curr_pos in enumerate(current_layer_pos['positions']):
                        if (next_idx < weights.shape[0] and curr_idx < weights.shape[1]):
                            weight_val = weights[next_idx, curr_idx]
                            abs_weight = abs(weight_val)
                            
                            # Only draw significant weights
                            if abs_weight > weight_threshold:
                                # Color based on weight value: red for negative, blue for positive
                                if weight_val > 0:
                                    color = plt.cm.Blues(abs_weight / max_abs_weight)
                                else:
                                    color = plt.cm.Reds(abs_weight / max_abs_weight)
                                
                                # Line width based on absolute weight
                                line_width = (abs_weight / max_abs_weight) * 2 + 0.2
                                
                                ax.plot([curr_pos[0], next_pos[0]], [curr_pos[1], next_pos[1]], 
                                       color=color, alpha=0.7, linewidth=line_width)
                                       
            elif i == len(hidden_layers_sorted):
                # Last hidden layer to output layers - special handling
                # Each output node connects to all neurons in the last hidden layer
                for output_idx, output_layer_name in enumerate(output_layers_sorted):
                    if output_idx < len(next_layer_pos['positions']):
                        output_pos = next_layer_pos['positions'][output_idx]
                        weight_tensor = self.layer_info[output_layer_name]['weight_tensor']
                        weights = weight_tensor.detach().numpy()
                        
                        # Normalize weights for color mapping
                        max_abs_weight = np.abs(weights).max()
                        
                        # Connect each neuron in last hidden layer to this output
                        for curr_idx, curr_pos in enumerate(current_layer_pos['positions']):
                            if curr_idx < weights.shape[1]:  # weights shape is [1, 20] for output layers
                                weight_val = weights[0, curr_idx]  # Output layers have shape [1, 20]
                                abs_weight = abs(weight_val)
                                
                                # Only draw significant weights
                                if abs_weight > weight_threshold:
                                    # Color based on weight value: red for negative, blue for positive
                                    if weight_val > 0:
                                        color = plt.cm.Blues(abs_weight / max_abs_weight)
                                    else:
                                        color = plt.cm.Reds(abs_weight / max_abs_weight)
                                    
                                    # Line width based on absolute weight
                                    line_width = (abs_weight / max_abs_weight) * 2 + 0.2
                                    
                                    ax.plot([curr_pos[0], output_pos[0]], [curr_pos[1], output_pos[1]], 
                                           color=color, alpha=0.7, linewidth=line_width)
        
        # Draw nodes (circles)
        for layer_data in node_positions:
            positions = layer_data['positions']
            color = layer_data['color']
            
            for pos in positions:
                circle = plt.Circle(pos, 0.15, color=color, alpha=0.8, zorder=3)
                ax.add_patch(circle)
        
        # Add layer labels
        for i, layer_data in enumerate(node_positions):
            positions = layer_data['positions']
            if positions:
                x = positions[0][0]
                max_y = max([pos[1] for pos in positions])
                
                label_y = max_y + 0.5
                ax.text(x, label_y, layer_data['name'], ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
                
                # Add individual output node labels
                if layer_data['name'] == 'Output':
                    output_labels = ['Ux', 'Uy', 'Sxx', 'Sxy', 'Syy']  # Based on alphabetical order of output layers
                    for idx, (pos, label) in enumerate(zip(positions, output_labels)):
                        if idx < len(output_labels):
                            ax.text(pos[0] + 0.25, pos[1], label, ha='left', va='center',
                                   fontsize=8, fontweight='bold', color='darkred')
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, (len(layers) - 1) * x_spacing + 0.5)
        
        # Calculate y limits
        all_y = []
        for layer_data in node_positions:
            all_y.extend([pos[1] for pos in layer_data['positions']])
        
        if all_y:
            y_margin = 1.0
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        # Remove axis ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add title with network info
        total_params = sum(info['weight_tensor'].numel() + 
                         (info['bias_tensor'].numel() if info['bias_tensor'] is not None else 0)
                         for info in self.layer_info.values())
        
        title_text = f'PINN Network Architecture with Weight Visualization\nPhysics-Informed Neural Network for Linear Elasticity\n8 Hidden Layers × 20 Neurons | {total_params:,} Parameters | Blue=Positive, Red=Negative'
        ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)
        
        # Add weight legend text
        legend_text = f"Edge Colors:\nBlue = Positive weights\nRed = Negative weights\nThickness = |weight|\nThreshold: {weight_threshold}"
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
        
        # Save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Weight-colored network visualization saved as '{save_path}'")
        
        return fig, ax
    
    def create_weight_heatmap(self, layer_name=None, save_path='weight_heatmap.png'):
        """Create heatmap visualization of weights for a specific layer"""
        if not self.layer_info:
            print("❌ No layer information available")
            return
        
        # Select layer to visualize
        if layer_name is None:
            # Default to first hidden layer
            layer_name = next((name for name in self.layer_info.keys() if 'hidden' in name), 
                            list(self.layer_info.keys())[0])
        
        if layer_name not in self.layer_info:
            print(f"❌ Layer '{layer_name}' not found")
            return
        
        weight_tensor = self.layer_info[layer_name]['weight_tensor']
        weight_matrix = weight_tensor.detach().numpy()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(weight_matrix, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Weight Matrix Heatmap: {layer_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Input Neurons')
        ax.set_ylabel('Output Neurons')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight Value', rotation=270, labelpad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Weight heatmap saved as '{save_path}'")
        
        return fig, ax
    
    def print_summary(self):
        """Print a comprehensive summary of the network"""
        if not self.layer_info:
            print("❌ No layer information available")
            return
        
        print("\n📊 NETWORK SUMMARY")
        print("=" * 60)
        
        total_params = 0
        total_weights = 0
        total_biases = 0
        
        for layer_name, info in self.layer_info.items():
            weight_params = info['weight_tensor'].numel()
            bias_params = info['bias_tensor'].numel() if info['bias_tensor'] is not None else 0
            layer_params = weight_params + bias_params
            
            total_weights += weight_params
            total_biases += bias_params
            total_params += layer_params
            
            print(f"🔹 {layer_name}:")
            print(f"   Architecture: {info['input_size']} → {info['output_size']}")
            print(f"   Parameters: {layer_params:,} ({weight_params:,} weights + {bias_params:,} biases)")
            
            # Weight statistics
            weights = info['weight_tensor'].detach().numpy()
            print(f"   Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
            print(f"   Weight mean: {weights.mean():.4f}, std: {weights.std():.4f}")
            print()
        
        print(f"📈 TOTAL STATISTICS:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Total Weights: {total_weights:,}")
        print(f"   Total Biases: {total_biases:,}")
        print(f"   Total Layers: {len(self.layer_info)}")


def main():
    """Main function to run the network visualizer"""
    model_path = "solidmechanics_model_pytorch.pt"
    
    print("🚀 PyTorch Neural Network Graph Visualizer")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = NetworkVisualizer(model_path)
    
    # Load and analyze model
    if not visualizer.load_model():
        return
    
    layer_order = visualizer.analyze_architecture()
    
    if not layer_order:
        print("❌ No valid layers found in the model")
        return
    
    # Print comprehensive summary
    visualizer.print_summary()
    
    # Create simple left-to-right visualization
    print("\n🎯 Creating clean network visualization...")
    visualizer.visualize_network_simple(figsize=(18, 8), save_path='pinn_network_clean.png')
    
    # Create weight-colored visualization
    print("\n🌈 Creating weight-colored network visualization...")
    visualizer.visualize_network_with_weights(figsize=(18, 8), weight_threshold=0.05, save_path='pinn_network_with_weights.png')
    
    # Create weight heatmaps for first few layers
    print("\n🔥 Creating weight heatmaps...")
    for i, layer_name in enumerate(layer_order[:3]):  # First 3 layers
        visualizer.create_weight_heatmap(layer_name, 
                                       save_path=f'weight_heatmap_{layer_name.replace(".", "_")}.png')
    
    print("\n✅ Visualization complete! Generated files:")
    print("   📁 pinn_network_clean.png - Clean left-to-right network visualization")
    print("   📁 pinn_network_with_weights.png - Network with weight-colored edges")
    print("   📁 weight_heatmap_*.png - Weight matrix heatmaps")
    
    # Display the network graph
    plt.show()


if __name__ == "__main__":
    main()