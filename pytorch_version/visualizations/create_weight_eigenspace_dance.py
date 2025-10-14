#!/usr/bin/env python3
"""
Weight Eigenspace Dance Visualizer for PINN
Shows PCA/SVD of weight matrices - watch principal components dance as the network learns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import glob
from visualize_network_graph import NetworkVisualizer
from PIL import Image
# Using NumPy/SciPy instead of sklearn to avoid version conflicts
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class WeightEigenspaceDancer:
    def __init__(self, model_snapshots_dir='weight_snapshots'):
        self.snapshots_dir = model_snapshots_dir
        self.weight_evolution = {}
        self.epochs = []
        
    def load_weight_evolution(self):
        """Load all weight snapshots and track evolution"""
        print("🎭 Loading weight evolution for eigenspace dance...")
        
        model_files = sorted(glob.glob(f'{self.snapshots_dir}/temp_model_epoch_*.pt'))
        
        if not model_files:
            print("❌ No model snapshots found")
            return False
        
        # Initialize storage for each layer
        first_file = model_files[0]
        visualizer = NetworkVisualizer(first_file)
        visualizer.load_model()
        visualizer.analyze_architecture()
        
        layer_names = list(visualizer.layer_info.keys())
        for layer_name in layer_names:
            self.weight_evolution[layer_name] = []
        
        # Load all epochs
        for model_file in model_files:
            epoch = int(model_file.split('_')[-1].split('.')[0])
            
            visualizer = NetworkVisualizer(model_file)
            if visualizer.load_model():
                visualizer.analyze_architecture()
                
                self.epochs.append(epoch)
                
                for layer_name, info in visualizer.layer_info.items():
                    weights = info['weight_tensor'].detach().numpy()
                    self.weight_evolution[layer_name].append(weights)
        
        print(f"✅ Loaded weight evolution for {len(self.epochs)} epochs across {len(layer_names)} layers")
        return True
    
    def compute_weight_pca(self, weight_matrix, n_components=3):
        """Compute PCA of weight matrix using pure NumPy"""
        
        if len(weight_matrix.shape) != 2 or weight_matrix.shape[0] <= 1:
            return None
            
        # Center the data (subtract mean)
        weight_centered = weight_matrix - np.mean(weight_matrix, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(weight_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute explained variance
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues / total_variance if total_variance > 0 else eigenvalues
        
        return {
            'components': eigenvectors[:, :n_components].T,
            'explained_variance': explained_variance[:n_components],
            'eigenvalues': eigenvalues,
            'weight_vector': weight_matrix.flatten()
        }
    
    def compute_weight_svd(self, weight_matrix):
        """Compute SVD of weight matrix"""
        
        if len(weight_matrix.shape) != 2:
            return None
        
        try:
            U, s, Vh = np.linalg.svd(weight_matrix, full_matrices=False)
            
            return {
                'U': U,           # Left singular vectors
                's': s,           # Singular values  
                'Vh': Vh,         # Right singular vectors
                'rank': np.sum(s > 1e-10),
                'condition_number': s[0] / s[-1] if s[-1] > 1e-10 else np.inf
            }
        except:
            return None
    
    def create_eigenspace_dance_frame(self, epoch_idx, layer_name='hidden_layers.0'):
        """Create a single frame of the eigenspace dance"""
        
        epoch = self.epochs[epoch_idx]
        print(f"💃 Creating eigenspace dance frame for epoch {epoch} - {layer_name}")
        
        if layer_name not in self.weight_evolution:
            print(f"❌ Layer {layer_name} not found")
            return None
        
        current_weights = self.weight_evolution[layer_name][epoch_idx]
        
        # Compute SVD for current epoch
        svd_data = self.compute_weight_svd(current_weights)
        if svd_data is None:
            print(f"❌ SVD computation failed for epoch {epoch}")
            return None
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.patch.set_facecolor('black')
        
        # === Singular Values Evolution (Top-left) ===
        ax = axes[0, 0]
        
        # Plot singular value evolution up to current epoch
        singular_values_evolution = []
        for i in range(epoch_idx + 1):
            weights = self.weight_evolution[layer_name][i]
            svd = self.compute_weight_svd(weights)
            if svd:
                singular_values_evolution.append(svd['s'])
        
        if singular_values_evolution:
            # Plot evolution of top singular values
            epochs_so_far = self.epochs[:epoch_idx + 1]
            max_components = min(5, len(singular_values_evolution[0]))
            
            colors = plt.cm.plasma(np.linspace(0, 1, max_components))
            for comp_idx in range(max_components):
                values = [sv[comp_idx] if comp_idx < len(sv) else 0 
                         for sv in singular_values_evolution]
                ax.plot(epochs_so_far, values, color=colors[comp_idx], 
                       linewidth=2, label=f'σ_{comp_idx+1}')
        
        ax.set_xlabel('Epoch', color='white')
        ax.set_ylabel('Singular Value', color='white')
        ax.set_title('Singular Value Evolution\n"Principal Component Strength"', 
                    color='white', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        
        # === Principal Component Directions (Top-middle) ===
        ax = axes[0, 1]
        
        # Visualize first few principal components as vectors
        U = svd_data['U']
        s = svd_data['s']
        
        # Plot principal directions as arrows
        max_components = min(3, U.shape[1])
        colors = ['red', 'blue', 'green']
        
        for comp_idx in range(max_components):
            # Take first few elements of each principal component
            pc_vector = U[:min(10, U.shape[0]), comp_idx]
            
            # Create positions for visualization
            positions = np.arange(len(pc_vector))
            
            # Plot as arrows showing direction and magnitude
            for i, val in enumerate(pc_vector):
                color = colors[comp_idx]
                arrow_props = dict(arrowstyle='->', color=color, alpha=0.8, lw=2)
                ax.annotate('', xy=(i, val), xytext=(i, 0), arrowprops=arrow_props)
                
                # Add star at tip
                ax.scatter(i, val, s=100*abs(val)*s[comp_idx], color=color, alpha=0.8, 
                          edgecolors='white', linewidth=1)
        
        ax.set_xlabel('Neuron Index', color='white')
        ax.set_ylabel('Component Weight', color='white')
        ax.set_title('Principal Component Directions\n"Dance Choreography"', 
                    color='white', fontweight='bold')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        # === Weight Matrix Heatmap (Top-right) ===
        ax = axes[0, 2]
        
        im = ax.imshow(current_weights, cmap='RdBu_r', aspect='auto')
        ax.set_title('Current Weight Matrix\n"Dance Floor"', 
                    color='white', fontweight='bold')
        ax.set_xlabel('Input Neurons', color='white')
        ax.set_ylabel('Output Neurons', color='white')
        ax.tick_params(colors='white')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight Value', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # === 3D Principal Component Space (Bottom-left) ===
        ax = axes[1, 0]
        ax.remove()  # Remove 2D axis
        ax = fig.add_subplot(2, 3, 4, projection='3d')
        
        # Project weight evolution into 3D PC space
        if len(singular_values_evolution) > 3:
            # Take first 3 principal components over time
            pc1_evolution = [sv[0] if len(sv) > 0 else 0 for sv in singular_values_evolution]
            pc2_evolution = [sv[1] if len(sv) > 1 else 0 for sv in singular_values_evolution]
            pc3_evolution = [sv[2] if len(sv) > 2 else 0 for sv in singular_values_evolution]
            
            # Plot 3D trajectory
            ax.plot(pc1_evolution, pc2_evolution, pc3_evolution, 
                   'cyan', linewidth=2, alpha=0.8)
            
            # Plot current position as bright star
            current_pos = (pc1_evolution[epoch_idx], pc2_evolution[epoch_idx], pc3_evolution[epoch_idx])
            ax.scatter(*current_pos, s=200, color='yellow', alpha=1.0, 
                      edgecolors='white', linewidth=2)
            
            # Plot history as fading trail
            for i in range(max(0, epoch_idx-10), epoch_idx):
                alpha = (i - max(0, epoch_idx-10)) / 10
                pos = (pc1_evolution[i], pc2_evolution[i], pc3_evolution[i])
                ax.scatter(*pos, s=50*alpha, color='cyan', alpha=alpha)
        
        ax.set_xlabel('PC1 (σ₁)', color='white')
        ax.set_ylabel('PC2 (σ₂)', color='white')
        ax.set_zlabel('PC3 (σ₃)', color='white')
        ax.set_title('3D Principal Component Trajectory\n"Dance Path"', 
                    color='white', fontweight='bold')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        
        # === Explained Variance (Bottom-middle) ===
        ax = axes[1, 1]
        
        # Show explained variance ratios
        explained_var = svd_data['explained_variance'] if 'explained_variance' in svd_data else s**2 / np.sum(s**2)
        
        bars = ax.bar(range(len(explained_var[:8])), explained_var[:8], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(explained_var[:8]))),
                     alpha=0.8, edgecolor='white', linewidth=1)
        
        ax.set_xlabel('Principal Component', color='white')
        ax.set_ylabel('Explained Variance Ratio', color='white')
        ax.set_title('Principal Component Importance\n"Dancer Hierarchy"', 
                    color='white', fontweight='bold')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, var) in enumerate(zip(bars, explained_var[:8])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{var*100:.1f}%', ha='center', va='bottom', 
                   color='white', fontsize=9, fontweight='bold')
        
        # === Condition Number Evolution (Bottom-right) ===
        ax = axes[1, 2]
        
        # Track condition number evolution
        condition_numbers = []
        for i in range(epoch_idx + 1):
            weights = self.weight_evolution[layer_name][i]
            svd = self.compute_weight_svd(weights)
            if svd and svd['condition_number'] != np.inf:
                condition_numbers.append(svd['condition_number'])
            else:
                condition_numbers.append(np.nan)
        
        epochs_so_far = self.epochs[:epoch_idx + 1]
        valid_mask = ~np.isnan(condition_numbers)
        
        if np.any(valid_mask):
            ax.semilogy(np.array(epochs_so_far)[valid_mask], 
                       np.array(condition_numbers)[valid_mask], 
                       'gold', linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch', color='white')
        ax.set_ylabel('Condition Number (log)', color='white')
        ax.set_title('Matrix Conditioning\n"Dance Stability"', 
                    color='white', fontweight='bold')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        # === Overall Styling ===
        
        # Dance phase detection
        if epoch < 100:
            dance_phase = "🌪️ Chaotic Freestyle"
            dance_desc = "Wild, unstructured movements"
        elif epoch < 500:
            dance_phase = "💃 Learning Choreography"
            dance_desc = "Patterns beginning to emerge"
        elif epoch < 1000:
            dance_phase = "🕺 Synchronized Routine"
            dance_desc = "Coordinated movements"
        else:
            dance_phase = "👑 Master Performance"
            dance_desc = "Elegant, refined dance"
        
        # Add statistics
        stats_text = f"Layer: {layer_name}\n"
        stats_text += f"Matrix Shape: {current_weights.shape}\n"
        stats_text += f"Rank: {svd_data['rank']}/{min(current_weights.shape)}\n"
        stats_text += f"Top Singular Value: {s[0]:.3f}\n"
        stats_text += f"Condition Number: {svd_data['condition_number']:.2e}\n"
        stats_text += f"Spectral Norm: {s[0]:.3f}"
        
        fig.text(0.02, 0.02, stats_text, color='white', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='darkblue', alpha=0.8))
        
        fig.suptitle(f'Weight Eigenspace Dance - {layer_name}\n'
                    f'{dance_phase} - Epoch {epoch}\n{dance_desc}', 
                    color='white', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_principal_component_orbit(self, layer_name='hidden_layers.0'):
        """Create orbital visualization of principal components"""
        print(f"🌌 Creating principal component orbit for {layer_name}...")
        
        if layer_name not in self.weight_evolution:
            return None
        
        # Compute SVD for all epochs
        pc_trajectories = []
        singular_value_trajectories = []
        
        for epoch_idx in range(len(self.epochs)):
            weights = self.weight_evolution[layer_name][epoch_idx]
            svd_data = self.compute_weight_svd(weights)
            
            if svd_data:
                # Store top 3 principal components (first 3 columns of U)
                if svd_data['U'].shape[1] >= 3:
                    pc_trajectories.append(svd_data['U'][:, :3])
                singular_value_trajectories.append(svd_data['s'][:min(8, len(svd_data['s']))])
        
        if not pc_trajectories:
            return None
        
        # Create orbital animation frames
        os.makedirs('orbit_frames', exist_ok=True)
        
        for frame_idx in range(len(pc_trajectories)):
            epoch = self.epochs[frame_idx]
            
            fig = plt.figure(figsize=(20, 12))
            fig.patch.set_facecolor('black')
            
            # === 3D Principal Component Orbit (Left) ===
            ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')
            
            # Plot trajectory up to current frame
            for i in range(min(frame_idx + 1, len(pc_trajectories))):
                pc_matrix = pc_trajectories[i]
                
                # Plot each neuron's position in PC space
                if pc_matrix.shape[1] >= 3:
                    x_coords = pc_matrix[:, 0]
                    y_coords = pc_matrix[:, 1]
                    z_coords = pc_matrix[:, 2]
                    
                    # Color by time (fading trail effect)
                    alpha = 0.3 + 0.7 * (i / max(1, frame_idx))
                    size = 20 + 80 * (i / max(1, frame_idx))
                    
                    # Color by singular values
                    sv = singular_value_trajectories[i]
                    colors = plt.cm.plasma(np.linspace(0, 1, len(x_coords)))
                    
                    ax1.scatter(x_coords, y_coords, z_coords, 
                              c=colors, s=size, alpha=alpha, 
                              edgecolors='white', linewidth=0.5)
            
            # Current position highlighted
            if frame_idx < len(pc_trajectories):
                current_pc = pc_trajectories[frame_idx]
                if current_pc.shape[1] >= 3:
                    ax1.scatter(current_pc[:, 0], current_pc[:, 1], current_pc[:, 2], 
                              s=200, c='yellow', alpha=1.0, 
                              edgecolors='white', linewidth=2, marker='*')
            
            ax1.set_xlabel('PC1', color='white')
            ax1.set_ylabel('PC2', color='white')
            ax1.set_zlabel('PC3', color='white')
            ax1.set_title('Principal Component Orbit\n"Neuron Dance in Eigenspace"', 
                         color='white', fontweight='bold')
            ax1.tick_params(colors='white')
            ax1.view_init(elev=20, azim=45 + frame_idx * 2)  # Slow rotation
            
            # === Singular Values Bar Chart (Top-right) ===
            ax2 = fig.add_subplot(2, 3, 2)
            
            current_sv = singular_value_trajectories[frame_idx]
            bars = ax2.bar(range(len(current_sv)), current_sv,
                          color=plt.cm.plasma(np.linspace(0, 1, len(current_sv))),
                          alpha=0.8, edgecolor='white', linewidth=1)
            
            # Animate bars with "dancing" effect
            for i, bar in enumerate(bars):
                dance_factor = 1 + 0.2 * np.sin(frame_idx * 0.3 + i * 0.5)
                bar.set_height(current_sv[i] * dance_factor)
            
            ax2.set_xlabel('Component Index', color='white')
            ax2.set_ylabel('Singular Value', color='white')
            ax2.set_title('Singular Value Spectrum\n"Dance Energy Levels"', 
                         color='white', fontweight='bold')
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
            
            # === Weight Matrix Evolution (Top-right) ===
            ax3 = fig.add_subplot(2, 3, 3)
            
            current_weights = self.weight_evolution[layer_name][frame_idx]
            im = ax3.imshow(current_weights, cmap='RdBu_r', aspect='auto')
            ax3.set_title('Weight Matrix\n"Dance Floor Pattern"', 
                         color='white', fontweight='bold')
            ax3.set_xlabel('Input Neurons', color='white')
            ax3.set_ylabel('Output Neurons', color='white')
            ax3.tick_params(colors='white')
            
            # === Cumulative Explained Variance (Bottom-left) ===
            ax4 = fig.add_subplot(2, 3, 5)
            
            explained_var = current_sv**2 / np.sum(current_sv**2)
            cumulative_var = np.cumsum(explained_var)
            
            ax4.plot(range(len(cumulative_var)), cumulative_var, 
                    'cyan', linewidth=3, marker='o', markersize=6)
            ax4.fill_between(range(len(cumulative_var)), cumulative_var, 
                           alpha=0.3, color='cyan')
            
            ax4.set_xlabel('Number of Components', color='white')
            ax4.set_ylabel('Cumulative Variance', color='white')
            ax4.set_title('Cumulative Explained Variance\n"Dance Complexity"', 
                         color='white', fontweight='bold')
            ax4.set_facecolor('black')
            ax4.tick_params(colors='white')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            # === Rank and Conditioning (Bottom-right) ===
            ax5 = fig.add_subplot(2, 3, 6)
            
            # Track rank and condition number evolution
            ranks = []
            condition_nums = []
            
            for i in range(frame_idx + 1):
                weights = self.weight_evolution[layer_name][i]
                svd = self.compute_weight_svd(weights)
                if svd:
                    ranks.append(svd['rank'])
                    condition_nums.append(svd['condition_number'] if svd['condition_number'] != np.inf else np.nan)
            
            epochs_so_far = self.epochs[:frame_idx + 1]
            
            # Plot rank evolution
            ax5_twin = ax5.twinx()
            line1 = ax5.plot(epochs_so_far, ranks, 'lime', linewidth=2, label='Matrix Rank')
            
            # Plot condition number
            valid_cond = [c for c in condition_nums if not np.isnan(c)]
            valid_epochs = [epochs_so_far[i] for i, c in enumerate(condition_nums) if not np.isnan(c)]
            
            if valid_cond:
                line2 = ax5_twin.semilogy(valid_epochs, valid_cond, 'orange', linewidth=2, label='Condition Number')
            
            ax5.set_xlabel('Epoch', color='white')
            ax5.set_ylabel('Matrix Rank', color='lime')
            ax5_twin.set_ylabel('Condition Number', color='orange')
            ax5.set_title('Matrix Health\n"Dance Form Quality"', 
                         color='white', fontweight='bold')
            ax5.set_facecolor('black')
            ax5.tick_params(colors='white')
            ax5_twin.tick_params(colors='white')
            ax5.grid(True, alpha=0.3)
            
            # Combined legend
            lines1, labels1 = ax5.get_legend_handles_labels()
            lines2, labels2 = ax5_twin.get_legend_handles_labels()
            ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # === Overall Title ===
            
            # Dance phase detection
            if epoch < 100:
                dance_style = "🌪️ Breakdancing"
            elif epoch < 500:
                dance_style = "💃 Ballet"
            elif epoch < 1000:
                dance_style = "🕺 Ballroom"
            else:
                dance_style = "👑 Swan Lake"
            
            fig.suptitle(f'Weight Eigenspace Dance - {layer_name}\n'
                        f'{dance_style} - Epoch {epoch}', 
                        color='white', fontsize=18, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'orbit_frames/eigenspace_dance_{layer_name.replace(".", "_")}_epoch_{epoch:04d}.png', 
                       dpi=100, bbox_inches='tight', facecolor='black')
            plt.close(fig)
        
        return True
    
    def create_dance_animations(self):
        """Create eigenspace dance animations for multiple layers"""
        print("💃 Creating eigenspace dance animations...")
        
        if not self.weight_evolution:
            return
        
        # Create output directories
        os.makedirs('dance_frames', exist_ok=True)
        os.makedirs('orbit_frames', exist_ok=True)
        
        # Select key layers to animate
        layers_to_animate = []
        for layer_name in self.weight_evolution.keys():
            if 'hidden_layers' in layer_name:
                layer_num = int(layer_name.split('.')[1])
                if layer_num in [0, 3, 7]:  # First, middle, last hidden layers
                    layers_to_animate.append(layer_name)
        
        print(f"🎭 Creating dances for layers: {layers_to_animate}")
        
        # Create frames for each layer
        for layer_name in layers_to_animate:
            print(f"\n💫 Creating eigenspace dance for {layer_name}...")
            
            # Subsample epochs for smooth animation
            epoch_step = max(1, len(self.epochs) // 30)  # Max 30 frames
            
            for epoch_idx in range(0, len(self.epochs), epoch_step):
                try:
                    self.create_eigenspace_dance_frame(epoch_idx, layer_name)
                    self.create_principal_component_orbit(layer_name)
                    
                except Exception as e:
                    print(f"⚠️ Error creating frame for epoch {self.epochs[epoch_idx]}: {e}")
                    continue
        
        # Create GIFs
        self.create_dance_gifs(layers_to_animate)
    
    def create_dance_gifs(self, layers):
        """Create GIFs from dance frames"""
        print("🎬 Creating eigenspace dance GIFs...")
        
        os.makedirs('animations', exist_ok=True)
        
        for layer_name in layers:
            # Dance frames
            frame_files = sorted(glob.glob(f'dance_frames/eigenspace_dance_{layer_name.replace(".", "_")}_epoch_*.png'))
            
            if frame_files:
                images = [Image.open(f) for f in frame_files]
                images[0].save(f'animations/eigenspace_dance_{layer_name.replace(".", "_")}.gif', 
                              save_all=True, append_images=images[1:], 
                              duration=200, loop=0)
                print(f"✅ Eigenspace dance GIF created for {layer_name}")
            
            # Orbit frames
            orbit_files = sorted(glob.glob(f'orbit_frames/eigenspace_dance_{layer_name.replace(".", "_")}_epoch_*.png'))
            
            if orbit_files:
                images = [Image.open(f) for f in orbit_files]
                images[0].save(f'animations/eigenspace_orbit_{layer_name.replace(".", "_")}.gif', 
                              save_all=True, append_images=images[1:], 
                              duration=150, loop=0)
                print(f"✅ Eigenspace orbit GIF created for {layer_name}")

def main():
    """Main function"""
    print("💃 Weight Eigenspace Dance Visualizer")
    print("=" * 50)
    
    dancer = WeightEigenspaceDancer()
    
    # Load weight evolution
    if not dancer.load_weight_evolution():
        print("❌ Failed to load weight evolution")
        print("💡 Run training script first to generate weight snapshots")
        return
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    print("\n💃 Creating eigenspace dance animations...")
    dancer.create_dance_animations()
    
    print("\n🎉 Eigenspace dance visualization complete!")
    print("📁 Generated files:")
    print("   📂 animations/eigenspace_dance_*.gif - Principal component dances")
    print("   📂 animations/eigenspace_orbit_*.gif - 3D orbital trajectories")
    print("   📂 dance_frames/ - Individual dance frames")
    print("   📂 orbit_frames/ - Orbital trajectory frames")
    print("\n💃 Watch your weight matrices perform their eigenspace ballet!")

if __name__ == "__main__":
    main()