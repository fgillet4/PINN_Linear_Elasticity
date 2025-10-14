#!/usr/bin/env python3
"""
Compare residuals across different experiments to analyze which approach performs best
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_residual_data(experiment_dir='.'):
    """Load all residual .npy files from experiment directory"""
    residual_files = glob.glob(os.path.join(experiment_dir, 'residuals_*.npy'))
    
    data = {}
    for file in residual_files:
        exp_name = os.path.basename(file).replace('residuals_', '').replace('.npy', '')
        data[exp_name] = np.load(file, allow_pickle=True).item()
        print(f"Loaded: {exp_name}")
    
    return data

def compare_experiments(data_dict, output_prefix='comparison'):
    """Create comprehensive comparison plots"""
    
    if not data_dict:
        print("No data to compare!")
        return
    
    # Figure 1: Main loss components comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment Comparison: Main Loss Components', fontsize=16, fontweight='bold')
    
    # Total Loss
    ax = axes[0, 0]
    for exp_name, data in data_dict.items():
        epochs = data['epochs']
        total_loss = data['residuals']['total_loss']
        ax.semilogy(epochs, total_loss, label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss (log scale)')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PDE Loss
    ax = axes[0, 1]
    for exp_name, data in data_dict.items():
        epochs = data['epochs']
        pde_loss = data['residuals']['pde_loss']
        ax.semilogy(epochs, pde_loss, label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PDE Loss (log scale)')
    ax.set_title('PDE Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Constitutive Loss
    ax = axes[1, 0]
    for exp_name, data in data_dict.items():
        epochs = data['epochs']
        const_loss = data['residuals']['constitutive_loss']
        ax.semilogy(epochs, const_loss, label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Constitutive Loss (log scale)')
    ax.set_title('Constitutive Equation Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Boundary Loss
    ax = axes[1, 1]
    for exp_name, data in data_dict.items():
        epochs = data['epochs']
        bc_loss = np.array(data['residuals']['boundary_u_loss']) + np.array(data['residuals']['boundary_s_loss'])
        ax.semilogy(epochs, bc_loss, label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Boundary Loss (log scale)')
    ax.set_title('Boundary Condition Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_main_components.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_main_components.png")
    
    # Figure 2: PDE Components (X and Y)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('PDE Component Comparison', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    for exp_name, data in data_dict.items():
        epochs = data['epochs']
        pde_x = data['residuals']['pde_x']
        ax.semilogy(epochs, pde_x, label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PDE X Residual (log scale)')
    ax.set_title('PDE X: ∂σₓₓ/∂x + ∂σₓᵧ/∂y = fₓ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for exp_name, data in data_dict.items():
        epochs = data['epochs']
        pde_y = data['residuals']['pde_y']
        ax.semilogy(epochs, pde_y, label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PDE Y Residual (log scale)')
    ax.set_title('PDE Y: ∂σₓᵧ/∂x + ∂σᵧᵧ/∂y = fᵧ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_pde_components.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_pde_components.png")
    
    # Figure 3: Constitutive Components
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Constitutive Equation Comparison', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    for exp_name, data in data_dict.items():
        epochs = data['epochs']
        const_xx = data['residuals']['const_xx']
        ax.semilogy(epochs, const_xx, label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('σₓₓ Residual (log scale)')
    ax.set_title('σₓₓ = (λ+2μ)εₓₓ + λεᵧᵧ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for exp_name, data in data_dict.items():
        epochs = data['epochs']
        const_yy = data['residuals']['const_yy']
        ax.semilogy(epochs, const_yy, label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('σᵧᵧ Residual (log scale)')
    ax.set_title('σᵧᵧ = (λ+2μ)εᵧᵧ + λεₓₓ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    for exp_name, data in data_dict.items():
        epochs = data['epochs']
        const_xy = data['residuals']['const_xy']
        ax.semilogy(epochs, const_xy, label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('σₓᵧ Residual (log scale)')
    ax.set_title('σₓᵧ = 2μεₓᵧ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_constitutive.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_constitutive.png")
    
    # Figure 4: Final residual comparison (bar chart)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    categories = ['Total', 'PDE', 'Constitutive', 'Boundary']
    x = np.arange(len(categories))
    width = 0.8 / len(data_dict)
    
    for i, (exp_name, data) in enumerate(data_dict.items()):
        residuals = data['residuals']
        final_values = [
            residuals['total_loss'][-1],
            residuals['pde_loss'][-1],
            residuals['constitutive_loss'][-1],
            residuals['boundary_u_loss'][-1] + residuals['boundary_s_loss'][-1]
        ]
        ax.bar(x + i*width, final_values, width, label=exp_name)
    
    ax.set_ylabel('Final Residual Value (log scale)')
    ax.set_title('Final Residual Comparison Across Experiments')
    ax.set_xticks(x + width * (len(data_dict) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_final_values.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_final_values.png")

def print_summary_table(data_dict):
    """Print comprehensive summary table"""
    print("\n" + "="*120)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*120)
    
    header = f"{'Experiment':<25} {'Total Loss':<12} {'PDE Loss':<12} {'Const Loss':<12} {'BC Loss':<12} {'Time (s)':<10}"
    print(header)
    print("-"*120)
    
    for exp_name, data in data_dict.items():
        residuals = data['residuals']
        final_total = residuals['total_loss'][-1]
        final_pde = residuals['pde_loss'][-1]
        final_const = residuals['constitutive_loss'][-1]
        final_bc = residuals['boundary_u_loss'][-1] + residuals['boundary_s_loss'][-1]
        training_time = data.get('training_time', 0)
        
        print(f"{exp_name:<25} {final_total:<12.4e} {final_pde:<12.4e} {final_const:<12.4e} {final_bc:<12.4e} {training_time:<10.2f}")
    
    print("="*120)
    
    # Find best performer in each category
    print("\n🏆 BEST PERFORMERS:")
    
    best_total = min(data_dict.items(), key=lambda x: x[1]['residuals']['total_loss'][-1])
    print(f"   Best Total Loss: {best_total[0]} ({best_total[1]['residuals']['total_loss'][-1]:.4e})")
    
    best_pde = min(data_dict.items(), key=lambda x: x[1]['residuals']['pde_loss'][-1])
    print(f"   Best PDE Loss: {best_pde[0]} ({best_pde[1]['residuals']['pde_loss'][-1]:.4e})")
    
    best_const = min(data_dict.items(), key=lambda x: x[1]['residuals']['constitutive_loss'][-1])
    print(f"   Best Constitutive Loss: {best_const[0]} ({best_const[1]['residuals']['constitutive_loss'][-1]:.4e})")
    
    best_bc = min(data_dict.items(), key=lambda x: x[1]['residuals']['boundary_u_loss'][-1] + x[1]['residuals']['boundary_s_loss'][-1])
    final_bc = best_bc[1]['residuals']['boundary_u_loss'][-1] + best_bc[1]['residuals']['boundary_s_loss'][-1]
    print(f"   Best Boundary Loss: {best_bc[0]} ({final_bc:.4e})")
    
    fastest = min(data_dict.items(), key=lambda x: x[1].get('training_time', float('inf')))
    print(f"   Fastest Training: {fastest[0]} ({fastest[1].get('training_time', 0):.2f}s)")

def analyze_convergence(data_dict):
    """Analyze convergence behavior"""
    print("\n" + "="*120)
    print("CONVERGENCE ANALYSIS")
    print("="*120)
    
    for exp_name, data in data_dict.items():
        residuals = data['residuals']
        epochs = data['epochs']
        total_loss = np.array(residuals['total_loss'])
        
        # Calculate improvement metrics
        initial_loss = total_loss[0]
        final_loss = total_loss[-1]
        improvement_ratio = initial_loss / final_loss
        
        # Find epoch where loss reduced to 10% of initial
        try:
            epoch_10pct = epochs[np.where(total_loss <= 0.1 * initial_loss)[0][0]]
        except:
            epoch_10pct = "Not reached"
        
        # Find epoch where loss reduced to 1% of initial
        try:
            epoch_1pct = epochs[np.where(total_loss <= 0.01 * initial_loss)[0][0]]
        except:
            epoch_1pct = "Not reached"
        
        print(f"\n{exp_name}:")
        print(f"  Initial Loss: {initial_loss:.4e}")
        print(f"  Final Loss: {final_loss:.4e}")
        print(f"  Improvement: {improvement_ratio:.2f}x")
        print(f"  Epoch to 10% of initial: {epoch_10pct}")
        print(f"  Epoch to 1% of initial: {epoch_1pct}")

if __name__ == "__main__":
    print("="*120)
    print("RESIDUAL COMPARISON ANALYSIS TOOL")
    print("="*120)
    
    # Check for experiment subdirectories
    exp_dirs = ['experiment_l2_regularization', 'experiment_lr_scheduling']
    
    for exp_dir in exp_dirs:
        if os.path.exists(exp_dir):
            print(f"\n\n{'='*60}")
            print(f"Analyzing: {exp_dir}")
            print('='*60)
            
            data = load_residual_data(exp_dir)
            if data:
                compare_experiments(data, output_prefix=f'{exp_dir}/comparison')
                print_summary_table(data)
                analyze_convergence(data)
    
    # Also check current directory
    print(f"\n\n{'='*60}")
    print("Analyzing: Current directory")
    print('='*60)
    
    data = load_residual_data('.')
    if data:
        compare_experiments(data, output_prefix='comparison')
        print_summary_table(data)
        analyze_convergence(data)
    
    print("\n✅ Analysis complete!")
