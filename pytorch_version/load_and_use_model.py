"""
Example script showing how to load and use a trained PINN model
"""

from model_loader import PINNLoader
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the trained model
    print("🚀 Loading trained PINN model...")
    model_path = "solidmechanics_model_pytorch.pt"
    
    try:
        # Initialize the model loader
        pinn = PINNLoader(model_path, device='cpu')
        
        # Example 1: Predict at specific points
        print("\n📍 Example 1: Predicting at specific points")
        x_points = np.array([0.5, 0.25, 0.75])
        y_points = np.array([0.5, 0.25, 0.75])
        
        # Get displacements only
        u_x, u_y = pinn.predict(x_points, y_points)
        
        # Get all outputs (displacements + stresses)
        u_x_all, u_y_all, s_xx, s_yy, s_xy = pinn.predict_all(x_points, y_points)
        
        print("Coordinates, displacements, and stresses:")
        for i in range(len(x_points)):
            print(f"  Point ({x_points[i]:.2f}, {y_points[i]:.2f}):")
            print(f"    Displacements: u_x = {u_x[i]:.6f}, u_y = {u_y[i]:.6f}")
            print(f"    Stresses: σ_xx = {s_xx[i]:.6f}, σ_yy = {s_yy[i]:.6f}, σ_xy = {s_xy[i]:.6f}")
        
        # Example 2: Predict on a grid and plot
        print("\n🎨 Example 2: Plotting displacement fields")
        pinn.plot_displacement_fields(
            x_range=(0, 1),
            y_range=(0, 1),
            resolution=100,
            figsize=(12, 5)
        )
        
        print("\n🎨 Example 2b: Plotting stress fields")
        pinn.plot_stress_fields(
            x_range=(0, 1),
            y_range=(0, 1),
            resolution=100,
            figsize=(15, 5)
        )
        
        # Example 3: Save predictions to file
        print("\n💾 Example 3: Saving predictions to files")
        pinn.save_predictions("displacement_field.npz", resolution=50)
        pinn.save_predictions("displacement_field.csv", resolution=50)
        
        # Example 4: Custom analysis - maximum displacement
        print("\n📊 Example 4: Finding maximum displacements")
        X, Y, U_x, U_y = pinn.predict_grid((0, 1), (0, 1), resolution=100)
        
        max_u_x = np.max(np.abs(U_x))
        max_u_y = np.max(np.abs(U_y))
        max_u_x_idx = np.unravel_index(np.argmax(np.abs(U_x)), U_x.shape)
        max_u_y_idx = np.unravel_index(np.argmax(np.abs(U_y)), U_y.shape)
        
        print(f"Maximum |u_x|: {max_u_x:.6f} at ({X[max_u_x_idx]:.3f}, {Y[max_u_x_idx]:.3f})")
        print(f"Maximum |u_y|: {max_u_y:.6f} at ({X[max_u_y_idx]:.3f}, {Y[max_u_y_idx]:.3f})")
        
        # Example 5: Line plots along specific directions
        print("\n📈 Example 5: Creating line plots")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot along x-direction at y=0.5
        x_line = np.linspace(0, 1, 100)
        y_line = np.full_like(x_line, 0.5)
        u_x_line, u_y_line = pinn.predict(x_line, y_line)
        
        ax1.plot(x_line, u_x_line, 'b-', label='u_x')
        ax1.plot(x_line, u_y_line, 'r-', label='u_y')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Displacement')
        ax1.set_title('Displacement along y=0.5')
        ax1.legend()
        ax1.grid(True)
        
        # Plot along y-direction at x=0.5
        y_line = np.linspace(0, 1, 100)
        x_line = np.full_like(y_line, 0.5)
        u_x_line, u_y_line = pinn.predict(x_line, y_line)
        
        ax2.plot(y_line, u_x_line, 'b-', label='u_x')
        ax2.plot(y_line, u_y_line, 'r-', label='u_y')
        ax2.set_xlabel('y')
        ax2.set_ylabel('Displacement')
        ax2.set_title('Displacement along x=0.5')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\n✅ All examples completed successfully!")
        print("\nℹ️  You can now use the PINNLoader class in your own scripts:")
        print("   from model_loader import PINNLoader")
        print("   pinn = PINNLoader('solidmechanics_model_pytorch.pt')")
        print("   u_x, u_y = pinn.predict(x_coords, y_coords)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the model file 'solidmechanics_model_pytorch.pt' exists in the current directory")

if __name__ == "__main__":
    main()