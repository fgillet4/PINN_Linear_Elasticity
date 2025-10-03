"""
PyTorch PINN Model Loader Framework
Simple interface to load and use trained PINN models for linear elasticity
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union

class PINN(nn.Module):
    """Physics-Informed Neural Network for Linear Elasticity"""
    
    def __init__(self):
        super(PINN, self).__init__()
        
        # These will be loaded from the model
        self.register_buffer('lb', torch.zeros(2))
        self.register_buffer('ub', torch.ones(2))
        
        # Hidden layers (8 layers, 20 neurons each)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(2, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 20),
        ])
        
        # Output layers for displacement and stress components
        self.output_Ux = nn.Linear(20, 1)
        self.output_Uy = nn.Linear(20, 1)
        self.output_Sxx = nn.Linear(20, 1)
        self.output_Syy = nn.Linear(20, 1)
        self.output_Sxy = nn.Linear(20, 1)
        
    def forward(self, x):
        # Scale input to [-1, 1]
        x_scaled = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        
        # Forward through hidden layers
        h = x_scaled
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
        
        # Outputs
        Ux = self.output_Ux(h)
        Uy = self.output_Uy(h)
        Sxx = self.output_Sxx(h)
        Syy = self.output_Syy(h)
        Sxy = self.output_Sxy(h)
        
        return Ux, Uy, Sxx, Syy, Sxy

class PINNLoader:
    """Framework for loading and using trained PINN models"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the PINN loader
        
        Args:
            model_path: Path to the saved .pt model file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)
        
        # Load the model
        self._load_model()
        
    def _load_model(self):
        """Load the trained model from file"""
        try:
            # Create model instance
            self.model = PINN()
            
            # Load state dict
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            
            # Set model to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"📱 Using device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict displacement fields u_x and u_y at given coordinates
        
        Args:
            x: X coordinates (can be numpy array or torch tensor)
            y: Y coordinates (can be numpy array or torch tensor)
            
        Returns:
            Tuple of (u_x, u_y) as numpy arrays
        """
        # Convert inputs to torch tensors
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype, device=self.device)
            
        # Ensure same shape
        x = x.flatten()
        y = y.flatten()
        
        # Stack coordinates
        coords = torch.stack([x, y], dim=1)
        
        # Predict
        with torch.no_grad():
            Ux, Uy, Sxx, Syy, Sxy = self.model(coords)
            u_x = Ux.squeeze().cpu().numpy()
            u_y = Uy.squeeze().cpu().numpy()
            
        return u_x, u_y
    
    def predict_all(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict all outputs: displacements and stresses at given coordinates
        
        Args:
            x: X coordinates (can be numpy array or torch tensor)
            y: Y coordinates (can be numpy array or torch tensor)
            
        Returns:
            Tuple of (u_x, u_y, s_xx, s_yy, s_xy) as numpy arrays
        """
        # Convert inputs to torch tensors
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype, device=self.device)
            
        # Ensure same shape
        x = x.flatten()
        y = y.flatten()
        
        # Stack coordinates
        coords = torch.stack([x, y], dim=1)
        
        # Predict
        with torch.no_grad():
            Ux, Uy, Sxx, Syy, Sxy = self.model(coords)
            u_x = Ux.squeeze().cpu().numpy()
            u_y = Uy.squeeze().cpu().numpy()
            s_xx = Sxx.squeeze().cpu().numpy()
            s_yy = Syy.squeeze().cpu().numpy()
            s_xy = Sxy.squeeze().cpu().numpy()
            
        return u_x, u_y, s_xx, s_yy, s_xy
    
    def predict_grid(self, x_range: Tuple[float, float], y_range: Tuple[float, float], 
                    resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict displacement fields on a regular grid
        
        Args:
            x_range: (x_min, x_max) range for x coordinates
            y_range: (y_min, y_max) range for y coordinates  
            resolution: Number of points in each dimension
            
        Returns:
            Tuple of (X, Y, U_x, U_y) where X,Y are meshgrids and U_x,U_y are displacement fields
        """
        # Create grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Flatten for prediction
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Predict
        u_x_flat, u_y_flat = self.predict(x_flat, y_flat)
        
        # Reshape back to grid
        U_x = u_x_flat.reshape(X.shape)
        U_y = u_y_flat.reshape(Y.shape)
        
        return X, Y, U_x, U_y
    
    def plot_displacement_fields(self, x_range: Tuple[float, float] = (0, 1), 
                                y_range: Tuple[float, float] = (0, 1), 
                                resolution: int = 100, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot the displacement fields u_x and u_y
        
        Args:
            x_range: Range for x coordinates
            y_range: Range for y coordinates
            resolution: Grid resolution
            figsize: Figure size
        """
        # Get predictions on grid
        X, Y, U_x, U_y = self.predict_grid(x_range, y_range, resolution)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot u_x
        im1 = ax1.contourf(X, Y, U_x, levels=50, cmap='viridis')
        ax1.set_title('Displacement u_x')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)
        
        # Plot u_y  
        im2 = ax2.contourf(X, Y, U_y, levels=50, cmap='plasma')
        ax2.set_title('Displacement u_y')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_stress_fields(self, x_range: Tuple[float, float] = (0, 1), 
                          y_range: Tuple[float, float] = (0, 1), 
                          resolution: int = 100, figsize: Tuple[int, int] = (15, 5)):
        """
        Plot the stress fields σ_xx, σ_yy, and σ_xy
        
        Args:
            x_range: Range for x coordinates
            y_range: Range for y coordinates
            resolution: Grid resolution
            figsize: Figure size
        """
        # Get predictions on grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Flatten for prediction
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Predict all outputs
        _, _, s_xx_flat, s_yy_flat, s_xy_flat = self.predict_all(x_flat, y_flat)
        
        # Reshape back to grid
        S_xx = s_xx_flat.reshape(X.shape)
        S_yy = s_yy_flat.reshape(Y.shape)
        S_xy = s_xy_flat.reshape(X.shape)
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot σ_xx
        im1 = ax1.contourf(X, Y, S_xx, levels=50, cmap='seismic')
        ax1.set_title('Stress σ_xx')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)
        
        # Plot σ_yy  
        im2 = ax2.contourf(X, Y, S_yy, levels=50, cmap='seismic')
        ax2.set_title('Stress σ_yy')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2)
        
        # Plot σ_xy  
        im3 = ax3.contourf(X, Y, S_xy, levels=50, cmap='seismic')
        ax3.set_title('Stress σ_xy')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_predictions(self, filename: str, x_range: Tuple[float, float] = (0, 1), 
                        y_range: Tuple[float, float] = (0, 1), resolution: int = 100):
        """
        Save predictions to a file
        
        Args:
            filename: Output filename (supports .npz, .csv)
            x_range: Range for x coordinates
            y_range: Range for y coordinates
            resolution: Grid resolution
        """
        X, Y, U_x, U_y = self.predict_grid(x_range, y_range, resolution)
        
        if filename.endswith('.npz'):
            np.savez(filename, X=X, Y=Y, U_x=U_x, U_y=U_y)
        elif filename.endswith('.csv'):
            # Flatten and save as CSV
            data = np.column_stack([X.flatten(), Y.flatten(), U_x.flatten(), U_y.flatten()])
            np.savetxt(filename, data, delimiter=',', header='x,y,u_x,u_y', comments='')
        else:
            raise ValueError("Unsupported file format. Use .npz or .csv")
            
        print(f"💾 Predictions saved to {filename}")