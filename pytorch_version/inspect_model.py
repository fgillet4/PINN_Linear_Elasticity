"""
Script to inspect the contents of a PyTorch .pt model file
"""

import torch
import numpy as np

def inspect_model(model_path):
    """Inspect the contents of a PyTorch .pt file"""
    
    print(f"🔍 Inspecting model file: {model_path}")
    print("=" * 50)
    
    try:
        # Load the model file
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"📦 File type: {type(checkpoint)}")
        print(f"📊 File size: {torch.numel(torch.tensor([0.0]) if not hasattr(torch, 'numel') else 0)} bytes")
        
        if isinstance(checkpoint, dict):
            print(f"\n🗂️ Dictionary keys: {list(checkpoint.keys())}")
            
            print(f"\n📋 Detailed contents:")
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    print(f"  '{key}': Tensor with shape {value.shape}, dtype {value.dtype}")
                    if value.numel() <= 10:  # Small tensors, show values
                        print(f"    Values: {value.flatten().tolist()}")
                    else:
                        print(f"    Values (first 5): {value.flatten()[:5].tolist()}...")
                else:
                    print(f"  '{key}': {type(value)} = {value}")
                    
            # Count total parameters
            total_params = 0
            trainable_params = 0
            
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    param_count = value.numel()
                    total_params += param_count
                    if 'weight' in key or 'bias' in key:
                        trainable_params += param_count
                        
            print(f"\n📊 Parameter Statistics:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            # Analyze layer structure
            print(f"\n🏗️ Model Architecture:")
            layers = {}
            for key in checkpoint.keys():
                if '.' in key:
                    layer_name = key.split('.')[0]
                    if layer_name not in layers:
                        layers[layer_name] = []
                    layers[layer_name].append(key)
                    
            for layer_name, params in layers.items():
                print(f"  {layer_name}:")
                for param in sorted(params):
                    shape = checkpoint[param].shape
                    print(f"    {param}: {shape}")
                    
        else:
            print(f"\n⚠️ Not a state dict - contains: {type(checkpoint)}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    model_path = "solidmechanics_model_pytorch.pt"
    inspect_model(model_path)