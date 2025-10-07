#!/usr/bin/env python3
"""
Test script to check weight initialization behavior
"""

import torch
import torch.nn as nn
import numpy as np

# Test with fixed seed
print("🔬 Testing weight initialization with fixed seed...")

torch.manual_seed(0)
np.random.seed(0)

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(5, 3)
        
        # Xavier initialization
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

# Create two models with same seed
model1 = TestNet()
print("Model 1 - Layer 1 weights:")
print(model1.layer1.weight.data[:2, :])  # Show first 2 rows

torch.manual_seed(0)  # Reset seed
model2 = TestNet()
print("\nModel 2 - Layer 1 weights (same seed):")
print(model2.layer1.weight.data[:2, :])

print(f"\nAre they identical? {torch.equal(model1.layer1.weight.data, model2.layer1.weight.data)}")

# Now test without fixed seed
print("\n" + "="*50)
print("🎲 Testing without fixed seed...")

class TestNet2(nn.Module):
    def __init__(self):
        super(TestNet2, self).__init__()
        self.layer1 = nn.Linear(2, 5)
        nn.init.xavier_normal_(self.layer1.weight)

model3 = TestNet2()
model4 = TestNet2()

print("Model 3 - Layer 1 weights:")
print(model3.layer1.weight.data[:2, :])
print("\nModel 4 - Layer 1 weights (different):")
print(model4.layer1.weight.data[:2, :])

print(f"\nAre they identical? {torch.equal(model3.layer1.weight.data, model4.layer1.weight.data)}")