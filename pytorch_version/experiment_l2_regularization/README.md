# L2 Regularization for Physics-Informed Neural Networks

## Theory and Mathematical Framework

### What is L2 Regularization?

L2 regularization (also called **weight decay** or **Tikhonov regularization**) adds a penalty term to the loss function that discourages large weights in the neural network. This is a fundamental technique in machine learning to prevent overfitting and improve generalization.

### Mathematical Formulation

The standard PINN loss function without regularization is:

```
L_total = L_PDE + L_constitutive + L_boundary
```

Where:
- **L_PDE**: Residual of the partial differential equations (momentum balance)
- **L_constitutive**: Residual of constitutive relations (stress-strain)
- **L_boundary**: Residual of boundary conditions

With L2 regularization, we modify this to:

```
L_total = L_physics + λ * L_L2

where:
L_physics = L_PDE + L_constitutive + L_boundary
L_L2 = Σ ||W_i||²₂  (sum of squared L2 norms of all weight matrices)
```

The L2 norm of a weight matrix W is:

```
||W||²₂ = Σᵢⱼ W²ᵢⱼ
```

The hyperparameter **λ** (lambda) controls the strength of regularization:
- λ = 0: No regularization (baseline)
- λ small (e.g., 1e-6): Weak regularization
- λ large (e.g., 1e-3): Strong regularization

### Why L2 Regularization?

#### 1. **Prevents Overfitting**
PINNs can "memorize" training points without truly learning the underlying physics. L2 regularization:
- Forces the network to use simpler, smoother functions
- Prevents individual weights from becoming too large
- Encourages distributed representations across many neurons

#### 2. **Improves Generalization**
By penalizing complexity, the network learns solutions that:
- Interpolate better between collocation points
- Extrapolate more reliably outside training domain
- Are less sensitive to noise in boundary conditions

#### 3. **Stabilizes Training**
Large weights can cause:
- Gradient explosion/vanishing
- Oscillatory behavior during optimization
- Poor convergence in later training phases

L2 regularization mitigates these by keeping weights bounded.

#### 4. **Implicit Prior on Solution Smoothness**
From a Bayesian perspective, L2 regularization corresponds to placing a Gaussian prior on the weights:

```
P(W) ~ exp(-λ||W||²₂)
```

This encodes the belief that the true solution should be representable with moderate-sized weights, implying smoothness.

### The Optimization Problem

We solve:

```
min_θ [L_physics(θ) + λ Σᵢ ||Wᵢ(θ)||²₂]
```

Where θ represents all trainable parameters (weights and biases).

The gradient with respect to weights becomes:

```
∂L_total/∂W = ∂L_physics/∂W + 2λW
```

This creates an additional "pull" toward zero on each weight, proportional to its current value.

### Algorithm

**Input:** 
- Collocation points X_col
- Boundary points X_bc with conditions
- Regularization strength λ
- Learning rate schedule

**Initialize:**
- Neural network weights W ~ N(0, σ²) using Xavier initialization
- Optimizer (Adam) with learning rate η

**For each epoch:**
1. **Forward pass:**
   - Compute u(x), σ(x) at collocation points
   - Compute gradients via automatic differentiation
   - Evaluate PDE residuals: r_PDE = ∂σ/∂x + f
   - Evaluate constitutive residuals: r_const = σ - C:ε(u)
   - Evaluate boundary residuals: r_BC = u(x_bc) - u_BC

2. **Compute losses:**
   ```
   L_PDE = mean(|r_PDE|)
   L_const = mean(|r_const|)
   L_BC = mean(|r_BC|)
   L_physics = L_PDE + L_const + L_BC
   ```

3. **Compute L2 penalty:**
   ```
   L_L2 = Σ_layers ||W_layer||²₂
   ```

4. **Total loss:**
   ```
   L_total = L_physics + λ * L_L2
   ```

5. **Backward pass:**
   - Compute ∂L_total/∂θ via backpropagation
   - Update: θ ← θ - η * ∂L_total/∂θ

6. **Track residuals:**
   - Record all individual loss components
   - Save for later analysis

### What We Can Analyze

#### 1. **Bias-Variance Tradeoff**
- **Low λ**: Low bias (can fit complex functions), high variance (overfits)
- **High λ**: High bias (restricted to simple functions), low variance (generalizes)
- **Optimal λ**: Balances both

#### 2. **Physics vs. Data Fidelity**
By tracking residuals separately, we can see:
- Does stronger regularization help satisfy physics better?
- Is there a tradeoff between different physics constraints?
- Which equations benefit most from regularization?

#### 3. **Weight Distribution Evolution**
We can analyze:
- How do weight magnitudes evolve during training?
- Does regularization prevent certain layers from dominating?
- Are solutions more "democratic" across neurons?

#### 4. **Convergence Behavior**
- Does regularization slow down initial learning?
- Does it help escape local minima?
- Does it lead to more stable final solutions?

#### 5. **Component-wise Analysis**
Our detailed tracking shows:
```
Total Loss
├── Physics Loss
│   ├── PDE Loss
│   │   ├── PDE X (momentum balance in x)
│   │   └── PDE Y (momentum balance in y)
│   ├── Constitutive Loss
│   │   ├── σ_xx equation
│   │   ├── σ_yy equation
│   │   └── σ_xy equation
│   └── Boundary Loss
│       ├── 8 individual BC residuals
└── L2 Penalty (λ * ||W||²)
```

### Expected Outcomes

#### **Too Little Regularization (λ ≈ 0)**
- May overfit to collocation points
- High-frequency oscillations in solution
- Poor generalization to test points
- Unstable during fine-tuning phase

#### **Moderate Regularization (λ ≈ 1e-5 to 1e-4)**
- Balanced physics satisfaction
- Smooth, physically reasonable solutions
- Good generalization
- Stable training

#### **Too Much Regularization (λ ≈ 1e-2)**
- Underfitting - can't capture solution complexity
- Poor physics satisfaction (high residuals)
- Weights too constrained
- Solution too smooth (loses important features)

### Connection to Elastic Net and Physics

In elasticity problems, L2 regularization is particularly relevant because:

1. **Physical solutions are smooth**: Real displacement/stress fields don't have discontinuities (except at material interfaces/cracks)

2. **Energy minimization**: Elastic problems minimize strain energy, which is quadratic in strain (similar to L2 penalty on weights)

3. **Well-posedness**: Regularization helps with uniqueness and stability of the inverse problem

### Experimental Design

We test λ ∈ {0, 10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³} to cover:
- Baseline (no regularization)
- Weak regularization (1e-6, 1e-5)
- Moderate regularization (1e-4)
- Strong regularization (1e-3)

Each experiment:
- Uses same random seed for fair comparison
- Runs for 5000 epochs with same LR schedule
- Tracks all 17 residual components every 10 epochs
- Saves data for cross-experiment analysis

### How to Interpret Results

**Good indicators:**
- ✅ Lower total physics loss (PDE + constitutive + BC)
- ✅ Balanced residuals across all components
- ✅ Smooth convergence curves
- ✅ Stable final values

**Warning signs:**
- ⚠️ Oscillating residuals
- ⚠️ Some components stuck at high values
- ⚠️ Very slow convergence
- ⚠️ Divergence in later epochs

### References

1. **Tikhonov Regularization**: A.N. Tikhonov, "Solution of incorrectly formulated problems", Soviet Math. Dokl., 1963

2. **Weight Decay in Neural Networks**: Krogh & Hertz, "A Simple Weight Decay Can Improve Generalization", NeurIPS 1991

3. **PINNs**: Raissi et al., "Physics-informed neural networks", Journal of Computational Physics, 2019

4. **Regularization in PINNs**: Wang et al., "Understanding and mitigating gradient flow pathologies in physics-informed neural networks", SIAM Journal on Scientific Computing, 2021

### Usage

```bash
# Run all L2 experiments
python experiment_l2_regularization.py

# Compare with baseline
cd ..
python compare_residuals.py
```

The comparison tool will identify which λ value gives the best physics satisfaction while maintaining good training dynamics.
