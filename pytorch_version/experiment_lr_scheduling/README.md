# Learning Rate Scheduling for Physics-Informed Neural Networks

## Theory and Mathematical Framework

### What is Learning Rate Scheduling?

Learning rate scheduling systematically adjusts the learning rate η during training to improve convergence speed, final accuracy, and training stability. This is crucial for PINNs where the loss landscape is complex due to competing physics constraints.

### Mathematical Foundation

#### Gradient Descent Update Rule

```
θ_{t+1} = θ_t - η_t * ∇L(θ_t)
```

Where:
- θ_t: Parameters at iteration t
- η_t: Learning rate at iteration t (time-varying!)
- ∇L(θ_t): Gradient of loss with respect to parameters

**Key Insight:** The learning rate controls the step size in parameter space. Choosing the right schedule can dramatically improve convergence.

### Why Learning Rate Scheduling?

#### The Multi-Phase Training Problem

PINN training typically has three phases:

**Phase 1: Exploration (Early Training)**
- Need large learning rate to quickly find good regions
- Loss decreases rapidly
- Network learns coarse structure of solution

**Phase 2: Refinement (Mid Training)**  
- Need moderate learning rate
- Balance between exploration and exploitation
- Network fits physics constraints more accurately

**Phase 3: Fine-Tuning (Late Training)**
- Need small learning rate
- Make small adjustments without overshooting
- Polish solution to minimize residuals

#### Why Constant Learning Rate Fails

```
η = constant (e.g., 0.01)
```

**Problems:**
- **Too large**: Oscillates around minimum, never converges precisely
- **Too small**: Converges too slowly, gets stuck in suboptimal regions
- **Just right?**: Unlikely to be optimal for all phases

### Learning Rate Schedules

We test 7 different scheduling strategies:

---

#### 1. **Constant Schedule**

```
η(t) = η_0
```

**Algorithm:**
```
For each epoch t:
    η_t = η_0
```

**Pros:**
- Simple, no hyperparameters
- Predictable behavior

**Cons:**
- Not optimal for any phase
- Can oscillate near convergence

**Use when:** Baseline comparison only

---

#### 2. **Manual Step Schedule**

```
η(t) = η_0,                    if t < 1000
       η_0/10,                 if 1000 ≤ t < 3000
       η_0/20,                 if t ≥ 3000
```

**Algorithm:**
```
Initialize: η = 0.01

At epoch 1000:
    η ← 0.001  (reduce by 10×)

At epoch 3000:
    η ← 0.0005 (reduce by 20×)
```

**Pros:**
- Domain knowledge incorporated
- Works well for this specific problem

**Cons:**
- Not adaptive to actual convergence
- Requires manual tuning
- Not transferable to other problems

**Use when:** You understand training dynamics well

---

#### 3. **Step Decay**

```
η(t) = η_0 * γ^⌊t/s⌋
```

Where:
- s: step size (e.g., 1000 epochs)
- γ: decay factor (e.g., 0.5)

**Algorithm:**
```
Initialize: η = η_0, step_size = 1000, γ = 0.5

Every 1000 epochs:
    η ← γ * η
```

**Example trajectory:**
```
Epoch 0-999:    η = 0.01
Epoch 1000-1999: η = 0.005
Epoch 2000-2999: η = 0.0025
Epoch 3000-3999: η = 0.00125
...
```

**Pros:**
- Systematic reduction
- Easy to understand

**Cons:**
- Discrete jumps can cause instability
- Fixed schedule regardless of progress

**Use when:** You want simple, predictable decay

---

#### 4. **Exponential Decay**

```
η(t) = η_0 * e^(-λt)
```

Or discrete version:
```
η(t) = η_0 * γ^t
```

Where γ ≈ 0.9995 (small decay each step)

**Algorithm:**
```
Initialize: η = η_0, γ = 0.9995

Every epoch:
    η ← γ * η
```

**Mathematical behavior:**
```
t = 0:    η = 0.01
t = 1000: η ≈ 0.0061
t = 2000: η ≈ 0.0037
t = 5000: η ≈ 0.00082
```

**Pros:**
- Smooth, continuous decay
- No sudden jumps
- Guaranteed convergence (η → 0)

**Cons:**
- May decay too slowly or too fast
- Single parameter controls everything

**Use when:** You want smooth decay without tuning schedules

---

#### 5. **Cosine Annealing**

```
η(t) = η_min + (η_0 - η_min)/2 * (1 + cos(πt/T_max))
```

Where T_max is the total number of epochs.

**Algorithm:**
```
Initialize: η_0 = 0.01, η_min = 0, T_max = 5000

For epoch t:
    η_t = η_min + (η_0 - η_min)/2 * (1 + cos(π * t/T_max))
```

**Shape:** Smooth cosine curve from η_0 to η_min

```
η
│   ╱‾‾‾╲
│  ╱     ╲
│ ╱       ╲___
│╱
└─────────────── t
0              T_max
```

**Pros:**
- Smooth, gradual decay
- Natural schedule (based on periodic functions)
- Fast initial learning, gentle final refinement

**Cons:**
- May converge to large η_min if T_max is wrong
- Single cycle may not be optimal

**Use when:** You know total training time

---

#### 6. **Cosine Annealing with Warm Restarts (SGDR)**

```
η(t) = η_min + (η_0 - η_min)/2 * (1 + cos(π * (t mod T_i)/T_i))

T_i = T_0 * T_mult^i  (restart period grows)
```

**Algorithm:**
```
Initialize: η_0 = 0.01, T_0 = 1000, T_mult = 2

Cycle 1 (epochs 0-999):
    Follow cosine from η_0 to η_min
    
At epoch 1000:
    RESTART: η ← η_0
    Next cycle length: T_1 = 1000 * 2 = 2000

Cycle 2 (epochs 1000-2999):
    Follow cosine from η_0 to η_min
    
At epoch 3000:
    RESTART: η ← η_0
    Next cycle length: T_2 = 2000 * 2 = 4000
...
```

**Shape:**
```
η
│ ╱‾╲   ╱‾‾‾‾╲    ╱‾‾‾‾‾‾‾‾╲
│╱  ╲ ╱      ╲  ╱          ╲
│    ╲╱       ╲╱
└────────────────────────────── t
     T_0      T_1        T_2
```

**Pros:**
- Escapes local minima via restarts
- Multiple "attempts" at convergence
- Adaptive exploration/exploitation

**Cons:**
- More hyperparameters
- Can be disruptive if restarts too frequent

**Use when:** Suspected local minima, complex loss landscape

---

#### 7. **Reduce on Plateau**

```
if loss plateaus for 'patience' epochs:
    η ← η * factor
```

**Algorithm:**
```
Initialize: η = η_0, factor = 0.5, patience = 500, best_loss = ∞

For each epoch t:
    current_loss = evaluate_loss()
    
    if current_loss < best_loss - threshold:
        best_loss = current_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve ≥ patience:
        η ← factor * η
        epochs_no_improve = 0
```

**Pros:**
- Adaptive to actual training progress
- Reduces only when needed
- Self-tuning behavior

**Cons:**
- Requires monitoring loss carefully
- Threshold and patience need tuning
- Can be too conservative or aggressive

**Use when:** You want automatic adaptation to convergence

---

### Theoretical Analysis

#### Convergence Theory

For convex problems, gradient descent with appropriate learning rate converges:

```
L(θ_T) - L(θ*) ≤ O(1/√T)    (with constant η)
L(θ_T) - L(θ*) ≤ O(log T/T)  (with diminishing η)
```

Where θ* is the optimal solution.

**For PINNs (non-convex):** Theory is incomplete, but empirically:
- Decreasing learning rate helps escape shallow local minima
- Too fast decay can trap in suboptimal regions
- Restarts help explore different solution basins

#### Learning Rate and Gradient Noise

The gradient in PINNs is computed over finite collocation points:

```
∇L(θ) ≈ 1/N Σᵢ ∇l(θ, xᵢ)
```

This introduces noise. The effective learning rate should satisfy:

```
Σ_t η_t = ∞           (reach any point in parameter space)
Σ_t η_t² < ∞          (converge to a point)
```

**Examples:**
- η_t = 1/t: Satisfies both (Robbins-Monro conditions) ✓
- η_t = 1/√t: Slower decay, also valid ✓
- η_t = constant: Violates second condition ✗

#### Adaptive Moments (Adam Optimizer)

We use Adam, which maintains per-parameter adaptive learning rates:

```
m_t = β₁ * m_{t-1} + (1-β₁) * g_t        (first moment, momentum)
v_t = β₂ * v_{t-1} + (1-β₂) * g_t²       (second moment, variance)

θ_t = θ_{t-1} - η_t * m_t / (√v_t + ε)
```

The schedule η_t modulates this adaptive rate, giving fine control.

### What We Can Analyze

#### 1. **Convergence Speed**
- Which schedule reaches low loss fastest?
- Is there a tradeoff between speed and final accuracy?

Track:
```
Epoch to reach 10% of initial loss
Epoch to reach 1% of initial loss
Final loss value
```

#### 2. **Stability**
- Which schedules show smooth convergence?
- Do any oscillate or diverge?

Track:
```
Standard deviation of loss in final 1000 epochs
Maximum loss increase (if any)
```

#### 3. **Physics Component Balance**
- Does LR schedule affect which equations converge first?
- Are PDE, constitutive, and BC losses balanced?

Track all 17 residual components separately.

#### 4. **Escape from Local Minima**
- Do restarts help find better solutions?
- Does adaptive scheduling prevent premature convergence?

Compare final loss across methods.

#### 5. **Generalization**
- Does slower decay lead to better generalization?
- Is there overfitting with aggressive schedules?

Would need validation set (not included in this experiment).

### Algorithm: Complete PINN Training with LR Scheduling

```
Input:
    Collocation points X_col
    Boundary data (X_bc, u_bc)
    Schedule type and parameters
    Total epochs T

Initialize:
    θ ~ Xavier initialization
    Optimizer = Adam(θ, lr=η_0)
    Scheduler = get_scheduler(schedule_type, params)

For t = 1 to T:
    # Forward pass
    u_pred, σ_pred = PINN(X_col; θ)
    r_PDE = PDE_residual(u_pred, σ_pred, X_col)
    r_const = constitutive_residual(u_pred, σ_pred, X_col)
    r_BC = boundary_residual(u_pred, σ_pred, X_bc, u_bc)
    
    # Compute loss
    L_total = mean(|r_PDE|) + mean(|r_const|) + mean(|r_BC|)
    
    # Backward pass
    g_t = ∇_θ L_total
    
    # Adam update
    m_t = β₁ * m_{t-1} + (1-β₁) * g_t
    v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
    θ_t = θ_{t-1} - η_t * m_t / (√v_t + ε)
    
    # Update learning rate
    η_{t+1} = scheduler.step(L_total, t)
    
    # Track residuals
    if t mod 10 == 0:
        save_residuals(t, L_total, r_PDE, r_const, r_BC)

Return: θ_T, residual_history
```

### Expected Outcomes

| Schedule | Convergence Speed | Final Accuracy | Stability | Best For |
|----------|------------------|----------------|-----------|----------|
| Constant | Slow | Poor | Oscillatory | Baseline only |
| Manual | Fast | Good | Stable | Known dynamics |
| Step | Medium | Good | Stable | Simple problems |
| Exponential | Medium | Good | Very stable | Smooth convergence |
| Cosine | Fast initially | Very good | Stable | Fixed time budget |
| Cosine+Restart | Variable | Excellent | Periodic jumps | Complex landscapes |
| Reduce on plateau | Adaptive | Good | Very stable | Unknown dynamics |

### Practical Guidelines

1. **Start with manual or step decay** for baseline
2. **Try cosine annealing** if you know training time
3. **Use warm restarts** if solution quality varies
4. **Use reduce on plateau** for automatic tuning
5. **Avoid constant** except for debugging

### Connection to Physics

In physics-informed learning, the loss landscape reflects:
- **Multiple competing constraints** (PDE, BC, constitutive)
- **Varying scales** (displacements vs. stresses)
- **Potential local minima** (different approximate solutions)

LR scheduling helps navigate this complex landscape:
- **Fast initial learning**: Find approximate solution quickly
- **Gradual refinement**: Satisfy all constraints simultaneously  
- **Restarts/adaptation**: Escape poor local solutions

### References

1. **Step Decay**: Standard practice since backpropagation era

2. **Exponential Decay**: Theoretical foundations in stochastic approximation (Robbins & Monro, 1951)

3. **Cosine Annealing**: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017

4. **Adaptive LR (Adam)**: Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015

5. **LR in PINNs**: Wang et al., "When and why PINNs fail to train: A neural tangent kernel perspective", Journal of Computational Physics, 2022

### Usage

```bash
# Run all LR scheduling experiments
python experiment_lr_scheduling.py

# This will test 7 schedules:
# - constant (baseline)
# - manual (domain knowledge)
# - step (systematic decay)
# - exponential (smooth decay)
# - cosine (smooth annealing)
# - cosine_warm_restart (SGDR)
# - reduce_on_plateau (adaptive)

# Compare results
cd ..
python compare_residuals.py
```

The comparison will show which schedule achieves the best balance between convergence speed, final accuracy, and training stability for this elasticity problem.
