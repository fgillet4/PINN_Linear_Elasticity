#!/usr/bin/env python
# coding: utf-8

# ## 1. Imports and Setup
import tensorflow as tf
import numpy as np
import time
import pickle
import os

print(f"Using TensorFlow version: {tf.__version__}")

# Set data type
DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)

# ## 2. Problem Parameters and Experimental Data

# Define physical constants and boundary parameters
pi = tf.constant(np.pi, dtype=DTYPE)

# Define the domain
xmin, xmax = 0., 1.
ymin, ymax = 0., 1.
lb = tf.constant([xmin, ymin], dtype=DTYPE)
ub = tf.constant([xmax, ymax], dtype=DTYPE)

# --- Trainable Material Parameters ---
# We treat Young's Modulus (E) and the non-linear parameter (k) as trainable variables.
# We start with an initial guess for each.
initial_E = 1.0
initial_k = 0.0
E = tf.Variable(initial_E, dtype=DTYPE, name="E")
k = tf.Variable(initial_k, dtype=DTYPE, name="k")

# A fixed parameter for this problem
nu = tf.constant(0.3, dtype=DTYPE) # Poisson's ratio

# --- Load Experimental Data ---
# Construct the path to the data file relative to the script's location
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir, 'experimental_data.csv')
    exp_data = np.loadtxt(data_file_path, delimiter=',', skiprows=1)
    X_data = tf.constant(exp_data[:, :2], dtype=DTYPE)
    u_data = tf.constant(exp_data[:, 2:], dtype=DTYPE)
    print(f"Successfully loaded {X_data.shape[0]} experimental data points from {data_file_path}")
except IOError:
    print(f"Error: Could not find experimental_data.csv at {data_file_path}")
    exit()

# ## 3. Neural Network and Physics-Informed Model

# Define the neural network architecture
num_hidden_layers = 4
num_neurons_per_layer = 32

input_layer = tf.keras.Input(shape=(2,), dtype=DTYPE)
scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)(input_layer)
x = scaling_layer
for _ in range(num_hidden_layers):
    x = tf.keras.layers.Dense(num_neurons_per_layer, activation=tf.keras.activations.get('tanh'),
        kernel_initializer='glorot_normal')(x)
# Output layer predicts the two displacement components (u_x, u_y)
output_layer = tf.keras.layers.Dense(2)(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# --- Define the Loss Functions ---

# Loss 1: The governing PDE (Equilibrium)
def get_physics_loss(model, x, y):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        u_pred = model(tf.stack([x[:,0], y[:,0]], axis=1))
        u = u_pred[:,0]
        v = u_pred[:,1]

        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)

    u_xx = tape.gradient(u_x, x)
    u_yy = tape.gradient(u_y, y)
    v_xx = tape.gradient(v_x, x)
    v_yy = tape.gradient(v_y, y)
    u_xy = tape.gradient(u_x, y)
    v_xy = tape.gradient(v_x, y)
    
    del tape

    # Non-linear constitutive model: sigma = E*epsilon + k*epsilon^3
    # This is a simplified 1D-like model for demonstration.
    # A full 2D/3D model would be more complex.
    G = E / (2 * (1 + nu))
    
    # Strains
    eps_xx = u_x
    eps_yy = v_y
    eps_xy = 0.5 * (u_y + v_x)

    # Stresses (non-linear)
    sigma_xx = E * eps_xx + k * (eps_xx**3)
    sigma_yy = E * eps_yy + k * (eps_yy**3)
    sigma_xy = G * 2 * eps_xy

    # Derivatives of stress
    # We need to compute these with another gradient tape
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        tape2.watch(y)
        # Re-evaluate strains inside this tape to track gradients
        u_pred_tape = model(tf.stack([x[:,0], y[:,0]], axis=1))
        u_tape = u_pred_tape[:,0]
        v_tape = u_pred_tape[:,1]
        eps_xx_tape = tape2.gradient(u_tape, x)
        eps_yy_tape = tape2.gradient(v_tape, y)
        eps_xy_tape = 0.5 * (tape2.gradient(u_tape, y) + tape2.gradient(v_tape, x))

        # Re-evaluate stresses to track gradients
        sigma_xx_tape = E * eps_xx_tape + k * (eps_xx_tape**3)
        sigma_yy_tape = E * eps_yy_tape + k * (eps_yy_tape**3)
        sigma_xy_tape = G * 2 * eps_xy_tape

    sigma_xx_x = tape2.gradient(sigma_xx_tape, x)
    sigma_yy_y = tape2.gradient(sigma_yy_tape, y)
    sigma_xy_x = tape2.gradient(sigma_xy_tape, x)
    sigma_xy_y = tape2.gradient(sigma_xy_tape, y)
    del tape2

    # Equilibrium equations (assuming no body forces for this example)
    residual_x = sigma_xx_x + sigma_xy_y
    residual_y = sigma_xy_x + sigma_yy_y

    return tf.reduce_mean(tf.square(residual_x)) + tf.reduce_mean(tf.square(residual_y))

# Loss 2: The experimental data
def get_data_loss(model, X_data, u_data):
    u_pred = model(X_data)
    return tf.reduce_mean(tf.square(u_pred - u_data))

# ## 4. Training

# Create collocation points for the physics loss
N_p = 2000
X_p = tf.random.uniform((N_p, 2), minval=[xmin, ymin], maxval=[xmax, ymax], dtype=DTYPE)
x_p = tf.Variable(X_p[:,0:1])
y_p = tf.Variable(X_p[:,1:2])

# Optimizer
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000, 5000],[1e-2, 1e-3, 5e-4, 1e-5])
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# The list of all trainable variables includes the model's weights and our material parameters
# The trainable variables are now handled inside the train_step function

def train_step():
    all_trainable_vars = model.trainable_variables + [E, k]
    with tf.GradientTape() as tape:
        # Calculate the individual loss components
        loss_p = get_physics_loss(model, x_p, y_p)
        loss_d = get_data_loss(model, X_data, u_data)
        
        # Weight the loss components (can be tuned)
        total_loss = 1.0 * loss_p + 10.0 * loss_d

    # Get all gradients in one call
    all_grads = tape.gradient(total_loss, all_trainable_vars)

    # Filter out None gradients to prevent errors
    filtered_grads_and_vars = [(g, v) for g, v in zip(all_grads, all_trainable_vars) if g is not None]

    # Apply gradients in one call
    optim.apply_gradients(filtered_grads_and_vars)
    
    return total_loss, loss_p, loss_d

# Training loop
N_epochs = 8000
hist = []

print("Starting training...")
t0 = time.time()

for i in range(N_epochs + 1):
    loss, loss_p, loss_d = train_step()
    hist.append(loss.numpy())
    
    if i % 100 == 0:
        print(f'It {i:05d}: Total Loss = {loss:.4e}, Physics Loss = {loss_p:.4e}, Data Loss = {loss_d:.4e}, E = {E.numpy():.4f}, k = {k.numpy():.4f}')

# Print final results
t1 = time.time()
print(f"\nComputation time: {t1-t0:.2f} seconds")

print("\n--- Discovered Material Parameters ---")
print(f"Linear Modulus (E): {E.numpy():.5f}")
print(f"Non-Linear Modulus (k): {k.numpy():.5f}")

# Save the trained model and parameters
model.save('nonlinear_discovery_model.keras')
with open('discovered_params.pkl', 'wb') as f:
    pickle.dump({'E': E.numpy(), 'k': k.numpy()}, f)

print("\nModel and discovered parameters have been saved.")
