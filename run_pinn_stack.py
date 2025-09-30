# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np

# Enable eager execution
tf.config.run_functions_eagerly(True)
from pyDOE2 import lhs
import matplotlib.pyplot as plt
from time import time
import pickle

# Set data type
DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)

# Set constants & model parameters
pi = tf.constant(np.pi, dtype=DTYPE)
E = tf.constant(4e11/3, dtype=DTYPE)   # Young's modulus
v = tf.constant(1/3, dtype=DTYPE)       # Poisson's ratio
E = E/1e11
lmda = tf.constant(E*v/(1-2*v)/(1+v), dtype=DTYPE)
mu = tf.constant(E/(2*(1+v)), dtype=DTYPE)
Q = tf.constant(4.0, dtype=DTYPE)

# Set boundary
xmin = 0.
xmax = 1.
ymin = 0.
ymax = 1.

def u_x_ext(x, y):
    utemp = tf.cos(2*pi*x) * tf.sin(pi*(y))
    return utemp
def u_y_ext(x, y):
    utemp = tf.sin(pi*x) * Q/4*tf.pow(y,4)
    return utemp
def f_x_ext(x,y):
    gtemp = 1.0*(-4*tf.pow(pi,2)*tf.cos(2*pi*x)*tf.sin(pi*y)+pi*tf.cos(pi*x)*Q*tf.pow(y,3)) + 0.5*(-9*tf.pow(pi,2)*tf.cos(2*pi*x)*tf.sin(pi*y)+pi*tf.cos(pi*x)*Q*tf.pow(y,3))
    return gtemp
def f_y_ext(x,y):
    gtemp = lmda*(3*tf.sin(pi*x)*Q*tf.pow(y,2)-2*tf.pow(pi,2)*tf.sin(2*pi*x)*tf.cos(pi*y)) + mu*(6*tf.sin(pi*x)*Q*tf.pow(y,2)-2*tf.pow(pi,2)*tf.sin(2*pi*x)*tf.cos(pi*y)-tf.pow(pi,2)*tf.sin(pi*x)*Q*tf.pow(y,4)/4)
    return gtemp

# Define boundary conditions at top 
def fun_b_yy(x, y):
    return (lmda+2*mu)*Q*tf.sin(pi*x)

# Set number of data points
N_bound = 50
N_r = 1000

# Lower bounds
lb = tf.constant([xmin, ymin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([xmax, ymax], dtype=DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(0)

# Boundary points
x_up = lhs(1,samples=N_bound,random_state=123)
x_up = xmin + (xmax-xmin)*x_up
y_up = np.empty(len(x_up))[:,None]
y_up.fill(ymax)
b_up = np.empty([len(x_up),2])
b_up[:,0,None] = u_x_ext(x_up, y_up)
b_up[:,1,None] = u_y_ext(x_up, y_up)
x_up_train = np.hstack((x_up, y_up))
eux_up_train = np.zeros([len(x_up),1])
Syy_up_train = fun_b_yy(x_up, y_up)

x_lo = lhs(1,samples=N_bound,random_state=123)
x_lo = xmin + (xmax-xmin)*x_lo
y_lo = np.empty(len(x_lo))[:,None]
y_lo.fill(ymin)
b_lo = np.empty([len(x_lo),2])
b_lo[:,0, None] = u_x_ext(x_lo, y_lo)
b_lo[:,1, None] = u_y_ext(x_lo, y_lo)
x_lo_train = np.hstack((x_lo, y_lo))
eux_lo_train = np.zeros([len(x_lo),1])
uy_lo_train = np.zeros([len(x_lo),1])

y_ri = lhs(1,samples=N_bound,random_state=123)
y_ri = ymin + (ymax-ymin)*y_ri
x_ri = np.empty(len(y_ri))[:,None]
x_ri.fill(xmax)
b_ri = np.empty([len(x_ri),2])
b_ri[:,0, None] = u_x_ext(x_ri, y_ri)
b_ri[:,1, None] = u_y_ext(x_ri, y_ri)
x_ri_train = np.hstack((x_ri, y_ri)) 
uy_ri_train = np.zeros([len(x_ri),1])
Sxx_ri_train = np.zeros([len(x_ri),1])

y_le = lhs(1,samples=N_bound,random_state=123)
y_le = ymin + (ymax-ymin)*y_le
x_le = np.empty(len(y_le))[:,None]
x_le.fill(xmin)
b_le = np.empty([len(x_le),2])
b_le[:,0, None] = u_x_ext(x_le, y_le)
b_le[:,1, None] = u_y_ext(x_le, y_le)
x_le_train = np.hstack((x_le, y_le))
uy_le_train = np.zeros([len(x_le),1])
Sxx_le_train = np.zeros([len(x_le),1])

X_b_train = np.concatenate((x_up_train, x_lo_train, x_ri_train, x_le_train))
X_train_list = [x_up_train, x_lo_train, x_ri_train, x_le_train]
eux_b_train = np.concatenate((eux_up_train, eux_lo_train))
uy_b_train = np.concatenate((uy_lo_train, uy_ri_train, uy_le_train))
Sxx_b_train = np.concatenate((Sxx_ri_train, Sxx_le_train))
Syy_b_train = Syy_up_train

# collocation points for PINNs
grid_pt = lhs(2,N_r)
grid_pt[:,0] = xmin + (xmax-xmin)*grid_pt[:,0]
grid_pt[:,1] = ymin + (ymax-ymin)*grid_pt[:,1]
xf = grid_pt[:,0]
yf = grid_pt[:,1]
X_col_train = np.hstack((xf[:,None],yf[:,None]))

fig = plt.figure(figsize=(9,9))
plt.scatter(X_b_train[:,0], X_b_train[:,1], c='blue', marker='X', label='Boundary Points')
plt.scatter(X_col_train[:,0], X_col_train[:,1], c='r', marker='.', alpha=0.1, label='Collocation Points')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.title('Positions of collocation points and boundary data');
plt.savefig('collocation_points.png')

# Define residual of the PDE in x direction
def fun_r_x(x, y, dsxxdx, dsxydy):
    return dsxxdx+dsxydy-f_x_ext(x,y)
# Define residual of the PDE in y direction
def fun_r_y(x, y, dsxydx, dsyydy):
    return dsxydx+dsyydy-f_y_ext(x,y)

def get_r(model, X_col_train):
    x = tf.constant(X_col_train[:, 0:1])
    y = tf.constant(X_col_train[:, 1:2])
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        tape2.watch(y)
        Ux, Uy, Sxx, Syy, Sxy = model(tf.stack([x[:,0], y[:,0]], axis=1))
        sxx = Sxx
        syy = Syy
        sxy = Sxy
    dsxxdx, dsxxdy = tape2.gradient(sxx, (x,y))
    dsyydx, dsyydy = tape2.gradient(syy, (x,y))
    dsxydx, dsxydy = tape2.gradient(sxy, (x,y))

    del tape2
    return fun_r_x(x, y, dsxxdx, dsxydy), fun_r_y(x, y, dsxydx, dsyydy)

def fun_r_const_x(x, y, duxdx, duydy,Sxx):
    return (lmda+2*mu)*duxdx+lmda*duydy-Sxx
def fun_r_const_y(x, y, duxdx, duydy, Syy):
    return (lmda+2*mu)*duydy+lmda*duxdx-Syy
def fun_r_const_xy(x, y, duxdy, duydx, Sxy):
    return 2*mu*0.5*(duxdy+duydx)-Sxy

def get_r_const(model, X_col_train):
    x = tf.constant(X_col_train[:, 0:1])
    y = tf.constant(X_col_train[:, 1:2])
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        tape2.watch(y)
        Ux, Uy, Sxx, Syy, Sxy = model(tf.stack([x[:,0], y[:,0]], axis=1))
        ux = Ux
        uy = Uy
    duxdx, duxdy = tape2.gradient(ux, (x,y))
    duydx, duydy = tape2.gradient(uy, (x,y))
    del tape2
    return fun_r_const_x(x, y, duxdx, duydy,Sxx), fun_r_const_y(x, y, duxdx, duydy, Syy), fun_r_const_xy(x, y, duxdy, duydx, Sxy)

def fun_b_r_ux(ux_up, ux_lo, eux_b_train):
    return   tf.concat([ux_up, ux_lo], 0)-eux_b_train

def fun_b_r_uy(uy_lo, uy_ri,uy_le, uy_b_train):
    return  tf.concat([uy_lo, uy_ri, uy_le], 0)-uy_b_train

def get_b_r_u(model, X_train_list, eux_b_train, uy_b_train):
    x_up_train, x_lo_train, x_ri_train, x_le_train = X_train_list[0], X_train_list[1], X_train_list[2], X_train_list[3]
    x_up, y_up = x_up_train[:, 0:1], x_up_train[:,1:2]
    x_lo, y_lo = x_lo_train[:, 0:1], x_lo_train[:,1:2]
    x_ri, y_ri = x_ri_train[:, 0:1], x_ri_train[:,1:2]
    x_le, y_le = x_le_train[:, 0:1], x_le_train[:,1:2]
    Ux_up, Uy_up, _, _, _  = model(tf.stack([x_up[:,0], y_up[:,0]], axis=1))
    ux_up = Ux_up
    Ux_lo, Uy_lo, _, _, _  = model(tf.stack([x_lo[:,0], y_lo[:,0]], axis=1))
    ux_lo, uy_lo = Ux_lo, Uy_lo
    _, Uy_ri, _, _, _  = model(tf.stack([x_ri[:,0], y_ri[:,0]], axis=1))
    uy_ri = Uy_ri
    _, Uy_le, _, _, _  = model(tf.stack([x_le[:,0], y_le[:,0]], axis=1))
    uy_le = Uy_le
    
    return fun_b_r_ux(ux_up, ux_lo, eux_b_train), fun_b_r_uy(uy_lo, uy_ri,uy_le, uy_b_train)

def fun_b_r_Sxx(Sxx_ri, Sxx_le, Sxx_b_train):
    return tf.concat([Sxx_ri,Sxx_le], 0)-Sxx_b_train

def fun_b_r_Syy(Syy_up, Syy_b_train):
    return Syy_up-Syy_b_train

def get_b_r_S(model,X_train_list , Sxx_b_train, Syy_b_train):
    x_up_train, x_lo_train, x_ri_train, x_le_train = X_train_list[0], X_train_list[1], X_train_list[2], X_train_list[3]
    x_up = tf.constant(x_up_train[:, 0:1])
    y_up = tf.constant(x_up_train[:, 1:2])
    _, _, _, Syy_up, _  = model(tf.stack([x_up[:,0], y_up[:,0]], axis=1))

    x_ri = tf.constant(x_ri_train[:, 0:1])
    y_ri = tf.constant(x_ri_train[:, 1:2])
    _, _, Sxx_ri, _, _ = model(tf.stack([x_ri[:,0], y_ri[:,0]], axis=1))

    x_le = tf.constant(x_le_train[:, 0:1])
    y_le = tf.constant(x_le_train[:, 1:2])
    _, _, Sxx_le, _, _ = model(tf.stack([x_le[:,0], y_le[:,0]], axis=1))
    
    return fun_b_r_Sxx(Sxx_ri, Sxx_le, Sxx_b_train), fun_b_r_Syy(Syy_up, Syy_b_train)

def compute_loss(model, X_col_train, X_train_list, eux_b_train, uy_b_train, Sxx_b_train, Syy_b_train):
    rx, ry = get_r(model, X_col_train)
    phi_r = tf.reduce_mean(tf.abs(rx)) + tf.reduce_mean(tf.abs(ry))

    rx_const, ry_const, rxy_const = get_r_const(model, X_col_train)
    phi_r_const = tf.reduce_mean(tf.abs(rx_const)) + tf.reduce_mean(tf.abs(ry_const))+tf.reduce_mean(tf.abs(rxy_const))

    r_ux, r_uy = get_b_r_u(model, X_train_list, eux_b_train, uy_b_train)
    phi_r_u = tf.reduce_mean(tf.abs(r_ux)) + tf.reduce_mean(tf.abs(r_uy))
    
    r_Sxx, r_Syy = get_b_r_S(model, X_train_list, Sxx_b_train, Syy_b_train)
    phi_r_S = tf.reduce_mean(tf.abs(r_Sxx)) + tf.reduce_mean(tf.abs(r_Syy))
    
    loss = phi_r + phi_r_const + phi_r_u + phi_r_S 
    return loss

def get_grad(model, X_col_train, X_train_list, eux_b_train, uy_b_train, Sxx_b_train, Syy_b_train):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, X_col_train, X_train_list, eux_b_train, uy_b_train, Sxx_b_train, Syy_b_train)
    g = tape.gradient(loss, model.trainable_variables)
    return loss, g

num_hidden_layers, num_neurons_per_layer = 8, 20
input_layer = tf.keras.Input(shape=(2,))

scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
x = scaling_layer(input_layer)

for _ in range(num_hidden_layers):
    x = tf.keras.layers.Dense(num_neurons_per_layer, activation=tf.keras.activations.get('tanh'),
        kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

output_Ux = tf.keras.layers.Dense(1)(x)
output_Uy = tf.keras.layers.Dense(1)(x)
output_Sxx = tf.keras.layers.Dense(1)(x)
output_Syy = tf.keras.layers.Dense(1)(x)
output_Sxy = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=[output_Ux, output_Uy, output_Sxx, output_Syy, output_Sxy])

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
optim = tf.keras.optimizers.Adam(learning_rate=lr)

@tf.function
def train_step():
    loss, grad_theta = get_grad(model, X_col_train,X_train_list,eux_b_train, uy_b_train, Sxx_b_train, Syy_b_train)
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    return loss

# Number of training epochs
N = 10000
hist = []

t0 = time()

for i in range(N+1):
  loss = train_step()
  if i==0:
    loss0 = loss
  hist.append(loss.numpy()/loss0.numpy())
  if i%50 == 0:
      print('It {:05d}: loss = {:10.8e}'.format(i,loss))
        
print('\nComputation time: {} seconds'.format(time()-t0))

# Set up meshgrid
N_plot = 600
xspace = np.linspace(lb[0], ub[0], N_plot + 1)
yspace = np.linspace(lb[1], ub[1], N_plot + 1)
X, Y = np.meshgrid(xspace, yspace)
Xgrid = np.vstack([X.flatten(),Y.flatten()]).T

# Determine predictions of u(t, x)
Ux_pred, Uy_pred, Sxx_pred, Syy_pred, Sxy_pred = model(tf.cast(Xgrid,DTYPE))
ux_pred = Ux_pred
uy_pred = Uy_pred 

# Reshape upred
Ux = ux_pred.numpy().reshape(N_plot+1,N_plot+1)
Uy = uy_pred.numpy().reshape(N_plot+1,N_plot+1)

U_total = [Ux, Uy]
U_total_name = ['Ux_NN', 'Uy_NN']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
for i,ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, U_total[i], cmap='seismic', vmin=-0.8, vmax=0.8)
    ax.set_title(U_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
fig.savefig('solution.png')

# calculate exact solutions
ux_ext_flat = u_x_ext(X.flatten(),Y.flatten())
uy_ext_flat = u_y_ext(X.flatten(),Y.flatten())

# Reshape
Ux_ext = ux_ext_flat.numpy().reshape(N_plot+1,N_plot+1)
Uy_ext = uy_ext_flat.numpy().reshape(N_plot+1,N_plot+1)

error_total = [abs(Ux-Ux_ext), abs(Uy-Uy_ext)]
error_total_name = ['point wise error Ux', 'point wise error Uy']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
for i,ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, error_total[i], cmap='jet')
    ax.set_title(error_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
fig.savefig('error_map.png')

# Determine predictions of stress
sxx_pred = Sxx_pred
syy_pred = Syy_pred
sxy_pred = Sxy_pred

# Reshape
Sxx = sxx_pred.numpy().reshape(N_plot+1,N_plot+1)
Syy = syy_pred.numpy().reshape(N_plot+1,N_plot+1)
Sxy = sxy_pred.numpy().reshape(N_plot+1,N_plot+1)

S_total = [Sxx, Syy, Sxy]
S_total_name = ['Sxx_NN', 'Syy_NN', 'Sxy_NN']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
for i,ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, S_total[i], cmap='seismic', vmin=-10, vmax=10)
    ax.set_title(S_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
fig.savefig('stress_map.png')

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
ax.semilogy(range(len(hist)), hist,'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$');
fig.savefig('loss_history.png')

filename = 'solidmechanics_model_stack.sav'
with open(filename, 'wb') as f:
    pickle.dump(model, f)