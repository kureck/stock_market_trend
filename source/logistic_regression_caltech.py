import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import itertools
import random
import time

cdict = {'red':   ((0.0, 0.0, 0.0),
                 (1.0, 1.0, 1.0)),
        'green': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 1.0, 1.0),
                 (1.0, 0.0, 0.0))}
BinaryRdBu = matplotlib.colors.LinearSegmentedColormap('BinaryRdBu', cdict, 2)
cdict = {'red':   ((0.0, 0.9, 0.9),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.9, 0.9),
                   (1.0, 0.9, 0.9)),
         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, 0.9, 0.9))}
LightRdBu = matplotlib.colors.LinearSegmentedColormap('LightRdBu', cdict)
cdict = {'red':   ((0.0, 1.0, 1.0),
                   (0.4, 0.7, 0.7),
                   (0.5, 0.0, 0.0),
                   (0.6, 0.7, 0.7),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.4, 0.7, 0.7),
                   (0.5, 0.0, 0.0),
                   (0.6, 0.7, 0.7),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 1.0, 1.0),
                   (0.4, 0.7, 0.7),
                   (0.5, 0.0, 0.0),
                   (0.6, 0.7, 0.7),
                   (1.0, 1.0, 1.0))}
HalfContour = matplotlib.colors.LinearSegmentedColormap('HalfContour', cdict)

logistic = lambda s: 1.0 / (1.0 + np.exp(-s))

d_x = 2

phi = lambda x: x
d_z = len( phi( np.ones((d_x+1,)) ) ) - 1

N = 100

P_x = lambda: np.array( [1.0] + [np.random.uniform(-1, 1) for i in range(d_x)] ) # simulates P(x)

def generate_target(d, hardness=20.0, offset_ratio=0.25, w_f=None):

  # randomize target weights
  if w_f is None:
      w_f = np.array([np.random.uniform(-hardness * offset_ratio, hardness * offset_ratio)] +
                     [np.random.uniform(-hardness, hardness) for i in range(d)])

  # create target distribution simulator
  f = lambda z: logistic(w_f.dot(z.T))
  P_f = lambda z: ( np.array([np.random.uniform() for i in range(z.shape[0])]) <= f(z) )*2.0-1.0
      # "*2.0-1.0" to scale from [0, 1] to [-1, 1] which are our actual label values

  return w_f, f, P_f

w_f, f, P_f = generate_target(d_z, hardness=12.0)

def generate_data_samples(N, P_x, phi, P_f):
    
  # create samples in our input space (x-space)
  x = np.array([P_x() for i in range(N)])
  
  # transform x-space samples to z-space samples
  z = np.apply_along_axis(phi, 1, x)
  
  # produce classification labels from target distribution
  y = P_f(z)
  
  # create function to calculate cross-entropy error from a hypothesis weight vector
  cross_entropy_error = lambda w: np.mean(np.log(1 + np.exp(-y * w.dot(z.T))))
  
  return x, z, y, cross_entropy_error

x, z, y, cross_entropy_error = generate_data_samples(N, P_x, phi, P_f)  

def generate_fill_data(s=300, phi=lambda x: x):
  # create grid of points
  x_1, x_2 = np.array(np.meshgrid(np.linspace(-1, 1, s), np.linspace(-1, 1, s)))
  
  # reshape the grid to an array of homogenized points
  x_grid = np.hstack((np.ones((s*s, 1)), np.reshape(x_1, (s*s, 1)), np.reshape(x_2, (s*s, 1))))
  
  # transform homogenized points into z-space
  z_grid = np.apply_along_axis(phi, 1, x_grid)
  
  return x_1, x_2, x_grid, z_grid

def apply_to_fill(z_grid, func):
  s = int(np.sqrt(z_grid.shape[0]))
  
  # calculate function at each point on the grid and reshape it back to a grid
  return np.reshape(func(z_grid), (s, s))

x_1, x_2, x_grid, z_grid = generate_fill_data(300, phi)
f_grid = apply_to_fill(z_grid, f)

def plot_data_set_and_hypothesis(x, y, x_1, x_2, f_grid=None, title=''):
  start_time = time.time()
  
  fig = plt.figure(figsize=(6, 6))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_aspect(1)
  ax.set_xlabel(r'$x_1$', fontsize=18)
  ax.set_ylabel(r'$x_2$', fontsize=18)
  if not title == '':
      ax.set_title(title, fontsize=18)
  ax.xaxis.grid(color='gray', linestyle='dashed')
  ax.yaxis.grid(color='gray', linestyle='dashed')
  ax.set_axisbelow(True)
  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.autoscale(False)
  
  if not f_grid is None:
      # plot background probability
      ax.pcolor(x_1, x_2, f_grid, cmap=LightRdBu, vmin=0, vmax=1)
      
      # plot decision boundary
      ax.contour(x_1, x_2, f_grid*2-1, cmap=HalfContour, levels=[-0.5, 0.0, 0.5], vmin=-1, vmax=1)
  
  # plot data set
  ax.scatter(x[:, 1], x[:, 2], s=40, c=y, cmap=BinaryRdBu, vmin=-1, vmax=1)
  
  print('Plot took {:.2f} seconds.'.format(time.time()-start_time))
  
  return fig

# target_fig = plot_data_set_and_hypothesis(x, y, x_1, x_2, f_grid,
#                                           title=r'Target, $N={:}$'.format(N))

# target_fig.savefig('1.png')

def gradient_descent(z, y, w_h=None, eta=1.0, max_iterations=10, epsilon=0.001):
  if w_h == None:
      w_h = np.array([0.0 for i in range(z.shape[1])])
  print w_h
  raw_input()
  
  # save a history of the weight vectors into an array
  w_h_i = [np.copy(w_h)]
  
  for i in range(max_iterations):
    subset_indices = range(z.shape[0])
    print subset_indices
    raw_input()
    # subset_indices = np.random.permutation(z.shape[0])[:N/8] # uncomment for stochastic gradient descent
    
    grad_E_in = np.mean(np.tile(- y[subset_indices] / ( 1.0 + np.exp(y[subset_indices] * w_h.dot(z[subset_indices].T)) ), (z.shape[1], 1)).T * z[subset_indices], axis=0)
    
    w_h -= eta * grad_E_in
    w_h_i.append(np.copy(w_h))
    if np.linalg.norm(grad_E_in) <= np.linalg.norm(w_h) * epsilon:
      break
  
  return np.array(w_h_i)

w_h_i = gradient_descent(z, y, eta=4.0)
w_h = w_h_i[-1]
print('Number of iterations: {:}'.format(w_h_i.shape[0]))

h = lambda z: logistic(w_h.dot(z.T))
h_grid = apply_to_fill(z_grid, h)

full_N_fig = plot_data_set_and_hypothesis(x, y, x_1, x_2, h_grid,
                                          title=r'Hypothesis, $N={:}$'.format(N))
full_N_fig.savefig('2.png')

exit()
def in_sample_error(z, y, h):
  y_h = (h(z) >= 0.5)*2-1
  return np.sum(y != y_h) / float(len(y))

def estimate_out_of_sample_error(P_x, phi, P_f, h, N=10000, phi_h=None):
  x = np.array([P_x() for i in range(N)])
  z = np.apply_along_axis(phi, 1, x)
  if not phi_h is None:
      z_h = np.apply_along_axis(phi_h, 1, x)
  else:
      z_h = z
  y = P_f(z)
  y_h = (h(z_h) >= 0.5)*2-1
  return np.sum(y != y_h) / float(N)

print('Target weights: {:}'.format(w_f))
print('Hypothesis weights: {:}'.format(w_h))
print('Hypothesis in-sample error: {:.2%}'.format(in_sample_error(z, y, h)))
print('Hypothesis out-of-sample error: {:.2%}'.format(estimate_out_of_sample_error(P_x, phi, P_f, h)))

N_subset = 10
subset_indices = np.random.permutation(N)[:N_subset]
x_subset = x[subset_indices, :]
z_subset = z[subset_indices, :]
y_subset = y[subset_indices]

w_h_i_subset = gradient_descent(z_subset, y_subset, eta=10.0)
w_h_subset = w_h_i_subset[-1]
print('Number of iterations: {:}'.format(w_h_i_subset.shape[0]))

h_subset = lambda z: logistic(w_h_subset.dot(z.T))
h_subset_grid = apply_to_fill(z_grid, h_subset)

subset_N_fig = plot_data_set_and_hypothesis(x_subset, y_subset, x_1, x_2, h_subset_grid,
                                            title=r'Hypothesis, $N={:}$'.format(N_subset))

subset_N_fig.savefig('3.png')

naked_fig = plot_data_set_and_hypothesis(x_subset, y_subset, x_1, x_2, None, title=r'Data, $N={:}$'.format(N))

naked_fig.savefig('4.png')

print('Target weights: {:}'.format(w_f))
print('Target in-sample error: {:.2%}'.format(in_sample_error(z, y, f)))
print('Target out-of-sample error: {:.2%}'.format(estimate_out_of_sample_error(P_x, phi, P_f, f)))

print('Hypothesis (N={:}) weights: {:}'.format(N, w_h))
print('Hypothesis (N={:}) in-sample error: {:.2%}'.format(N, in_sample_error(z, y, h)))
print('Hypothesis (N={:}) out-of-sample error: {:.2%}'.format(N, estimate_out_of_sample_error(P_x, phi, P_f, h)))

print('Hypothesis (N={:}) weights: {:}'.format(N_subset, w_h_subset))
print('Hypothesis (N={:}) in-sample error: {:.2%}'.format(N_subset, in_sample_error(z_subset, y_subset, h_subset)))
print('Hypothesis (N={:}) out-of-sample error: {:.2%}'.format(N_subset, estimate_out_of_sample_error(P_x, phi, P_f, h_subset)))

start_time = time.time()

error_histories = []

for runs in range(10):
  N = 201
  x = np.array([P_x() for i in range(N)])
  z = np.apply_along_axis(phi, 1, x)
  y = P_f(z)
  
  error_history = []
  
  for N_subset in range(1, N+1, 4):
    x_subset = x[:N_subset, :]
    z_subset = z[:N_subset, :]
    y_subset = y[:N_subset]
    
    w_h = gradient_descent(z_subset, y_subset)[-1]
    h = lambda z: logistic(w_h.dot(z.T))
    
    error_history.append([N_subset,
                          in_sample_error(z_subset, y_subset, h),
                          estimate_out_of_sample_error(P_x, phi, P_f, h)])
  
  error_histories.append(error_history)

error_history = np.mean(np.array(error_histories), axis=0)

print('Error history took {:.2f} seconds.'.format(time.time()-start_time))

target_error = estimate_out_of_sample_error(P_x, phi, P_f, f, N=100000)

start_time = time.time()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'Number of Samples ($N$)', fontsize=18)
ax.set_ylabel(r'Error ($E$)', fontsize=18)
ax.set_title(r'Learning Curve'.format(N), fontsize=18)
ax.set_xlim(0, error_history[-1, 0])
ax.set_ylim(0, 1)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_axisbelow(True)

ax.plot(error_history[:, 0], error_history[:, 1], 'r-', label='In-Sample')
ax.plot(error_history[:, 0], error_history[:, 2], 'b-', label='Out-of-Sample')
ax.plot(error_history[[0, -1], 0], [target_error]*2, 'm-', label='Target')
ax.legend()

print('Plot took {:.2f} seconds.'.format(time.time()-start_time))

def plot_gradient_descent(w_h_i, cross_entropy_error):
  start_time = time.time()
  
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_xlabel(r'Iteration', fontsize=18)
  ax.set_ylabel(r'In-Sample Error ($E_{in}$)', fontsize=18)
  ax.set_title(r'Gradient Descent Evolution'.format(N), fontsize=18)
  ax.set_xlim(0, w_h_i.shape[0]-1)
  ax.set_ylim(0, 1)
  ax.xaxis.grid(color='gray', linestyle='dashed')
  ax.yaxis.grid(color='gray', linestyle='dashed')
  ax.set_axisbelow(True)
  
  ax.plot(range(w_h_i.shape[0]), np.apply_along_axis(cross_entropy_error, 1, w_h_i), 'r-')
  
  print('Plot took {:.2f} seconds.'.format(time.time()-start_time))

plot_gradient_descent(w_h_i, cross_entropy_error)

def visualize_error_surface_slices(w_h_i, cross_entropy_error, s=150, figsize=(6, 6)):
  d_z = w_h_i.shape[1]
  
  w_h_i_mean = np.mean(w_h_i, axis=0)
  w_h_i_std = np.std(w_h_i, axis=0)
  w_h_i_min_extent = w_h_i_mean - 4 * np.max(w_h_i_std)
  w_h_i_max_extent = w_h_i_mean + 4 * np.max(w_h_i_std)
  w_h_i_colors = cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, w_h_i.shape[0]-1),
                                   cmap=cm.gist_rainbow_r).to_rgba(range(w_h_i.shape[0]-1))
  
  for i_x, i_y in itertools.combinations(list(range(1, d_z)) + [0], 2):

    start_time = time.time()
    
    components = list(range(d_z))
    components.remove(i_x)
    components.remove(i_y)
    
    w_zs = [w_h[i] * np.ones((s*s, 1)) for i in components]
    w_x, w_y = np.array(np.meshgrid(np.linspace(w_h_i_min_extent[i_x], w_h_i_max_extent[i_x], s),
                                    np.linspace(w_h_i_min_extent[i_y], w_h_i_max_extent[i_y], s)))
    
    restack = [None] * (d_z)
    restack[i_x] = np.reshape(w_x, (s*s, 1))
    restack[i_y] = np.reshape(w_y, (s*s, 1))
    for i_z, w_z in zip(components, w_zs):
      restack[i_z] = w_z
    w_grid = np.hstack(restack)
    
    error_grid = np.reshape(np.apply_along_axis(cross_entropy_error, 1, w_grid), (s, s))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)
    ax.set_xlabel(r'$w_{:}$'.format(i_x), fontsize=18)
    ax.set_ylabel(r'$w_{:}$'.format(i_y), fontsize=18)
    if d_z == 3:
      ax.set_title(r'Error Surface ({:} view)'.format({0: '"top-to-bottom"',
                                                       1: '"right-to-left"',
                                                       2: '"back-to-front"'}[components[0]]),
                   fontsize=18)
    else:
      ax.set_title(r'Error Surface', fontsize=18)
    ax.set_xlim(w_h_i_min_extent[i_x], w_h_i_max_extent[i_x])
    ax.set_ylim(w_h_i_min_extent[i_y], w_h_i_max_extent[i_y])
    ax.autoscale(False)
    
    ax.pcolor(w_x, w_y, error_grid, cmap=cm.gist_heat, vmin=np.min(error_grid), vmax=np.max(error_grid))
    
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')
    
    for i in range(w_h_i.shape[0]-1):
      ax.plot(w_h_i[i:i+2, i_x], w_h_i[i:i+2, i_y], '-', c=w_h_i_colors[i])
    
    print('Plot took {:.2f} seconds.'.format(time.time()-start_time))

visualize_error_surface_slices(w_h_i, cross_entropy_error, s=150)

d_x = 2

phi = lambda x: np.array([1, x[1], x[2], x[1]*x[2], x[1]**2, x[2]**2])
d_z = len( phi( np.ones((d_x+1,)) ) ) - 1

N = 30

w_f, f, P_f = generate_target(d_z, w_f=np.array([-3, 2, 3, 6, 9, 10]))
x, z, y, cross_entropy_error = generate_data_samples(N, P_x, phi, P_f)
x_1, x_2, x_grid, z_grid = generate_fill_data(300, phi)
f_grid = apply_to_fill(z_grid, f)
target_fig = plot_data_set_and_hypothesis(x, y, x_1, x_2, f_grid, title=r'Target, $N={:}$'.format(N))
target_fig.savefig('7.png')

w_h_i = gradient_descent(z, y, eta=4.0)
w_h = w_h_i[-1]
print('Number of iterations: {:}'.format(w_h_i.shape[0]))

h = lambda z: logistic(w_h.dot(z.T))
h_grid = apply_to_fill(z_grid, h)
hypothesis_fig = plot_data_set_and_hypothesis(x, y, x_1, x_2, h_grid, title=r'Hypothesis, $N={:}$'.format(N))
hypothesis_fig.savefig('8.png')

print('Target weights: {:}'.format(w_f))
print('Hypothesis weights: {:}'.format(w_h))
print('Hypothesis in-sample error: {:.2%}'.format(in_sample_error(z, y, h)))
print('Hypothesis out-of-sample error: {:.2%}'.format(estimate_out_of_sample_error(P_x, phi, P_f, h)))

plot_gradient_descent(w_h_i, cross_entropy_error)

visualize_error_surface_slices(w_h_i, cross_entropy_error, s=150)