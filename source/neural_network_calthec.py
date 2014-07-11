import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import itertools
import random
import time

matplotlib.rc('font', family='serif')

cdict = {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, 0.0, 0.0))}
BinaryRdBu = matplotlib.colors.LinearSegmentedColormap('BinaryRdBu', cdict, 2)
cdict = {'red':   ((0.0, 0.7, 0.7),
                   (0.3, 0.2, 0.2),
                   (0.5, 0.5, 0.5),
                   (0.7, 0.6, 0.6),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.7, 0.7),
                   (0.3, 0.2, 0.2),
                   (0.5, 0.5, 0.5),
                   (0.7, 0.2, 0.2),
                   (1.0, 0.7, 0.7)),
         'blue':  ((0.0, 1.0, 1.0),
                   (0.3, 0.6, 0.6),
                   (0.5, 0.5, 0.5),
                   (0.7, 0.2, 0.2),
                   (1.0, 0.7, 0.7))}
RdGrayBu = matplotlib.colors.LinearSegmentedColormap('LightRdBu', cdict)
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

def generate_fill_data(s=336, phi=lambda x: x, extents=(-1, 1, -1, 1)):
  # create grid of points
  x_1, x_2 = np.array(np.meshgrid(np.linspace(extents[0], extents[1], s), np.linspace(extents[2], extents[3], s)))
  # reshape the grid to an array of homogenized points
  x_grid = np.hstack((np.ones((s*s, 1)), np.reshape(x_1, (s*s, 1)), np.reshape(x_2, (s*s, 1))))
  # transform homogenized points into z-space
  z_grid = np.apply_along_axis(phi, 1, x_grid)
  return x_1, x_2, x_grid, z_grid

def apply_to_fill(z_grid, func):
  s = int(np.sqrt(z_grid.shape[0]))
  # calculate function at each point on the grid and reshape it back to a grid
  return np.reshape(func(z_grid), (s, s))

def create_data_plot(title='', extents=(-1, 1, -1, 1)):
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
  ax.set_xlim(extents[0], extents[1])
  ax.set_ylim(extents[2], extents[3])
  ax.autoscale(False)
  return fig, ax

def plot_model(data_plot, x_1, x_2, f_grid):
  fig, ax = data_plot
  ax.pcolor(x_1, x_2, f_grid, cmap=LightRdBu, vmin=-1, vmax=1)
  ax.contour(x_1, x_2, f_grid, cmap=HalfContour, levels=[-0.5, 0.0, 0.5], vmin=-1, vmax=1, antialiased=True)

def plot_data_set(data_plot, x, y):
  fig, ax = data_plot
  ax.scatter(x[:, 1], x[:, 2], s=40, c=y, cmap=BinaryRdBu, vmin=-1, vmax=1)

# x_1, x_2, x_grid, z_grid = generate_fill_data()
# h_grid = apply_to_fill(z_grid, lambda x: np.sign(np.array([0.1, 0.5, 0.3]).dot(x.T)))
# h_fig = create_data_plot('Perceptron')
# plot_model(h_fig, x_1, x_2, h_grid)

# x = np.array([[1, -0.5, -0.5], [1, 0.5, -0.5], [1, -0.5, 0.5], [1, 0.5, 0.5]])
# y = np.array([1, -1, -1, 1])

# w_h_1 = np.array([-0.1, 1, 0.1])
# h_1 = lambda x: np.sign(w_h_1.dot(x.T))
# h_1_grid = apply_to_fill(z_grid, h_1)

# w_h_2 = np.array([0.1, 0.1, -1])
# h_2 = lambda x: np.sign(w_h_2.dot(x.T))
# h_2_grid = apply_to_fill(z_grid, h_2)

# h_1_fig = create_data_plot('Hypothesis $h_1$')
# plot_model(h_1_fig, x_1, x_2, h_1_grid)
# plot_data_set(h_1_fig, x, y)

# h_2_fig = create_data_plot('Hypothesis $h_2$')
# plot_model(h_2_fig, x_1, x_2, h_2_grid)
# plot_data_set(h_2_fig, x, y)

# for x1 in [1, -1]:
#   for x2 in [1, -1]:
#     print('x1={: 2d}, x2={: 2d}, sign(-1.5 + x1 + x2)={: 2d}'.format(x1, x2, int(np.sign(-1.5 + x1 + x2))))

# for x1 in [1, -1]:
#   for x2 in [1, -1]:
#     print('x1={: 2d}, x2={: 2d}, sign(-1.5 + x1 + x2)={: 2d}'.format(x1, x2, int(np.sign(1.5 + x1 + x2))))

# w_h_and = np.array([-1.5, 1, 1])
# h_and = lambda x: np.sign(w_h_and.dot(np.column_stack((np.ones((x.shape[0],)), h_1(x), h_2(x))).T))
# h_and_grid = apply_to_fill(z_grid, h_and)

# h_and_fig = create_data_plot('$h_1$ AND $h_2$ Hypothesis')
# plot_model(h_and_fig, x_1, x_2, h_and_grid)
# plot_data_set(h_and_fig, x, y)

# w_h_or = np.array([1.5, 1, 1])
# h_or = lambda x: np.sign(w_h_or.dot(np.column_stack((np.ones((x.shape[0],)), h_1(x), h_2(x))).T))
# h_or_grid = apply_to_fill(z_grid, h_or)

# h_or_fig = create_data_plot('$h_1$ OR $h_2$ Hypothesis')
# plot_model(h_or_fig, x_1, x_2, h_or_grid)
# plot_data_set(h_or_fig, x, y)

# AND = lambda x1, x2: np.sign(-1.5 + x1 + x2)
# OR = lambda x1, x2: np.sign(1.5 + x1 + x2)
# for x1 in [1, -1]:
#   for x2 in [1, -1]:
#     print('x1={: 2d}, x2={: 2d}, OR(AND(x1, -x2), AND(-x1, x2))={: 2d}'.format(x1, x2, int(OR(AND(x1, -x2), AND(-x1, x2)))))

# w_h_anb = np.array([1, -1, -1.5])
# h_anb = lambda x: np.sign(w_h_anb.dot(np.column_stack((h_1(x), h_2(x), np.ones((x.shape[0],)))).T))
# h_anb_grid = apply_to_fill(z_grid, h_anb)

# w_h_nab = np.array([-1, 1, -1.5])
# h_nab = lambda x: np.sign(w_h_nab.dot(np.column_stack((h_1(x), h_2(x), np.ones((x.shape[0],)))).T))
# h_nab_grid = apply_to_fill(z_grid, h_nab)

# w_h_xor = np.array([1, 1, 1.5])
# h_xor = lambda x: np.sign(w_h_xor.dot(np.column_stack((h_anb(x), h_nab(x), np.ones((x.shape[0],)))).T))
# h_xor_grid = apply_to_fill(z_grid, h_xor)

# h_anb_fig = create_data_plot(r'$h_1$ AND NOT $h_2$ Hypothesis')
# plot_model(h_anb_fig, x_1, x_2, h_anb_grid)
# plot_data_set(h_anb_fig, x, y)

# h_nab_fig = create_data_plot(r'NOT $h_1$ AND $h_2$ Hypothesis')
# plot_model(h_nab_fig, x_1, x_2, h_nab_grid)
# plot_data_set(h_nab_fig, x, y)

# h_xor_fig = create_data_plot(r'$h_1$ XOR $h_2$ Hypothesis')
# plot_model(h_xor_fig, x_1, x_2, h_xor_grid)
# plot_data_set(h_xor_fig, x, y)

## NEURAL

# fig = plt.figure(figsize=(8, 4))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel(r'$s$', fontsize=18)
# ax.set_ylabel(r'$\tanh(s)$', fontsize=18)
# ax.xaxis.grid(color='gray', linestyle='dashed')
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.set_axisbelow(True)
# ax.set_xlim(-4, 4)
# ax.set_ylim(-1.2, 1.2)
# ax.plot(np.linspace(-6, 6, 1000), np.sign(np.linspace(-6, 6, 1000)), 'b-', label='Sign')
# ax.plot(np.linspace(-6, 6, 200), np.tanh(np.linspace(-6, 6, 200)), 'r-', label='Hyperbolic Tangent')
# ax.legend(loc=4)

# h_1 = lambda x: np.tanh(np.array([0, 8, 0.1]).dot(x.T))
# h_2 = lambda x: np.tanh(np.array([0, 0.1, -4]).dot(x.T))
# h_anb = lambda x: np.tanh(np.array([1, -1, -1.1]).dot(np.column_stack((h_1(x), h_2(x), np.ones((x.shape[0],)))).T))
# h_nab = lambda x: np.tanh(np.array([-1, 1, -1.1]).dot(np.column_stack((h_1(x), h_2(x), np.ones((x.shape[0],)))).T))
# h_xor = lambda x: np.tanh(np.array([2, 2, 1.5]).dot(np.column_stack((h_anb(x), h_nab(x), np.ones((x.shape[0],)))).T))

# h_xor_fig = create_data_plot()
# plot_model(h_xor_fig, x_1, x_2, apply_to_fill(z_grid, h_xor))
# plot_data_set(h_xor_fig, x, y)

def initialize_weights(d, P_w=lambda: np.random.normal()):
  L = len(d) - 1
  d_homo = [d_l + 1 for d_l in d]
  w = [ [None] * d_homo[0] ]
  for l in range(1, L + 1):
    w_l = [None]
    for j in range(1, d_homo[l]):
      w_l_j = np.array([P_w() for i in range(d_homo[l-1])])
      w_l.append(w_l_j)
    w.append(w_l)
  return d_homo, L, w

def infer_network_properties_from_w(w):
  d = [len(w_l)-1 for w_l in w]
  d_homo = [len(w_l) for w_l in w]
  L = len(d)-1
  return d, d_homo, L

def draw_network(d, d_homo, L, w, highlights=[], forward=True):
  d_max = max(d)
  
  get_pos = lambda j, l: np.array([1.0 + sum([3.0 * np.sqrt(np.mean([d_homo[l_], d_homo[l_-1]]) / 2.0) for l_ in range(1, l+1)]),
                                   -1.0 + 2.0 * d_homo[l] / 2.0 - j * 2.0])
  
  width = float(1 + get_pos(d_max, L)[0])
  height = float(2 + d_max * 2)
  
  fig = plt.figure(figsize=(width, height))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_xlim(0, width)
  ax.set_ylim(-height/2, height/2)
  ax.autoscale(False)
  
  weight_colorscale = cm.ScalarMappable(norm=matplotlib.colors.Normalize(-3, 3), cmap=RdGrayBu)
  weight_colorscale.set_array(np.linspace(-3, 3, 1000))
  fig.colorbar(weight_colorscale, ticks=np.linspace(-3, 3, 6*2+1))
  
  def parse_highlights(highlight):
    highlight_type = highlight[0]
    highlight_l = int(highlight[2])
    if highlight_type == 'w':
      highlight_i, highlight_j = map(int, highlight[4:].split(','))
    else:
      highlight_i = None
      highlight_j = int(highlight[4])
    highlight_lji = (highlight_l, highlight_j, highlight_i)
    return (highlight_type, highlight_lji)
  
  highlight_color = (0, 1, 0, 1)
  highlights = map(parse_highlights, highlights)
  x_highlights = set(map(lambda a: a[1], filter(lambda a: a[0] == 'x', highlights)))
  w_highlights = set(map(lambda a: a[1], filter(lambda a: a[0] == 'w', highlights)))
  if len(x_highlights) + len(w_highlights) > 0:
    dim_alpha = 0.25
  else:
    dim_alpha = 1.0
  
  for l in range(L+1):
    props = {'fontsize': 24, 'ha': 'center', 'va': 'center'}
    for j in range(d_homo[l]):
      if (l, j, None) in x_highlights:
        color = highlight_color
      else:
        color = (0, 0, 0, dim_alpha)
      pos = get_pos(j, l)
      if forward:
        ax.text(pos[0], pos[1], '$x^{{({:})}}_{:}$'.format(l, j), color=color, **props)
      else:
        ax.text(pos[0], pos[1], '$\delta^{{({:})}}_{:}$'.format(l, j), color=color, **props)
    if l > 0:
      for j in range(1, d_homo[l]):
        for i in range(d_homo[l-1]):
          if (l, j, i) in w_highlights:
            color = highlight_color
          else:
            color = weight_colorscale.to_rgba(w[l][j][i])
            color = tuple([c for c in color[:-1]] + [dim_alpha])
          props = {'ec': color, 'fc': color, 'width': 0.025, 'head_width': 0.08}
          pos_a = get_pos(i, l-1)
          pos_b = get_pos(j, l)
          vec = pos_b - pos_a
          pos_a += vec / np.linalg.norm(vec) * 0.6
          pos_b -= vec / np.linalg.norm(vec) * 0.6
          vec -= vec / np.linalg.norm(vec) * 1.25
          if forward:
            ax.arrow(pos_a[0], pos_a[1], vec[0], vec[1], **props)
          else:
            ax.arrow(pos_b[0], pos_b[1], -vec[0], -vec[1], **props)
          if (l, j, i) in w_highlights:
            props = {'fontsize': 18, 'ha': 'center', 'va': 'center'}
            pos = (pos_a+pos_b)/2.0
            ax.text(pos[0], pos[1], '$w^{{({:})}}_{{{:}, {:}}}$'.format(l, i, j), color=(0, 0, 0, 1), **props)
            #ax.annotate('$w^{{({:})}}_{{{:}, {:}}}$'.format(l, i, j),
            #            xy=pos_a+vec/2,
            #            xytext=pos_a+vec/2+np.array([0, np.linalg.norm(vec)/2]),
            #            arrowprops=dict(width=0.025, ec='k', fc='k', shrink=0.1),
            #            fontsize=24,
            #            ha='center')
    return fig, ax

# d = [2, 1] # d_0 = 2, d_1 = 1
# d_homo, L, w = initialize_weights(d, P_w=lambda: 1)
# draw_network(d, d_homo, L, w)

# draw_network(d, d_homo, L, w, highlights=['w_1_0,1', 'x_0_0', 'x_1_1'])

# d = [2, 2, 1]
# d_homo, L, w = initialize_weights(d, P_w=lambda: 1)
# draw_network(d, d_homo, L, w)

# d = [2, 7, 4, 9, 6, 8, 1]
# d_homo, L, w = initialize_weights(d)
# draw_network(d, d_homo, L, w)

#Step-by-Step Example

extents = (-3, 3, -3, 3)
# -3 <= x_1 <= 3
# -3 <= x_2 <= 3

# d_f = [2, 2, 1]
# d_homo_f, L_f, w_f = initialize_weights(d_f)
# draw_network(d_f, d_homo_f, L_f, w_f)

theta = np.tanh

# x_n = np.array([1] + [np.random.normal() for i in range(d_f[0])])
# x_n

# x = []

# l = 0
# x.append(x_n)
# draw_network(d_f, d_homo_f, L_f, w_f, highlights=['x_0_0', 'x_0_1', 'x_0_2'])
# print(x)

# l += 1
# x_l = np.ones((d_homo_f[l],))
# print(l)
# print(x_l)

# print(w_f)

# for j in range(1, d_homo_f[l]):
#   print('j = {:}'.format(j))
#   w_l_j = w_f[l][j]
#   print(w_l_j)
#   s_l_j = w_l_j.dot(x[l-1])
#   print(s_l_j)
#   x_l[j] = theta(s_l_j)
#   print(x_l)
#   print('')

# l += 1
# x_l = np.ones((d_homo_f[l],))
# print(l)
# print(x_l)

# for j in range(1, d_homo_f[l]):
#   print('j = {:}'.format(j))
#   w_l_j = w_f[l][j]
#   print(w_l_j)
#   s_l_j = w_l_j.dot(x[l-1])
#   print(s_l_j)
#   x_l[j] = theta(s_l_j)
#   print(x_l)
#   print('')

def forward_propagation_all(x_0, w, theta=np.tanh):
  d, d_homo, L = infer_network_properties_from_w(w)
  x = [np.array(x_0)]
  for l in range(1, L+1):
    x_l = np.ones((d_homo[l],))
    for j, w_j in enumerate(w[l][1:], 1):
      x_l[j] = theta(w_j.dot(x[l-1]))
    x.append(x_l)
  return x

def forward_propagation(x_0, w, theta=np.tanh):
  return forward_propagation_all(x_0, w, theta=theta)[-1][1]

def back_propagation(x, y, w, max_iterations=10000, eta=1.0):
  d, d_homo, L = infer_network_properties_from_w(w)
  
  errors = []
  for iteration in range(max_iterations):
    
    # choose random sample from data set
    n = random.randrange(len(y))
    x_n = x[n]
    y_n = y[n]
    
    # propagate forward to get node outputs
    x = forward_propagation_all(x_n, w)
    x_L = x[L][1]
    
    errors.append((y_n - x_L)**2)
    
    # calculate the delta for the last node
    delta_L = 2 * ( 1 - x_L**2 ) * ( x_L - y_n )
    
    # prepare arrays to hold deltas for each node
    delta = [None] * (L+1)
    delta[L] = np.array([delta_L])
    
    # going from the top down
    for l in reversed(range(1, L+1)):
      
      # walk through each node in this layer
      for j in range(d_homo[l]):
        
        # calculate the derivative using this delta and the lower layer's output
        delta_w_l = eta * np.outer(delta[l], x[l-1])
        
        # update this layer's weights using gradient descent
        w[l][j] -= delta_w_l[j]
      
      # prepare array to hold deltas for the lower layer
      delta_lower = np.zeros((d_homo[l-1],))
      
      # walk through nodes on the lower layer
      for i in range(d_homo[l-1]):
        
        # calculate deltas for each node in the lower layer
        delta_lower[i] = ( 1 - x_l[l-1][i] ** 2 ) * sum( [delta[l][j]*delta[l][j][i] for j in range(d_homo[l])] )
      
      # save the deltas
      delta[l-1] = delta_lower

## Complex Example
extents = (-3, 3, -3, 3)

d_f = [2, 200, 200, 1]
d_f_homo, L_f, w_f = nc.initialize_weights(d_f)

f_x = lambda x: nc.forward_propagation(x, w_f)
f = lambda x: np.apply_along_axis(f_x, 1, x)

x_1, x_2, x_grid, z_grid = nc.generate_fill_data(s=200, extents=extents)
f_grid = nc.apply_to_fill(z_grid, f)
f_fig = nc.create_data_plot(extents=extents)
nc.plot_model(f_fig, x_1, x_2, f_grid)

P_x = lambda: np.array( [1.0] + [np.random.normal() for i in range(d_f[0])] )

N = 300
x = np.array([P_x() for i in range(N)])
y = np.sign(theta(x))

D_fig = nc.create_data_plot(extents=extents)
nc.plot_data_set(D_fig, x, y)

x_1, x_2, x_grid, z_grid = nc.generate_fill_data(s=200, extents=extents)

hidden_breadth = 20
num_of_hidden_layers = 5
layer_breadths = [d_f] + [hidden_breadth+1] * num_of_hidden_layers + [2]
h_layers = [[]]
for j, layer_breadth in enumerate(layer_breadths[1:], 1):
  h_layers.append([np.array([np.random.uniform(-1, 1) for c in range(layer_breadths[j])]) for i in range(layer_breadth-1)])

h_x = lambda x: nc.forward_propagation(x, h_layers)
h = lambda x: np.apply_along_axis(h_x, 1, x)
h_grid = nc.apply_to_fill(z_grid, h)
h_fig = nc.create_data_plot(extents=extents)
nc.plot_model(h_fig, x_1, x_2, h_grid)
nc.plot_data_set(h_fig, x, y)

nc.back_propagation(x, y, h_layers, eta=0.01, max_iterations=50000)

h_x = lambda x: nc.forward_propagation(x, h_layers)
h = lambda x: np.apply_along_axis(h_x, 1, x)
h_grid = nc.apply_to_fill(z_grid, h)

h_x = lambda x: nc.forward_propagation(x, h_layers)
h = lambda x: np.apply_along_axis(h_x, 1, x)
h_grid = nc.apply_to_fill(z_grid, h)