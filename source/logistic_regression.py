import numpy as np
import scipy.optimize

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def gradient_descent(x, y, w_h=None, eta=1.0, max_iterations=100, epsilon=0.001):
  if w_h == None:
      w_h = np.array([0.0 for i in range(x.shape[1])])
  
  # save a history of the weight vectors into an array
  w_h_i = [np.copy(w_h)]
  
  for i in range(max_iterations):
    subset_indices = range(x.shape[0])
    
    c = - y[subset_indices] / ( 1.0 + np.exp(y[subset_indices] * w_h.dot(x[subset_indices].T)) )
    
    b = np.tile(c, (x.shape[1], 1)).T

    a = b * x[subset_indices]
    
    grad_E_in = np.mean(a, axis=0)

    w_h -= eta * grad_E_in
    w_h_i.append(np.copy(w_h))
    if np.linalg.norm(grad_E_in) <= np.linalg.norm(w_h) * epsilon:
      break
  
  return np.array(w_h_i)

w_h_i = gradient_descent(x, y, eta=4.0)
w_h = w_h_i[-1]
print w_h
print('Number of iterations: {:}'.format(w_h_i.shape[0]))
h = lambda x: logistic(w_h.dot(x.T))