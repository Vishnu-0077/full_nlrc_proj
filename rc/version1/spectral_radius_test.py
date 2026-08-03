import numpy as np
from scipy import linalg
n = 10
W = np.random.rand(n,n) - 0.5

eigen_values = np.linalg.eigvals(W)
spectral_radius = np.max(np.abs(eigen_values))
target_radius = 0.95

W_new = W * target_radius / spectral_radius
print(W_new)