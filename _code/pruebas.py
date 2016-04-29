from gp import SMKernel
from gp import TWOPI, MTWOPISQ

import numpy as np

# Probar que el kernel espectral funciona

P, Q = 1, 1
mi_kernel = SMKernel(P, Q)
x, z = np.matrix([3]), np.matrix([4])
hyperparameters = [3, 1, 1]
assert mi_kernel.cov_function(hyperparameters, x.T, z) == 3*np.exp(-2*np.pi*np.pi)
assert mi_kernel.compute_pder(hyperparameters, 0, x, z) == np.exp(-2*np.pi*np.pi)
assert mi_kernel.compute_pder(hyperparameters, 1, x, z) < 1e-20 # Es cero
assert mi_kernel.compute_pder(hyperparameters, 2, x, z) == 6*np.pi*np.pi*np.exp(MTWOPISQ)

P, Q = 2, 1
assert mi_kernel.compute_pder(hyperparameters, 0, x, z) ==  np.cos(4*np.pi)*np.exp(-2*np.pi*np.pi*0.1)
