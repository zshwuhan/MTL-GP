import scipy.io
import numpy as np
from gp import GaussianProcess, SMKernel, DotKernel
from gp import LogisticFunction
from numpy.random import normal as dnorm

means, sigma, N = [-6,0,2], 0.8, 10

X = np.matrix(
    np.concatenate(
        [dnorm(i, sigma, N) for i in means]
    )
)

y = np.matrix(
    [-1]*N + [1]*N + [-1]*N
).T

# my_GP = GaussianProcess(DotKernel(), X)
my_GP = GaussianProcess(SMKernel(X.shape[0], 3), X)
my_GP.add_task(y)
my_GP.sigmoid = LogisticFunction()

x_star = np.matrix(3)
f_mode, log_ML = my_GP.gpc_find_mode(0)
print my_GP.gpc_make_prediction(0, f_mode, x_star)


