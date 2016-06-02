import scipy.io
import numpy as np
from gp import GaussianProcess, SMKernel, DotKernel, SEKernel
from gp import LogisticFunction
from numpy.random import normal as dnorm

means, sigma, N = [-6,0,2], 0.8, 30

X = np.matrix(
    np.concatenate(
        [dnorm(i, sigma, N) for i in means]
    )
)
# my_GP = GaussianProcess(DotKernel(), X, LogisticFunction())
my_GP = GaussianProcess(SEKernel(), X, LogisticFunction())
# my_GP = GaussianProcess(SMKernel(X.shape[0], 3), X, LogisticFunction())

y = np.matrix(
    [-1]*N + [1]*N + [-1]*N
).T
my_GP.add_task(y)

y2 = []
for i in range(3*N):
    if X[0,i] > 0:
        y2.append(1)
    else:
        y2.append(-1)
y2 = np.matrix(y2).T
my_GP.add_task(y2)

# print [-1+0.1*i for i in range(30)]
# x_star = np.matrix([-1+0.1*i for i in range(50)])
x_star = np.matrix(dnorm(0,sigma,N))
# f_mode, log_ML = my_GP.gpc_find_mode([0, 1], my_GP.cov_function.INITIAL_GUESS)
# print my_GP.gpc_optimize([0])
tasks = [0]
my_GP.hyperparameters = my_GP.gpc_optimize(tasks)[0]
print my_GP.hyperparameters

y_star = my_GP.gpc_make_prediction([0], x_star)
for i in range(len(y_star)):
    print x_star[0, i], y_star[i]


