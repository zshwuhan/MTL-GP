import math
import numpy as np

from gp import GaussianProcess, DotKernel
from gp import SMKernel


N_ROWS = 5
N_COLS = 10

X = np.matrix([(0.1*i)**j for j in range(N_ROWS) for i in range(1,N_COLS+1)]).reshape((N_ROWS,N_COLS))
y1 = np.matrix([(0.2*i) + np.random.normal(0,0.01,1)[0] for i in range(1,N_COLS+1)]).T
my_GP = GaussianProcess(SMKernel(N_ROWS, 1), X)
# my_GP = GaussianProcess(DotKernel(), X)
my_GP.add_task(y1)
x_star = np.matrix([1.5**j for j in range(N_ROWS)]).T
my_GP.gpr_make_prediction(my_GP.cov_function.INITIAL_GUESS, 0, x_star)

print my_GP.gpr_optimize(0, x_star)
print my_GP.mean, my_GP.variance
print my_GP.mlog_ML
