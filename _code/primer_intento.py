import math
import numpy as np

from gp import GaussianProcess, DotKernel
from gp import SMKernel

N_ROWS = 5
N_COLS = 100

X = np.matrix([(0.1*(1+i))**(1+j) for j in range(N_ROWS) for i in range(N_COLS)]).reshape((N_ROWS,N_COLS))
y1 = np.matrix([np.cos(0.1*(1+i)) for i in range(N_COLS)]).T
y2 = np.matrix([np.sin(0.1*(1+i)) for i in range(N_COLS)]).T
# my_GP = GaussianProcess(SMKernel(N_ROWS, 1), X)
my_GP = GaussianProcess(DotKernel(), X)
my_GP.add_task(y1)
my_GP.add_task(y2)

x_star = np.matrix([3**(1+j) for j in range(N_ROWS)]).T
my_GP.gpr_make_prediction(my_GP.cov_function.INITIAL_GUESS, 0, x_star)

print my_GP.gpr_optimize(0, x_star)
print my_GP.mean, my_GP.variance
print my_GP.mlog_ML

my_GP.gpr_make_prediction(my_GP.cov_function.INITIAL_GUESS, 0, x_star)
print my_GP.mean, my_GP.variance

