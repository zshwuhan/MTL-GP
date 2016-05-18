# Get the dataset here: http://www.gaussianprocess.org/gpml/data/
import scipy.io
import numpy as np
from gp import GaussianProcess, SMKernel, DotKernel

# Load training set
train = np.matrix(scipy.io.loadmat("/home/victor/Documents/MTL-GP/_data/sarcos_inv.mat")["sarcos_inv"])
# Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations)
Xtrain = train[:50, :21].T
# Outputs (7 joint torques)
Ytrain = train[:50, 21]

# Load test set
test = np.matrix(scipy.io.loadmat("/home/victor/Documents/MTL-GP/_data/sarcos_inv_test.mat")["sarcos_inv_test"])
Xtest = test[:100, :21].T
Ytest = test[:100, 21]

my_GP = GaussianProcess(SMKernel(Xtrain.shape[0], 1), Xtrain)
# my_GP = GaussianProcess(DotKernel(), Xtrain)
my_GP.add_task(Ytrain)
my_GP.gpr_normalize()
x_star = Xtest[:, 0:100]
my_GP.gpr_make_prediction(my_GP.cov_function.INITIAL_GUESS, 0, x_star)

print my_GP.gpr_optimize(0, x_star)
print my_GP.mean, my_GP.variance
print my_GP.mlog_ML

'''
for i in range(10):
    x_star = Xtest[:, i]
    my_GP.gpr_make_prediction(my_GP.cov_function.INITIAL_GUESS, 0, x_star)
    my_GP.gpr_optimize(0, x_star)
    print my_GP.mean, Ytest[i,0]
'''