# Get the dataset here: http://www.gaussianprocess.org/gpml/data/
import scipy.io
import numpy as np
from gp import GaussianProcess, SMKernel, DotKernel, SEKernel
from Utilities import normalize

# Load training set
train = np.matrix(scipy.io.loadmat("/home/victor/Documents/MTL-GP/_data/sarcos_inv.mat")["sarcos_inv"])
# Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations)
Xtrain = train[:100, :21].T
# Outputs (7 joint torques)
Ytrain = normalize(train[:100, 21:23])[0]

# Load test set
test = np.matrix(scipy.io.loadmat("/home/victor/Documents/MTL-GP/_data/sarcos_inv_test.mat")["sarcos_inv_test"])
Xtest = test[:100, :21].T
Ytest = normalize(test[:100, 21:23])[0]

# my_GP = GaussianProcess(SMKernel(Xtrain.shape[0], 3), Xtrain)
# my_GP = GaussianProcess(DotKernel(), Xtrain)
my_GP = GaussianProcess(SEKernel(), Xtrain)
my_GP.add_task(Ytrain[:,0])
my_GP.add_task(Ytrain[:,1])
# Ytrain = my_GP.gpr_normalize()
x_star = Xtest[:, 0:100]
# my_GP.gpr_make_prediction(my_GP.cov_function.INITIAL_GUESS, [0, 1], x_star)

print my_GP.gpr_optimize([0,1], x_star)
# print my_GP.mean.T
# print Ytest.T
print np.mean(abs((my_GP.mean[:,0] - Ytest[:,0])))
print np.mean(abs((my_GP.mean[:,1] - Ytest[:,1])))
print my_GP.mlog_ML

'''
for i in range(10):
    x_star = Xtest[:, i]
    my_GP.gpr_make_prediction(my_GP.cov_function.INITIAL_GUESS, 0, x_star)
    my_GP.gpr_optimize(0, x_star)
    print my_GP.mean, Ytest[i,0]
'''