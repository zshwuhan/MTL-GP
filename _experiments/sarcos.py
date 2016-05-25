import scipy.io
import numpy as np
import os
from _code.gp import GaussianProcess, SMKernel, DotKernel, SEKernel
from _code.Utilities import normalize, center
from _code.Utilities import mean_squared_error as mse
from _code.Utilities import num_rows, num_cols

NUM_TRAIN_CASES = 100
NUM_TEST_CASES = 10

# Training set
train = np.matrix(scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "sarcos_inv.mat")["sarcos_inv"])
Xtrain, Xmean, Xstd = normalize(train[:NUM_TRAIN_CASES, :21].T)
Ytrain, Ymean = center(train[:NUM_TRAIN_CASES, 21:])

# Test set
test = np.matrix(scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "sarcos_inv_test.mat")["sarcos_inv_test"])
Xtest = normalize(test[:NUM_TEST_CASES, :21].T, Xmean, Xstd)[0]
Ytest = center(test[:NUM_TEST_CASES, 21:], Ymean)[0]

my_GP = GaussianProcess(SEKernel(), Xtrain)
# for i in range(num_cols(Ytest)): my_GP.add_task(Ytrain[:,i])

print '-------------------------------'
print 'UNITAREA'
print '-------------------------------'
for i in range(num_cols(Ytest)):
    my_GP = GaussianProcess(SEKernel(), Xtrain)
    # my_GP = GaussianProcess(SMKernel(Xtrain.shape[0], 3), Xtrain)
    my_GP.add_task(Ytrain[:,i])
    my_GP.gpr_optimize([0], Xtest)
    print i+1, mse(my_GP.mean, Ytest[:,i])
    # print i+1, np.mean(abs((my_GP.mean - Ytest[:,i])))

print '-------------------------------'
print 'MULTITAREA'
print '-------------------------------'
my_GP = GaussianProcess(SEKernel(), Xtrain)
# my_GP = GaussianProcess(SMKernel(Xtrain.shape[0], 3), Xtrain)
for i in range(num_cols(Ytest)): my_GP.add_task(Ytrain[:,i])
my_GP.gpr_optimize(range(num_cols(Ytest)), Xtest)
for i in range(num_cols(Ytest)):
    print i+1, mse(my_GP.mean[:,i], Ytest[:,i])
    # print i+1, np.mean(abs((my_GP.mean[:,i] - Ytest[:,i])))

