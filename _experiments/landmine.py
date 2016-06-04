import scipy.io
import numpy as np
import os
from numpy import int16
from _code.gp import GaussianProcess, SMKernel, DotKernel, SEKernel
from _code.gp import LogisticFunction
from _code.Utilities import center
from _code.Utilities import mean_squared_error as mse
from _code.Utilities import num_rows, num_cols

NUM_TRAIN_CASES = 100
NUM_TEST_CASES = 10
NUM_TOTAL_CASES = NUM_TRAIN_CASES+NUM_TEST_CASES

inputs_data = scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "LandmineData.mat")['feature']
targets_data = scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "LandmineData.mat")['label']

def  get_data_task(task):
    X = np.matrix(inputs_data[0, task]).T
    y = np.matrix(targets_data[0, task], dtype=int16)
    y[y==0] = -1
    return X, y

X, y = get_data_task(1)

# Training set
Xtrain = X[:,:NUM_TRAIN_CASES]
Ytrain = y[:NUM_TRAIN_CASES,:]

# Test set
Xtest = X[:,NUM_TRAIN_CASES:NUM_TOTAL_CASES]
Ytest = y[NUM_TRAIN_CASES:NUM_TOTAL_CASES,:]

my_GP = GaussianProcess(SEKernel(), sigmoid_function=LogisticFunction())
my_GP.add_task(Xtrain, Ytrain)

tasks = [0]
my_GP.hyperparameters = my_GP.gpc_optimize(tasks)[0]
print my_GP.hyperparameters

y_star = my_GP.gpc_make_prediction([0], Xtest)
print y_star
for i in range(len(y_star)):
    print Ytest[i,0], y_star[i]
