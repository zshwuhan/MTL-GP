import scipy.io
import numpy as np
import os
from _code.gp import GaussianProcess, SMKernel, DotKernel, SEKernel
from _code.gp import LogisticFunction
from _code.Utilities import center
from _code.Utilities import mean_squared_error as mse
from _code.Utilities import num_rows, num_cols

CASE1 = 1

NUM_TRAIN_CASES = 40
NUM_TEST_CASES = 10

# Training set
Xtrain = scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "usps_resampled.mat")["train_patterns"]
Ytrain = scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "usps_resampled.mat")["train_labels"]
Xtrain = np.concatenate(
    (Xtrain[:,Ytrain[CASE1,:]>0][:,:NUM_TRAIN_CASES], Xtrain[:,Ytrain[CASE1,:]<0][:,:NUM_TRAIN_CASES]), axis=1
)
Ytrain = np.concatenate(
    (Ytrain[:,Ytrain[CASE1,:]>0][CASE1,:NUM_TRAIN_CASES], Ytrain[:,Ytrain[CASE1,:]<0][CASE1,:NUM_TRAIN_CASES]), axis=1
)
Xtrain = np.matrix(Xtrain)
Ytrain = np.matrix(Ytrain).T

# Test set
Xtest = scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "usps_resampled.mat")["test_patterns"]
Ytest = scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "usps_resampled.mat")["test_labels"]
Xtest = np.concatenate(
    (Xtest[:,Ytest[CASE1,:]>0][:,:NUM_TEST_CASES], Xtest[:,Ytest[CASE1,:]<0][:,:NUM_TEST_CASES]), axis=1
)
Ytest = np.concatenate(
    (Ytest[:,Ytest[CASE1,:]>0][CASE1,:NUM_TEST_CASES], Ytest[:,Ytest[CASE1,:]<0][CASE1,:NUM_TEST_CASES]), axis=1
)
Xtest = np.matrix(Xtest)
Ytest = np.matrix(Ytest).T


my_GP = GaussianProcess(SEKernel(), LogisticFunction())
my_GP.add_task(Xtrain, Ytrain)

tasks = [0]
my_GP.hyperparameters = my_GP.gpc_optimize(tasks)[0]
print my_GP.hyperparameters

f_star = my_GP.gpc_make_prediction([0], Xtest)
for i in range(len(f_star)):
    print Ytest[i,0], 1 if f_star[i]>0.5 else -1

CASE2 = 3

# Training set
Ytrain2 = scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "usps_resampled.mat")["train_labels"]
Ytrain2 = np.concatenate(
    (Ytrain2[:,Ytrain2[CASE1,:]>0][CASE2, :NUM_TRAIN_CASES], Ytrain2[:, Ytrain2[CASE1, :] < 0][CASE2, :NUM_TRAIN_CASES]), axis=1
)
Ytrain2 = np.matrix(Ytrain2).T

my_GP = GaussianProcess(SEKernel(), LogisticFunction())
my_GP.add_task(Xtrain, Ytrain)
my_GP.add_task(Xtrain, Ytrain2)

tasks = [0,1]
my_GP.hyperparameters = my_GP.gpc_optimize(tasks)[0]
print my_GP.hyperparameters

f_star = my_GP.gpc_make_prediction([0], Xtest)
for i in range(len(f_star)):
    print Ytest[i,0], 1 if f_star[i]>0.5 else -1

CASE3 = 7

# Training set
Ytrain3 = scipy.io.loadmat(".." + os.sep + "_data" + os.sep + "usps_resampled.mat")["train_labels"]
Ytrain3 = np.concatenate(
    (Ytrain3[:,Ytrain3[CASE1,:]>0][CASE3, :NUM_TRAIN_CASES], Ytrain3[:, Ytrain3[CASE1, :] < 0][CASE3, :NUM_TRAIN_CASES]), axis=1
)
Ytrain3 = np.matrix(Ytrain3).T

my_GP = GaussianProcess(SEKernel(), LogisticFunction())
my_GP.add_task(Xtrain, Ytrain)
my_GP.add_task(Xtrain, Ytrain2)
my_GP.add_task(Xtrain, Ytrain3)

tasks = [0,1,2]
my_GP.hyperparameters = my_GP.gpc_optimize(tasks)[0]
print my_GP.hyperparameters

f_star = my_GP.gpc_make_prediction([0], Xtest)
for i in range(len(f_star)):
    print Ytest[i,0], 1 if f_star[i]>0.5 else -1