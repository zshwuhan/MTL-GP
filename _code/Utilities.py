import math
import numpy as np

from numpy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b

def compute_cov_matrix(cov_function, hyperparameters, X):
    """

    :param cov_function:
    :param X: Design matrix
    :return:
    """
    n, D = num_cols(X), num_rows(X)
    K = np.matrix(np.zeros((n, n)))
    for i in range(n):
        for j in range(i+1):
            K[i, j] = cov_function(hyperparameters, X[:,i], X[:,j])
            K[j, i] = K[i, j]
    return K

def compute_pder_matrix(pder_function, hyperparameters, coord, X):
    n, D = num_cols(X), num_rows(X)
    dK = np.matrix(np.zeros((n, n)))
    for i in range(n):
        for j in range(i+1):
            dK[i, j] = pder_function(hyperparameters, coord, X[:,i], X[:,j])
            dK[j, i] = dK[i, j]
    return dK

def num_rows(m):
    """

    :param m: A matrix
    :return:
    """
    return m.shape[0]

def num_cols(m):
    """

    :param m: A matrix
    :return:
    """
    return m.shape[1]

def backslash(m, v):
    """

    :param m:
    :param v:
    :return:
    """
    # TODO:
    # Comprobar que la matrix m es cuadrada
    # Comprobar que v es un vector columna
    # de tamanio n
    return inv(m)*v

def hadamard_prod(m1, m2):
    # TODO:
    # Comprobar que son matrices del
    # mismo tamanio
    return np.matrix(np.array(m1)*np.array(m2))

def trace_of_prod(m1, m2):
    # TODO:
    # Comprobar que son matrices del
    # mismo tamanio, sym, def pos., etc.
    return hadamard_prod(m1, m2).sum()

def compute_k_star(cov_function, hyperparameters, X, x):
    n = num_cols(X)
    m = num_cols(x)
    return np.concatenate(
        [
            np.matrix([cov_function(hyperparameters, X[:,i], x[:,j]) for i in range(n)]).T for j in range(m)
        ], axis=1
    )
    # return np.matrix([cov_function(hyperparameters, X[:,i], x) for i in range(n)]).T

def dot(x, z):
    result = np.dot(x, z)
    if type(result) is np.matrix:
        return result[0, 0]
    return result

def hyperparameters_SMK(P, Q, hyperparameters):
    w = hyperparameters[0:Q]
    mu = [[]]*Q
    v = [[]]*Q
    for q in range(Q):
        mu[q] = hyperparameters[Q+q*P:Q+(q+1)*P]
    for q in range(Q):
        v[q] = hyperparameters[Q*(1+P)+q*P:Q*(1+P)+(q+1)*P]
    return w, mu, v

def normalize(M, means=None, stds=None):
    if type(M) is not np.matrix:
        print 'error'
        exit(0)
    if means is None or stds is None:
        means, stds = [], []
        for i in range(num_rows(M)):
            v = M[i,:]
            mean, std = np.mean(v), np.std(v)
            means.append(mean); stds.append(std);
            M[i,:] = (v - mean)/std
    elif means is not None and stds is not None:
        for i in range(num_rows(M)):
            v = M[i,:]
            mean, std = means[i], stds[i]
            M[i,:] = (v - mean)/std
    else:
        print 'ERRORCIN'
        exit(0)
    return M, means, stds

def center(M, means=None):
    if type(M) is not np.matrix:
        print 'error'
        exit(0)
    if means is None:
        means = []
        for i in range(num_cols(M)):
            v = M[:,i]
            mean = np.mean(v)
            means.append(mean)
            M[:,i] = v - mean
    else:
        for i in range(num_cols(M)):
            v = M[:,i]
            mean = means[i]
            M[:,i] = v - mean
    return M, means

def mean_squared_error(predictions, targets):
    tao = predictions - targets
    return dot(tao.T, tao)/num_rows(tao)

'''
def compute_k_star_star(cov_function, hyperparameters, x, z):
    n = num_rows(X)
    return np.matrix([cov_function(hyperparameters, X[:,i], x) for i in range(n)]).T

def maximize(f, fprime=None):
    def g(x): return 0 - f(x)

    if fprime is not None:
        def gprime(x): return 0 - fprime(x)
        return
'''

if __name__ == '__main__':
    print 'hola'

    def cov_function(x, z):
        return np.dot(x.T, z)

    X = np.matrix(range(1,13)).reshape((3, 4))
    print X
    print compute_cov_matrix(cov_function, X)

    print 'adios'