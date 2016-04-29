import math
import numpy as np

from Utilities import compute_cov_matrix, compute_pder_matrix
from Utilities import backslash, hadamard_prod
from Utilities import num_rows, num_cols
from Utilities import compute_k_star
from Utilities import trace_of_prod
from Utilities import dot
from Utilities import hyperparameters_SMK
from numpy.linalg import cholesky, inv
from scipy.optimize import fmin_l_bfgs_b as l_bfgs
from scipy.optimize import fmin_cg
eps = 0.001

log, exp, cos, sin = np.log, np.exp, np.cos, np.sin

TWOPI = 2*math.pi
MTWOPISQ = -TWOPI*math.pi

class CovFunction:
    def __init__(self, cov_function):
        self.cov_function = cov_function

    def cov_matrix(self, hyperparameters, X):
        return compute_cov_matrix(self.cov_function, hyperparameters, X)

    def pder_matrix(self, hyperparameters, i, X):
        return compute_pder_matrix(self.compute_pder, hyperparameters, i, X)
        '''
        h1 = hyperparameters
        h2 = hyperparameters
        h1[coord] += eps
        h2[coord] -= eps
        num = self.cov_matrix(h1, X) - self.cov_matrix(h2, X)
        denom = 2*eps
        return num/denom
        '''

class DotKernel(CovFunction):
    def __init__(self):
        self.INITIAL_GUESS = [27.]
        def cov_function(hyperparameters, x, z):
            l = hyperparameters[0]
            return log(l)*dot(x.T, z)
        self.cov_function = cov_function
        def compute_pder(hyperparameters, i, x, z):
            l = hyperparameters[0]
            return dot(x.T, z)/l
        self.compute_pder = compute_pder

class SMKernel(CovFunction):
    def __init__(self, P, Q):
        self.P, self.Q = P, Q
        self.INITIAL_GUESS = [2.3]*(1+2*P)*Q
        def cov_function(hyperparameters, x, z):
            # TODO:
            # Comprobar que el numero de hip. es correcto

            # TODO:
            # Una posible mejora: hacer que todos los hypp
            # puedan tomar cualquier valor real
            # (p.ej. si un hyp puede tomar cualquier valor
            # pos considerar el logaritmo de dicho hypp).

            P, Q = self.P, self.Q
            w = hyperparameters[0:Q]
            mu = [[]]*Q
            v = [[]]*Q
            for q in range(Q):
                mu[q] = hyperparameters[Q+q*P:Q+(q+1)*P]
            for q in range(Q):
                v[q] = hyperparameters[Q*(1+P)+q*P:Q*(1+P)+(q+1)*P]
            tao = x - z
            sum = 0
            for q in range(Q):
                prod = w[q]
                prod *= cos(TWOPI*dot(tao.T, mu[q]))
                arg_exp = 0
                for p in range(P):
                    arg_exp += ((tao[p,0]**2)*v[q][p])
                arg_exp *= (MTWOPISQ)
                prod *= exp(arg_exp)
                sum += prod
            # TODO:
            # Recomprobar que la formulita
            # esta bien copiada.
            return sum

        self.cov_function = cov_function

        def compute_pder(hyperparameters, i, x, z):
            w, mu, v = hyperparameters_SMK(P, Q, hyperparameters)
            tao = x - z
            if i < Q:
                # El parametro es una w_q
                # TODO: Copypasteado (se podria hacer mejor)
                q = i
                prod = cos(TWOPI*dot(tao.T, mu[q]))
                arg_exp = 0
                for p in range(P):
                    arg_exp += ((tao[p,0]**2)*v[q][p])
                arg_exp *= (MTWOPISQ)
                prod *= exp(arg_exp)
                return prod
            elif i < Q*(P+1):
                q = (i-Q)/P
                p = (i-Q)%P
                prod = -TWOPI*(tao[p, 0]**2)*w[q]
                arg_exp = 0
                for j in range(P):
                    arg_exp += ((tao[j,0]**2)*v[q][p])
                arg_exp *= MTWOPISQ
                prod *= exp(arg_exp)
                prod *= sin(TWOPI*dot(tao.T, mu[q]))
                return prod
            elif i < Q*(2*P+1):
                q = (i-Q*(P+1))/P
                p = (i-Q)%P
                prod = MTWOPISQ*tao[p, 0]*w[q]
                prod *= cos(TWOPI*dot(tao.T, mu[q]))
                arg_exp = 0
                for j in range(P):
                    arg_exp += ((tao[j,0]**2)*v[q][p])
                arg_exp *= MTWOPISQ
                prod *= exp(arg_exp)
                return prod
            else:
                # TODO: Implementar esto
                print 'Sin implementar (oops)'
                return None
        self.compute_pder = compute_pder

class SigmoidFunction:
    pass

class LogisticFunction(SigmoidFunction):
    def log_likelihood(self, y, f):
        n = num_rows(y)
        # TODO:
        # Comprobar que y, f son del
        # mismo tamanio.
        return [-log(1 + exp(-y[i,0]*f[i,0])) for i in range(n)]

    def der_log_likelihood(self, y, f):
        n = num_rows(y)
        # TODO:
        # Comprobar que y, f son del
        # mismo tamanio.
        # Tambien tratar y, f como matr.
        pi = exp(-self.log_likelihood(y, f))
        t = 0.5*(y + 1)
        return t - pi

    def der2_log_likelihood(self, y, f):
        pi = exp(-self.log_likelihood(y, f))
        return hadamard_prod(pi, 1-pi)

class GaussianProcess:
    def __init__(self, cov_function, X):
        self.cov_function = cov_function
        self.X = X
        self.n, self.D = num_cols(X), num_rows(X)
        self.num_tasks = 0
        self.Y = []

        self.hyperparameters =  cov_function.INITIAL_GUESS
        self.K = cov_function.cov_matrix(self.hyperparameters, X)

        # TODO:
        # Matriz de covarianza

    def add_task(self, y):
        self.Y.append(y)
        self.num_tasks += 1

    def gpr_make_prediction(self, hyperparameters, task, x):
        sigma_n = eps
        I = np.matrix(np.eye(self.n))
        y = self.Y[task]
        self.L = cholesky(self.K + sigma_n*I)
        L = self.L
        alpha = backslash(L.T, backslash(L, y))
        k_star = compute_k_star(self.cov_function.cov_function, hyperparameters, self.X, x)
        self.mean = np.dot(k_star.T, alpha)
        v = backslash(L, k_star)
        k_star_star = self.cov_function.cov_function(hyperparameters, x, x)
        self.variance = k_star_star - np.dot(v.T, v)
        self.mlog_ML = 0.5*(np.dot(y.T, alpha)[0, 0] + sum(log(np.diag(L))))
        return self.mlog_ML

    def der_mlogML(self, task, i):
        alpha = backslash(self.L.T, backslash(self.L, self.Y[task]))
        L_inv = inv(self.L)
        return -0.5*trace_of_prod(alpha*alpha.T - L_inv.T*L_inv, self.cov_function.pder_matrix(self.hyperparameters, i, self.X))
        # return -0.5*trace_of_prod(alpha*alpha.T - L_inv.T*L_inv, self.cov_function.pder_matrix())
        # return -0.5*trace_of_prod(alpha*alpha.T - )

    def gradient_mlogML(self, task):
        return np.array([self.der_mlogML(task, i) for i in range(len(self.hyperparameters))])

    def gpr_optimize(self, task, x):
        def my_prediction(hyperparameters):
            lml = self.gpr_make_prediction(hyperparameters, task, x)
            print lml
            return lml
        def my_grad(hyperparameters):
            grad = self.gradient_mlogML(task)
            print grad
            return grad
        # return fmin_cg(my_prediction, self.cov_function.INITIAL_GUESS)
        return l_bfgs(my_prediction, self.cov_function.INITIAL_GUESS, fprime=my_grad)

    def gpc_find_mode(self, task):
        f = 0
        I = np.matrix(np.eye(self.n))
        # TODO:
        # Cambiar la condicion
        # (logicamente)
        while True:
            W = None #TODO: Ciertamente definir W
            K = self.K
            sqrt_W = np.sqrt(W)
            L = cholesky(I + sqrt_W*K*sqrt_W)
            b = W*f + 0 # TODO: Completar
            a = b - sqrt_W*backslash(L.T, backslash(L, sqrt_W*K*b))
            f = K*a
            # TODO:
            # Comprobar aqui la convergencia?
        # TODO: Tambien se devuelve el logML
        return f

    def gpc_make_prediction(self):
        I = np.matrix(np.eye(self.n))
        W = None #TODO: Ciertamente definir W
        K = self.K
        sqrt_W = np.sqrt(W)
        L = cholesky(I + sqrt_W*K*sqrt_W)
        k_star = None # TODO: Definir esto como matriz fila
        f_mean = None # TODO: Definir bien esto
        v = backslash(L, sqrt_W*k_star)
        k_star_star = None # TODO: Definir bien esto
        f_var = k_star_star - np.dot(v.T, v)
        pi_star = None # TODO: Acabamos con esto. Bieeeeen

    def gpc_optimize(self):
        # TODO: Implementar esta mierda
        pass
