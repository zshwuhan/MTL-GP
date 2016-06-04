import math
import numpy as np
import scipy.stats
from numpy.linalg.linalg import LinAlgError
from Utilities import compute_cov_matrix, compute_pder_matrix
from Utilities import backslash, hadamard_prod
from Utilities import num_rows, num_cols
from Utilities import compute_k_star
from Utilities import trace_of_prod
from Utilities import dot
from Utilities import hyperparameters_SMK
from numpy.linalg import cholesky, inv, norm
from scipy.optimize import fmin_l_bfgs_b as l_bfgs
from scipy.integrate import quad as integrate

eps = 0.1
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

class SEKernel(CovFunction):
    def __init__(self):
        self.INITIAL_GUESS = [log(2.6), log(7.0)]
        # self.INITIAL_GUESS = [9.1, 9.1]
        # self.INITIAL_GUESS = [0.1, 0.1]
        def cov_function(hyperparameters, x, z):
            sigma, l = hyperparameters[0], hyperparameters[1]
            tao = x - z
            return (sigma**2)*exp(-dot(tao.T, tao)/(2*l*l))
        self.cov_function = cov_function
        def compute_pder(hyperparameters, i, x, z):
            sigma, l = hyperparameters[0], hyperparameters[1]
            tao = x - z
            if i==0:
                return 2*sigma*exp(-dot(tao.T, tao)/(2*l*l))
            elif i==1:
                r_squared = dot(tao.T, tao)
                return (r_squared/l**3)*(sigma**2)*exp(-r_squared/(2*l*l))
            else:
                #TODO: controlar error
                # print 'error'
                return None
        self.compute_pder = compute_pder

class DotKernel(CovFunction):
    def __init__(self):
        self.INITIAL_GUESS = [2.]
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
        self.INITIAL_GUESS = [0.1]*(1+2*P)*Q
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
                # print 'Sin implementar (oops)'
                return None
        self.compute_pder = compute_pder

class SigmoidFunction:
    pass

class LogisticFunction(SigmoidFunction):
    def evaluate(self, z):
        # if type(z) == tuple:
        #     z = np.array(z)
        return 1./(1 + exp(-z))

    def log_likelihood(self, y, f):
        n = num_rows(y)
        # TODO:
        # Comprobar que y, f son del
        # mismo tamanio.
        # return np.matrix([-log(1 + exp(-y[i,0]*f[i,0])) for i in range(n)]).T
        sum = 0
        for i in range(n):
            sum -= log(1 + exp(-y[i,0]*f[i, 0]))
        return sum

    def gradient_log_likelihood(self, y, f):
        n = num_rows(y)
        # TODO:
        # Comprobar que y, f son del
        # mismo tamanio.
        # Tambien tratar y, f como matr.
        n = num_rows(y)
        p = np.matrix([1/(1+exp(-f[i,0])) for i in range(n)]).T
        t = 0.5*(y + 1)
        return t - p

    def hessian_log_likelihood(self, y, f):
        n = num_rows(y)
        p = np.matrix([1./(1+exp(-f[i,0])) for i in range(n)]).T
        return np.matrix(np.diag([-p[i,0]*(1-p[i,0]) for i in range(n)]))

    def der3_log_likelihood(self, y, f):
        n = num_rows(y)
        p = np.matrix([1./(1+exp(-f[i,0])) for i in range(n)]).T
        return np.matrix([p[i,0]*(1-p[i,0])*(2*p[i,0]-1) for i in range(n)]).T

class GaussianProcess:
    def __init__(self, cov_function, sigmoid_function=None):
        self.cov_function = cov_function
        # self.X = X
        # self.n, self.D = num_cols(X), num_rows(X)
        self.num_tasks = 0
        self.Y = None

        self.hyperparameters =  cov_function.INITIAL_GUESS
        # self.K = cov_function.cov_matrix(self.hyperparameters, X)
        self.sigmoid = sigmoid_function
        # TODO:
        # Matriz de covarianza

    def add_task(self, X, y):
        if self.num_tasks == 0:
            self.X = []
            self.X.append(X)
            self.Y = y
        else:
            self.X.append(X)
            self.Y = np.concatenate((self.Y, y), axis=1)
        self.num_tasks += 1

    def gpr_normalize(self):
        for task in range(len(self.Y)):
            y = self.Y[task]
            mean, std = np.mean(y), np.std(y)
            self.Y[task] = (y - mean)/std
        return self.Y

    def gpr_make_prediction(self, hyperparameters, tasks, x):
        sigma_n = eps
        I = np.matrix(np.eye(self.n))
        y = self.Y[:, tasks]
        # self.K = self.cov_function.cov_matrix(hyperparameters, self.X)
        K = self.cov_function.cov_matrix(hyperparameters, self.X)
        self.L = cholesky(K + sigma_n*I)
        del K
        L = self.L
        alpha = backslash(L.T, backslash(L, y))
        k_star = compute_k_star(self.cov_function.cov_function, hyperparameters, self.X, x)
        self.mean = np.dot(k_star.T, alpha)
        v = backslash(L, k_star)
        # k_star_star = self.cov_function.cov_function(hyperparameters, x, x)
        k_star_star = compute_k_star(self.cov_function.cov_function, hyperparameters, x, x)
        self.variance = k_star_star - np.dot(v.T, v)
        self.mlog_ML = 0
        for task in tasks:
            self.mlog_ML += 0.5*(np.dot(y[:, task].T, alpha[:,task]) + sum(log(np.diag(L))))
        return self.mlog_ML

    def der_mlogML(self, hyperparameters, tasks, i):
        der = 0
        L_inv = inv(self.L)
        for task in tasks:
            alpha = backslash(self.L.T, backslash(self.L, self.Y[:, task]))
            der -= 0.5*trace_of_prod(alpha*alpha.T - L_inv.T*L_inv, self.cov_function.pder_matrix(self.hyperparameters, i, self.X))
        return der

    def gradient_mlogML(self, hyperparameters, task):
        return np.array([self.der_mlogML(hyperparameters, task, i) for i in range(len(self.hyperparameters))])

    def gpr_optimize(self, tasks, x):
        def my_prediction(hyperparameters):
            lml = self.gpr_make_prediction(hyperparameters, tasks, x)
            # print lml
            return lml
        def my_grad(hyperparameters):
            grad = self.gradient_mlogML(hyperparameters, tasks)
            # print grad
            return grad
        # return fmin_cg(my_prediction, self.cov_function.INITIAL_GUESS)
        self.hyperparameters = l_bfgs(my_prediction, self.cov_function.INITIAL_GUESS, fprime=my_grad, maxfun=10)
        return self.hyperparameters

    def gpc_find_mode(self, tasks, hyperparameters):
        try:
            num_tasks = len(tasks)
            # TODO: modificar esta linea (puede fallar)
            n = num_cols(self.X[0])
            f = np.matrix([0]*n*num_tasks).reshape((n, num_tasks))
            self.mlog_ML = 0
            K = [[]]*num_tasks
            for task in range(num_tasks):
                I = np.matrix(np.eye(n))
                y = self.Y[:,task]
                while True:
                    f_old = np.copy(f[:,task])
                    W = -self.sigmoid.hessian_log_likelihood(y, f[:,task])
                    K[task] = self.cov_function.cov_matrix(hyperparameters, self.X[task])
                    sqrt_W = np.sqrt(W)
                    L = cholesky(I + sqrt_W*K[task]*sqrt_W)
                    b = W*f[:,task] + self.sigmoid.gradient_log_likelihood(y, f[:,task])
                    a = b - sqrt_W*backslash(L.T, backslash(L, sqrt_W*K[task]*b))
                    f[:,task] = K[task]*a
                    if norm(f[:,task]-f_old) < 0.01:
                        break
                self.mlog_ML += (0.5*dot(a.T, f[:,task]) - self.sigmoid.log_likelihood(y, f[:,task]) + sum(log(np.diag(L))))
            self.f, self.a, self.sqrt_W, self.K, self.I = f, a, sqrt_W, K, I
            return f, self.mlog_ML
        except LinAlgError:
            return None, np.inf


    def gpc_gradient_mlogML(self, hyperparameters, tasks):
        f, a, sqrt_W, K, I = self.f, self.a, self.sqrt_W, self.K, self.I
        num_tasks = len(tasks)
        num_hyper = len(hyperparameters)
        gradient = np.array([0.0]*num_hyper)
        for task in range(num_tasks):
            y = self.Y[:,task]
            self.L = cholesky(I + sqrt_W*K[task]*sqrt_W)
            L = self.L
            R = sqrt_W*backslash(L.T, backslash(L, sqrt_W))
            C = backslash(L, sqrt_W*K[task])
            s2 = np.matrix(np.diag(np.diag(K[task]) - np.diag(C.T*C)))*self.sigmoid.der3_log_likelihood(y, f[:,task])
            for j in range(num_hyper):
                C = self.cov_function.pder_matrix(hyperparameters, j, self.X[task])
                s1 = 0.5*(a.T*C*a - trace_of_prod(R, C))[0, 0]
                b = C*self.sigmoid.gradient_log_likelihood(y, f[:,task])
                s3 = b - K[task]*R*b
                gradient[j] += (-s1 -dot(s2.T,s3))
        return gradient

    def gpc_make_prediction(self, tasks, x_star):
        task = tasks[0]
        y, f = self.Y[:,task], self.f
        num_tasks = len(tasks)
        # TODO: Revisar esta linea(puede fallar):
        I = np.matrix(np.eye(num_cols(self.X[task])))
        W = -self.sigmoid.hessian_log_likelihood(y, f)
        # TODO: Revisar las dos siguientes lineas:
        # self.K = self.cov_function.cov_matrix(self.hyperparameters, self.X[task])
        # K = self.K + eps*I
        K = self.cov_function.cov_matrix(self.hyperparameters, self.X[task])
        K = K + eps*I
        sqrt_W = np.sqrt(W)
        self.L = cholesky(I + sqrt_W*K*sqrt_W)
        L = self.L
        k_star = compute_k_star(self.cov_function.cov_function, self.hyperparameters, self.X[task], x_star)
        # f_mean = dot(k_star.T, self.sigmoid.gradient_log_likelihood(y, f))
        f_mean = (k_star.T)*(self.sigmoid.gradient_log_likelihood(y, f))
        v = backslash(L, sqrt_W*k_star)
        k_star_star = compute_k_star(self.cov_function.cov_function, self.hyperparameters, x_star, x_star)
        f_var = k_star_star - np.dot(v.T, v)

        # TODO:
        # esto se puede hacer mejor
        def aux_fun(z, i):
            return self.sigmoid.evaluate(z)*scipy.stats.norm(f_mean[i,0], f_var[i,i]).pdf(z)

        # pi_star = integrate(aux_fun, -np.inf, np.inf)
        self.pi_star = []
        for i in range(num_cols(x_star)):
            self.pi_star.append(integrate(aux_fun, -np.inf, np.inf, args=(i,))[0])
        return self.pi_star
        """def aux_fun(f_i):
            return 1 if f_i >= 0.5 else -1
        return map(aux_fun, f_mean)"""

    def gpc_optimize(self, tasks):
        def my_prediction(hyperparameters):
            mlml = self.gpc_find_mode(tasks, hyperparameters)[1]
            print mlml
            return mlml
        def my_grad(hyperparameters):
            grad = self.gpc_gradient_mlogML(hyperparameters, tasks)
            # print grad
            return grad
        return l_bfgs(my_prediction, np.array(self.cov_function.INITIAL_GUESS), fprime=my_grad, maxfun=10)
