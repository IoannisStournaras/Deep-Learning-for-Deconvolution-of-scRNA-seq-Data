import numpy as np
from numpy import random
import numpy.linalg as LA
import scipy.sparse as sp
from scipy.stats import entropy
from sys import exit
from tqdm import tqdm

class GNMF():
    """
	Attributes
	----------
	W : matrix of basis vectors
	H : matrix of coefficients
	frob_error : frobenius norm
	"""
    def __init__(self, X, rank=10, **kwargs):
        self.X = X
        self._rank = rank
        self.X_dim, self._samples = self.X.shape


    def check_non_negativity(self):
        if self.X.min()<0:
            return 0
        else:
            return 1

    def frobenius_norm(self):
        """ Euclidean error between X and W*H """

        if hasattr(self,'H') and hasattr(self,'W'):
            error = LA.norm(self.X - np.dot(self.W, self.H))
        else:
            error = None

        return error

    def kl_divergence(self):
        """ KL Divergence between X and W*H """

        if hasattr(self,'H') and hasattr(self,'W'):
            V = np.dot(self.W, self.H)
            error = entropy(self.X, V).sum()
        else:
            error = None

        return error

    def initialize_w(self):
        """ Initalize W to random values [0,1]."""

        self.W = np.random.random((self.X_dim, self._rank))

    def initialize_h(self):
        """ Initalize H to random values [0,1]."""

        self.H = np.random.random((self._rank, self._samples))


    def compute_graph(self, weight_type='heat-kernel', param=0.3):
        if weight_type == 'heat-kernel':
            samples = np.matrix(self.X.T)
            sigma= param
            A= np.zeros((samples.shape[0], samples.shape[0]))

            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    num = -(LA.norm(samples[i] - samples[j]))
                    A[i][j]= np.exp(num/sigma)

            return A
        elif weight_type == 'dot-weighting':
            samples = np.matrix(self.X.T)
            A= np.zeros((samples.shape[0], samples.shape[0]))

            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    A[i][j]= np.dot(samples[i],samples[j])

            return A


    def compute_factors(self, max_iter=100, lmd=0, weight_type='heat-kernel', param=None):

        if self.check_non_negativity():
            pass
        else:
            print("The given matrix contains negative values")
            exit()

        if not hasattr(self,'W'):
            self.initialize_w()

        if not hasattr(self,'H'):
            self.initialize_h()

        A = self.compute_graph(weight_type, param)

        D = np.matrix(np.diag(np.asarray(A).sum(axis=0)))

        self.frob_error = np.zeros(max_iter)

        for i in range(max_iter):

            self.update_w(lmd, A, D)

            self.update_h(lmd, A, D)

            self.frob_error[i] = self.frobenius_norm()


    def update_h(self, lmd, A, D):

        eps = 2**-8
        h_num = lmd*np.dot(A, self.H.T)+np.dot(self.X.T, self.W )
        h_den = lmd*np.dot(D, self.H.T)+np.dot(self.H.T, np.dot(self.W.T, self.W))


        self.H = np.multiply(self.H.T, (h_num+eps)/(h_den+eps))
        self.H = self.H.T
        self.H[self.H <= 0] = eps
        self.H[np.isnan(self.H)] = eps


    def update_w(self, lmd, A, D):

        XH = self.X.dot(self.H.T)
        WHtH = self.W.dot(self.H.dot(self.H.T)) + 2**-8
        self.W *= XH
        self.W /= WHtH