from sklearn.decomposition import NMF
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm 
from numpy.linalg import inv
import time

class BMF:
    def __init__(self, rank = 2 ,max_iter =100 ,lamda_init_W = 0.04, lamda_init_H = 0.04,  
                lamda_INC_W = 1.1 , lamda_INC_H = 1.1 , tol = 1e-4, seed = "nndsvd",
                W_init = None, H_init=None, beta_loss='frobenius', alpha = 0):

        self.rank = rank
        self.iter = max_iter
        self.lamda_W = lamda_init_H
        self.lamda_W_inc = lamda_INC_W
        self.lamda_H = lamda_init_H
        self.lamda_H_inc = lamda_INC_H
        self.error = tol 
        
        if (isinstance(W_init == None, bool) and isinstance(H_init == None, bool)):
            print("Initializing using standard NMF")
            self.NMF_model = NMF(n_components=2, init=seed , beta_loss=beta_loss, random_state=2,
                                tol=tol, alpha=alpha)
            self.W = np.array([])
            self.H = np.array([])
        else:
            self.W = W_init
            self.H = H_init 

    def normalize(self):
        '''
        Function normalizing W,H making algorithms more robust
        Returns:
            W: normalized W
            h: normalized H
        '''
        DW = np.diag(self.W.max(axis=0))
        DH = np.diag(self.H.max(axis=1))
        DW_sqrt = np.sqrt(DW)
        DH_sqrt = np.sqrt(DH)
        DW_inv = inv(DW)
        DW_inv_sqrt = np.sqrt(DW_inv)
        DH_inv = inv(DH)
        DH_inv_sqrt = np.sqrt(DH_inv)
        W_norm = np.dot(self.W,np.dot(DW_inv_sqrt,DH_sqrt))
        H_norm = np.dot(DH_inv_sqrt,np.dot(DW_sqrt,self.H))
        return W_norm, H_norm

    def update_H(self,V):
        eps = np.finfo(float).eps
        nom = np.dot(self.W.T,V) + 3*self.lamda_H*(self.H**2)
        denom = np.dot(np.dot(self.W.T,self.W),self.H)+(2*self.lamda_H*self.H**3)+(self.lamda_H*self.H) + eps
        return self.H*(nom/denom)

    def update_W(self,V):
        eps = np.finfo(float).eps
        nom = np.dot(V,self.H.T) + 3*self.lamda_W*(self.W**2)
        denom = np.dot(self.W,np.dot(self.H,self.H.T))+(2*self.lamda_W*self.W**3)+(self.lamda_W*self.W) + eps
        return self.W*(nom/denom)

    def update_lambda(self):
        self.lamda_W *= self.lamda_W_inc
        self.lamda_H *= self.lamda_H_inc

    def fit_transform(self, V, algorithm = "Penalty"):

        #Initialization
        if (self.W.size==0 and self.H.size==0):
            print('Executing standard NMF decomposition...')
            self.W = self.NMF_model.fit_transform(V)  
            self.H = self.NMF_model.components_
            print("Converged at iteration",self.NMF_model.n_iter_)
            print('Recostruction error is', self.NMF_model.reconstruction_err_)
            time.sleep(1)
        
        self.W, self.H = self.normalize()
        if algorithm == "Penalty":
            loop = tqdm(range(self.iter))
            for i in loop:
                self.H = self.update_H(V)
                self.W = self.update_W(V)
                
                constraint_H = (self.H**2-self.H)**2
                constraint_W = (self.W**2-self.W)**2
                
                if (constraint_H < self.error/2).all() and (constraint_W < self.error/2).all():
                    print("Lower error achieved")
                    break
                else: 
                    self.update_lambda()

        return self.W, self.H