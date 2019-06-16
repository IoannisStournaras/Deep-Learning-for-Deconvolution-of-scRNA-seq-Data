from sklearn.decomposition import NMF
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm 
from numpy.linalg import inv
import time

class BMF:
    def __init__(self, rank = 2 ,max_iter =100 ,lamda_init_W = 0.04, lamda_init_H = 0.04,  
                lamda_INC_W = 1.1 , lamda_INC_H = 1.1 , tol = 1e-4, seed = "nndsvd",
                W_init = None, H_init=None, beta_loss='frobenius', alpha = 0,w0=None,h0=None):

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
            assert((W_init>=0).all() and (H_init>=0).all()), "Decompositions contain negative elements, must be Non Negative"
            self.W = W_init
            self.H = H_init 

        self.thres_w = w0
        self.thres_h = h0


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

    def discretization(self,X):
        '''
        Function searching for optimal initial thresholds in a linear domain
        Inputs:
            X: input matrix [dimesions, n_points]

        Returns:
            w: optimal initial threshold for W
            h: optimal initial threshold for H
        '''
        print('Searching for optimal initial thresholds.... ')
        upper_W = self.W.max()
        linspace_W = np.arange(0,upper_W,0.1)
        upper_H = self.H.max()
        linspace_H = np.arange(0,upper_H,0.1)
        error = 1e10
        loop = tqdm(range(linspace_H.shape[0]))
        for i in loop:
            for j in range(linspace_W.shape[0]):
                newH = self.sign(self.H,linspace_H[i])
                newW = self.sign(self.W,linspace_W[j])
                approx = np.dot(newW,newH)
                new_error = norm(X-approx,'fro')
                if new_error < error:
                    error = new_error
                    h = linspace_H[i]
                    w = linspace_W[j]
        self.thres_w = w
        self.thres_h = h

    def phi(self,X):
        """
        Function approximating the Heaviside step function
        """
        return 1/(1+np.exp(-100*X))

    def phi_derivative(self,X):  
        nom = np.exp(-100*X)*100
        denom = (1+np.exp(-100*X))**2
        return nom/denom

    def sign(self,X,threshold):
        """
        Heaviside step function
        """
        Y = np.zeros(X.shape)
        i = np.where((X-threshold)>0)
        Y[i] = 1
        return Y

    def gradient_direction(self,X):
        '''
        Compute the gradient directions gw, gh of F(w,h) 
        Inputs:
            X: input matrix DxN
            w0: threshold for W -> parameter to be optimised
            h0: threshold for H -> parameter to be optimised

        Returns:
            1x2 numpy array containing:
                g_W: gradient direction for W
                g_H: gradient direction for H
        '''
        lamda = 100
        #calculating W* and H* 
        W_star = self.phi(self.W-self.thres_w,lamda)
        H_star = self.phi(self.H-self.thres_h,lamda)
        
        #Calculating g(1) step-by-step
        X_H = np.dot(X,H_star.T)
        W_H_H = np.dot(W_star,np.dot(H_star,H_star.T))
        derivative_W_star = self.phi_derivative(self.W-self.thres_w,lamda)
        total = (X_H - W_H_H)*derivative_W_star
        g_W = np.sum(total)
        
        #Calculating g(2) step-by-step
        X_W = np.dot(W_star.T,X)
        W_W_H = np.dot(W_star.T,np.dot(W_star,H_star))
        derivative_H_star = self.phi_derivative(self.H-self.thres_h,lamda)
        total = (X_W - W_W_H)*derivative_H_star
        g_H = np.sum(total)
        
        return np.array([[g_W,g_H]])

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
        assert (algorithm=='Penalty' or algorithm=='Thresholding'), "Only Penalty and Thresholding algorithms supported"
        assert ((V>=0).all()), "Input contains negative elements, matrix must be Non Negative"

        #Initialization
        if (self.W.size==0 and self.H.size==0):
            print('Executing standard NMF decomposition...')
            self.W = self.NMF_model.fit_transform(V)  
            self.H = self.NMF_model.components_
            print("Converged at iteration",self.NMF_model.n_iter_)
            print('Recostruction error is', self.NMF_model.reconstruction_err_)
            time.sleep(1)
        
        self.W, self.H = self.normalize()
        init_error = norm(V-np.dot(self.W,self.H),'fro')

        if algorithm == "Penalty":
            loop = tqdm(range(self.iter))
            for i in loop:
                
                self.H = self.update_H(V)
                self.W = self.update_W(V)
                update_error = norm(V-np.dot(self.W,self.H),'fro')
                
                constraint_H = (self.H**2-self.H)**2
                constraint_W = (self.W**2-self.W)**2
                
                if (constraint_H < self.error/2).all() and (constraint_W < self.error/2).all():
                    print("Lower error achieved")
                    break
                elif abs(init_error - update_error) < self.error :
                    print('Converged after',i,'iterations')
                    break
                else: 
                    self.update_lambda()
                    init_error = update_error
            return self.W, self.H
        elif algorithm == 'Thresholding':
            self.Thres_algorithm(V)
            return self.thres_w, self.thres_h


    def Thres_algorithm(self,X):
        '''
        Inputs:
            lamda: large constant used to approximate the heaviside function
            X: input matrix DxN
        Returns:
            w: updated threshold for W
            h: updated threshold for H
        '''
        lamda = 100
        iteration = 0 
        w, h = self.thres_w, self.thres_h
        approx = np.dot(self.sign(self.W,self.thres_w),self.sign(self.H,self.thres_h))
        init_error = norm((X-approx),'fro')**2
        sigma = 0.4
        delta = 0.1
        
        loop = tqdm(range(1,self.iter +1))
        for iteration in loop:
            approx = np.dot(self.phi(self.W-w,lamda),self.phi(self.H-h,lamda))
            F_k = norm(X-approx,'fro')**2
            gk = self.gradient_direction(X)
            dk = -gk
            if norm(gk)**2<self.error:
                break
            
            #Wolfie line search method
            k=1
            step_size = 2
            w_k1 = w + step_size*dk[0,0]
            h_k1 = h + step_size*dk[0,1]
            flag = True
            scalara=0
            scalarb=10
            #internal = tqdm(range(10))
            for k in range(10):
                approx = np.dot(self.phi(self.W-w_k1,lamda),self.phi(self.H-h_k1,lamda))
                F_k1 = norm(X-approx,'fro')**2
                if 0.5*(F_k1-F_k) <= delta*step_size*np.dot(gk,dk.T):
                    g_k1 = self.gradient_direction(X)
                    if np.dot(g_k1,dk.T)>=sigma*np.dot(gk,dk.T):
                        break
                    else:
                        if scalarb <10:
                            scalara = step_size
                            step_size = (scalara+scalarb)/2
                        else:
                            step_size = 1.2*step_size

                else:
                    scalarb = step_size
                    step_size = (scalara+scalarb)/2
    
            
            w, h  = w_k1, h_k1
            
            # stopping criteria
            print(iteration, " iteration with error", np.abs(F_k1-F_k))
            if np.abs(F_k1-F_k)< self.error: break
            
            
        new_approx = np.dot(self.sign(self.W,w),self.sign(self.H,h))
        if norm(X-new_approx,'fro')**2 > init_error:
            w = self.thres_w
            h = self.thres_h 
            
        self.thres_h = h
        self.thres_w = 2


def my_NMF(X,rank,epsilon,n_iter=2,seed=42):
    '''
    Negative Matrix Factorization function minimizing the Frobenius norm
    Inputs:
        X: Input matrix DxN
        rank: number of bases to compute (column rank of W)
        epsilon: error rate 

    Returns:
        W: basis DxK where K is the rank 
        H: coefficients KxN where K is the rank
    '''
    np.random.seed(seed)
    if X.min()<0:
        raise ValueError("Non negative entries")
    
    #initialize W,H matrices
    W = np.random.rand(X.shape[0],rank)
    H = np.random.rand(rank,X.shape[1])
    iteration = 0
    
    while True:
        F = norm(X-np.dot(W,H),'fro')**2
        wv = np.dot(W.T,X)
        wwh = np.dot(W.T,np.dot(W,H))
        #wwh = np.amax(wwh,axis=0,keepdims=True,initial=np.finfo(float).eps)
        wwh = np.maximum(wwh,np.finfo(float).eps)
        H = H*wv/wwh
        vh = np.dot(X,H.T)
        whh = np.dot(W,np.dot(H,H.T))
       # whh = np.amax(whh,axis=0,keepdims=True,initial=np.finfo(float).eps)
        whh = np.maximum(whh,np.finfo(float).eps)
        W = W*vh/whh
        
        #stopping criteria
        F_update = norm(X-np.dot(W,H),'fro')**2
        if np.abs(F_update-F)<epsilon:
            iteration+=1
        else: iteration=0
        if iteration==n_iter: break
            
    return W,H



