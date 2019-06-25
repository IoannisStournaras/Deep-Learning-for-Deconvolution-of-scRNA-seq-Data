import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

def Permute_Dataset(Original_data,load_ready=False):
    if load_ready:
        path = '/home/john/Desktop/Dissertation/Dataset1/Corrupted_1.npy'
        Variants = np.load(path)
    else:
        counts = np.zeros(Original_data.shape[1])
        Variants = Original_data.copy()
        loop = tqdm(range(100000))
        for num in loop:
            idx =np.random.randint(0,Original_data.shape[1],size=2)
            counts[idx] += 1
            i = Original_data[:,idx] #column id
            a=np.where(i[:,0]!=0)[0] #list of cell 1
            b= np.where(i[:,1]!=0)[0] #list of cell 2
            c = random.choice(a)
            d = random.choice(b)
            Variants[d,idx[0]] = Original_data[d,idx[1]]
            Variants[c,idx[0]] = Original_data[c,idx[1]]
            Variants[c,idx[1]] = Original_data[c,idx[0]] 
            Variants[d,idx[1]] = Original_data[d,idx[0]]
        plt.title('Distribution of permutations',fontsize=16)
        plt.xlabel('Number of changes occured')
        plt.hist(counts,bins=60)
        plt.show()

    return Variants

def assign_labels(cells_ID,coeffients,inverse=False):
    """
    Fuction assigning labels to cluster
    Input: coefficients [rank,n_poins]
    output: dictionart {cell_id : label}
    """
    if inverse:
        ID1 = list(np.where(np.diff(coeffients, axis=0)>=0)[1])
        ID2 = list(np.where(np.diff(coeffients, axis=0)<0)[1])
    else:
        ID1 = list(np.where(np.diff(coeffients, axis=0)<=0)[1])
        ID2 = list(np.where(np.diff(coeffients, axis=0)>0)[1])
    mydict1 = dict.fromkeys(np.array(cells_ID)[ID1], 0)
    mydict2 = dict.fromkeys(np.array(cells_ID)[ID2], 1)
    predictions = dict.fromkeys(cells_ID)
    predictions.update(mydict1)
    predictions.update(mydict2)
    return predictions

def cluster_acc(Y_pred, Y):
  assert Y_pred.size == Y.size , "Sizes do not match"
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = max(init_lr * (0.9 ** (epoch//10)), 0.00001)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

# Fuctions taking in a module and applying the specified weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)