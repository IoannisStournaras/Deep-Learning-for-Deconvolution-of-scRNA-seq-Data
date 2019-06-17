import numpy as np
import torch
from torch.utils import data
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __getitem__(self, index):
        row = self.data[index]
        row = torch.from_numpy(np.array(row, dtype='float32'))
        label = self.labels[index]
        label = torch.from_numpy(np.array(label))
        return row, label, torch.from_numpy(np.array(index))

    def __len__(self):
        n, _ = self.data.shape
        return n

def Permute_Dataset(Original_data,load_ready=False):
    if load_ready:
        path = '/home/john/Desktop/Dissertation/data/Corrupted_1.npy'
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
        ID1 = list(np.where(np.diff(coeffients, axis=0)>0)[1])
        ID2 = list(np.where(np.diff(coeffients, axis=0)<0)[1])
    else:
        ID1 = list(np.where(np.diff(coeffients, axis=0)<0)[1])
        ID2 = list(np.where(np.diff(coeffients, axis=0)>0)[1])
    mydict1 = dict.fromkeys(np.array(cells_ID)[ID1], 0)
    mydict2 = dict.fromkeys(np.array(cells_ID)[ID2], 1)
    predictions = dict.fromkeys(cells_ID)
    predictions.update(mydict1)
    predictions.update(mydict2)
    return predictions