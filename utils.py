import torch
from torch import nn
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import os
import operator
from operator import itemgetter
import json
from  more_itertools import unique_everseen
from sklearn.preprocessing import LabelEncoder
import pickle

def create_Dataset(dir_path,filename='Dataset_1.npy'):
    for file in os.listdir(dir_path):
        if file.endswith(".json"):
            path_ = os.path.join(dir_path, file)
            with open(path_) as datafile:
                data = json.load(datafile)
                features = list(data.values())    
                flat_list = [item for sublist in features for item in sublist]
                flat_unique = list(unique_everseen(flat_list))
                variants = os.path.join(dir_path,'variants')
                with open(variants, 'wb') as f:
                    pickle.dump(flat_unique, f)                     
                dataset = np.zeros((len(flat_unique),len(data.keys())),dtype=np.int8)
                loop = tqdm(data.keys())
                for col,key in enumerate(loop):
                    for val in data[key]:
                        row = flat_unique.index(val)
                        dataset[row,col] = 1
                save_path = os.path.join(dir_path,filename)
                np.save(save_path,dataset)
    return dataset

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

def donor_info(dataset, models_labels, top=20,path='/home/john/Desktop/Dissertation/Dataset1',var=[]):    
    output_dir = {} 
    for file in os.listdir(path):
        if file.endswith(".json"):
            path_ = os.path.join(path, file)
            with open(path_) as datafile:
                data = json.load(datafile)
                features = list(data.values())    
                flat_list = [item for sublist in features for item in sublist]
                if len(var) ==0:
                    variants = list(unique_everseen(flat_list))
                else: variants = var
                cells_ID = list(data.keys())
    
    #Find cells IDs for each donor 
    donor1, donor2 = set(range(0,dataset.shape[0])), set(range(0,dataset.shape[0]))
    for key,value in models_labels.items():
        donor1 = donor1 & set(np.where(value==0)[0])
        donor2 = donor2 & set(np.where(value==1)[0])
    donor1 = list(donor1)
    donor2 = list(donor2)
    output_dir["donor1_index"] = donor1
    output_dir["donor2_index"] = donor2        
    output_dir["donor1_cells"] = list(itemgetter(*donor1)(cells_ID))
    output_dir["donor2_cells"] = list(itemgetter(*donor2)(cells_ID))
    
    #Find number of cells per variant
    Donor1_variants_total = dataset[donor1].sum(axis=0)
    Donor2_variants_total = dataset[donor2].sum(axis=0)
    
    if top!=None:
        ind_2 = np.argpartition(Donor2_variants_total, -top)[-top:]
        ind_1 = np.argpartition(Donor1_variants_total, -top)[-top:]
        top20_names = list(itemgetter(*ind_1)(variants))
        top20_values = list(Donor1_variants_total[ind_1])
        output_dir["donor1_top20"] ={t[1]:t[0] for t in sorted(zip(top20_values, top20_names))}
        top20_names = list(itemgetter(*ind_2)(variants))
        top20_values = list(Donor2_variants_total[ind_2])
        output_dir["donor2_top20"] ={t[1]:t[0] for t in sorted(zip(top20_values, top20_names))}
        
    variants_ = Donor1_variants_total[np.where(Donor1_variants_total>0)]
    names_1 = list(itemgetter(*np.where(Donor1_variants_total>0)[0])(variants))
    output_dir["donor1_variants"] ={t[1]:t[0] for t in zip(variants_, names_1)}
    variants_ = Donor2_variants_total[np.where(Donor2_variants_total>0)]
    names_2 = list(itemgetter(*np.where(Donor2_variants_total>0)[0])(variants))
    output_dir["donor2_variants"] ={t[1]:t[0] for t in zip(variants_, names_2)}
    
    unique_1 = [i for i in names_1 if i not in names_2]
    unique_2 = [i for i in names_2 if i not in names_1]
    b = output_dir["donor1_variants"]
    c = {var : b[var] for var in unique_1}
    output_dir['donor1_cells_unique'] = sorted(c.items(), key=operator.itemgetter(1))    
    b = output_dir["donor2_variants"]
    c = {var : b[var] for var in unique_2}
    output_dir['donor2_cells_unique'] = sorted(c.items(), key=operator.itemgetter(1))
    return output_dir

# Fuctions taking in a module and applying the specified weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except: pass

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)