from __future__ import print_function
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from PAE-NMF import VAE
from Datasets import Dataset
from BMF import BMF
from tqdm import tqdm
import numpy as np
import pickle
import json

parser = argparse.ArgumentParser(description='VAE for NMF')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--cuda', action='store_false', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=22, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embeddings', default=2, type=int,
                    help='number of clusters (default: 2)')

args = parser.parse_args()
print(torch.cuda.is_available())
args.cuda = not args.cuda and torch.cuda.is_available()
print(args.cuda)
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

print('Loading the dataset...')
path = '/home/john/Desktop/Dissertation/Dataset1/labels_PCA'
with open(path, 'rb') as f:
    labels_dict = pickle.load(f)
labels_ID = list(labels_dict.keys())

path = '/home/john/Desktop/Dissertation/Dataset1/Dataset_1.npy'
df_train = np.load(path) 
labels = np.array(list(labels_dict.values()))
print('Creating DataLoader...')
train_dataset = Dataset(data=df_train,labels=labels)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)

print('Created the DataLoader')

#Initializing model-optimizer-scheduler
model = VAE(df_train.shape[1],[400,100],args.embeddings,[100,400]).to(device)
print(model)

if __name__ == "__main__":
    model.fit(epochs=args.epochs, train_loader = train_loader, lr = args.lr, device=device)
    path_to_save_params = '/home/john/Desktop/Dissertation/Dataset3/Pretrained Weights/3HL'
    torch.save(model.state_dict(), path_to_save_params)

    print('Executing Binary Matrix Factorization...')
    model = model.to('cpu')
    row = torch.from_numpy(np.array(df_train, dtype='float32'))
    row = row.to('cpu')
    recon_batch, loglamba, logkappa, H = model(row)
    W_learned = model.H.weight.detach().numpy()
    for layer in model.decode_layers:
        Wi = layer.weight.detach().numpy()
        W_learned = Wi@W_learned
    H_learned = H.detach().numpy().T

    model_1 = BMF(W_init=W_learned,H_init=H_learned,tol=1e-5)
    W_final, H_final = model_1.fit_transform(df_train.T)
    path = '/home/john/Desktop/Dissertation//Dataset3/Decompositions/3HL_BMF_basis.npy'
    np.save(path, W_final)
    path = '/home/john/Desktop/Dissertation//Dataset3/Decompositions/3HL_BMF_coeff.npy'
    np.save(path, H_final)

