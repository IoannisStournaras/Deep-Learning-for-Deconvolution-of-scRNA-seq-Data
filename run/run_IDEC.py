from __future__ import print_function
import sys
sys.path.insert(0,'../Networks')
sys.path.insert(0,'..')
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from sklearn.cluster import KMeans
from Networks.IDEC import IDEC
from Datasets import Dataset
from tqdm import tqdm
import numpy as np
import pickle
import os

dir_path = "/home/s1879286/Dissertation/Deep-Learning-for-Deconvolution-of-scRNA-seq-Data"
pretrain_path = os.path.join(dir_path,'Pretrained','pretrained_AE_3.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE for NMF')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train IDEC (default: 10)')
    parser.add_argument('--AE_epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train Auto-encoder (default: 20)')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--pretrain', type=str, default=pretrain_path,
                        help='Path to load pretrained weights for auto-encoder')
    parser.add_argument('--clusters', default=2, type=int,
                        help='number of clusters (default: 2)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_target', default=1, type=int,
                        help='how many epochs to wait before update target distribution')
    parser.add_argument('--tol', default=0.001, type=float,
                        help='percentage of clusters changed from previous update')
    args = parser.parse_args()
    print(torch.cuda.is_available())
    args.cuda = not args.cuda and torch.cuda.is_available()
    print(args.cuda)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    print('Loading the dataset...')
    path = os.path.join(dir_path,'labels_3.npy')
    labels = np.load(path)    
    #with open(path, 'rb') as f:
    #    labels_dict = pickle.load(f)
    #labels_ID = list(labels_dict.keys())

    path = os.path.join(dir_path,'Dataset_3.npy')
    df_train = np.load(path)
    df_train = df_train.T
    #labels = np.array(list(labels_dict.values()))

    train_dataset = Dataset(data=df_train,labels=labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            **kwargs)

    valid_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=256,
                                            shuffle=False)
    print('Created the DataLoader')
    input_dims = df_train.shape[1]
    encoder_dims = [750,700,1000]
    decoder_dims = encoder_dims[::-1]
    model = IDEC(n_input=input_dims, 
                encode=encoder_dims, 
                latent= args.clusters,
                decode= decoder_dims,
                n_clusters=args.clusters)

    if args.pretrain != "":
        print("Loading model from %s..." % args.pretrain)
        model.load_model(args.pretrain)
    else:
        path = os.path.join(dir_path,"Pretrained","pretrained_AE_3.pt")
        train_loader_AE = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=64,
                                            shuffle=True,
                                            **kwargs)
        model.pretrain(train_loader_AE, path, num_epochs=args.AE_epochs)
    #    model.load_model(path)
    path = os.path.join(dir_path,'Pretrained','IDEC_3.pt')
    model.initialize_kmeans(valid_loader)
    model.fit(valid_loader, train_loader,path, lr=args.lr, num_epochs=args.epochs,
             update_target=args.update_target,gama=args.gamma,tolerance=args.tol)
    model.save_model(path)
