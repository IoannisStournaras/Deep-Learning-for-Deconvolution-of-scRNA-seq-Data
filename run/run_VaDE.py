import sys
sys.path.insert(0,'../Networks')
sys.path.insert(0,'..')
import torch
import torch.utils.data
import argparse
import pickle
import numpy as np
from Datasets import Dataset
from VADE import VaDE
from BMF import BMF
import os

dir_path = '/home/s1879286/Dissertation/Deep-Learning-for-Deconvolution-of-scRNA-seq-Data'
pretrain = os.path.join(dir_path,'Pretrained','pretrained_SDAE_3.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SDAE Pretrain')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dropout', type=int, default=0.1, metavar='N',
                        help='input dropout for training (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--AE_epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train Auto-encoder (default:30)') 
    parser.add_argument('--clusters', default=2, type=int,
                        help='number of clusters (default: 2)')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--pretrain', type=str, default=pretrain,
                        help='Path to load pretrained weights for VaDE')


    args = parser.parse_args()
    print(torch.cuda.is_available())
    args.cuda = not args.cuda and torch.cuda.is_available()
    print(args.cuda)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    
    print('Loading the dataset...')
    path = os.path.join(dir_path,'labels_3.npy')
  #  with open(path, 'rb') as f:
  #      labels_dict = pickle.load(f)
  #  labels_ID = list(labels_dict.keys())
    labels = np.load(path)
    path = os.path.join(dir_path,'Dataset_3.npy')
    df_train = np.load(path)
    df_train = df_train.T
    #labels = np.array(list(labels_dict.values()))

    print('Creating DataLoader...')
    train_dataset = Dataset(data=df_train,labels=labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            **kwargs)
    valid_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=128,
                                            **kwargs)
    print('Created the DataLoader')

    vade = VaDE(input_dim=df_train.shape[1], z_dim=args.clusters, n_centroids=args.clusters, binary=True,
        encodeLayer=[750,700,1000], decodeLayer=[1000,700,750])

    if args.pretrain != "":
        print("Loading model from %s..." % args.pretrain)
        vade.load_model(args.pretrain)
    else:
        path = pretrain
        train_loader_AE = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=64,
                                            shuffle=True,
                                            **kwargs)
        vade.pretrain(train_loader_AE, path,num_epochs=args.AE_epochs, cuda=args.cuda,lr=0.0005)
        vade.load_model(path)


    vade.initialize_gmm(train_loader)
    path = os.path.join(dir_path,'Pretrained','VaDE_3.pt') 
    vade.fit(train_loader, valid_loader, path, lr=args.lr, num_epochs=args.epochs ,anneal=True)
    vade.save_model(path)
    print(vade.class_prop)
