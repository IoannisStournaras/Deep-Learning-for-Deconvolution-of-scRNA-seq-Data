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
    parser.add_argument('--clusters', default=2, type=int,
                        help='number of clusters (default: 2)')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--pretrain', type=str, default="/home/john/Desktop/Dissertation/Dataset1/Pretrained Weights/pretrained_SDAE1.pt",
                        help='Path to load pretrained weights for VaDE')


    args = parser.parse_args()
    print(torch.cuda.is_available())
    args.cuda = not args.cuda and torch.cuda.is_available()
    print(args.cuda)

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
    valid_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=512,
                                            **kwargs)
    print('Created the DataLoader')

    vade = VaDE(input_dim=df_train.shape[1], z_dim=args.clusters, n_centroids=args.clusters, binary=True,
        encodeLayer=[700,500,1000], decodeLayer=[1000,500,700])

    if args.pretrain != "":
        print("Loading model from %s..." % args.pretrain)
        vade.load_model(args.pretrain)
    else:
        path = "/home/john/Desktop/Dissertation/Dataset1/Pretrained Weights/pretrained_SDAE1.pt"
        train_loader_AE = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=64,
                                            shuffle=True,
                                            **kwargs)
        vade.pretrain(train_loader_AE, path, cuda=args.cuda)
        vade.load_model(path)


    vade.initialize_gmm(train_loader)
    vade.fit(train_loader, valid_loader, lr=args.lr, num_epochs=args.epochs ,anneal=True)
    vade.save_model("/home/john/Desktop/Dissertation/Dataset1/Pretrained Weights/VaDE_3HL1.pt")
