from __future__ import print_function
import argparse
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from Datasets import Dataset
import pickle
import matplotlib.pyplot as plt


class RICA(nn.Module):
    def __init__(self, input_dim, n_clusters,penalty=0.1):
        super(RICA, self).__init__()
        self.penalty = penalty
        self.weight = nn.Parameter(torch.randn(n_clusters,input_dim))
        self.transform = None

    def forward(self,x):
        "x [dim,batch]"
        latent = self.weight.matmul(x) 
        self.transform = latent.t()
        return latent

    def loss_function(self, x):
        latent = self.forward(x)
        reconstruction = 0.5 * torch.sum((self.weight.t().matmul(latent) - x) ** 2) 
        latent_loss = self.penalty * torch.sum(torch.abs(latent))
        return reconstruction + latent_loss


def train(epoch,model,optimizer,scheduler):
    loop = tqdm(train_loader)
    epoch_loss = 0
    for batch_idx, (x, _, _) in enumerate(loop):
        
        data = x.t().to(device) 
        loss = model.loss_function(data)
        
        epoch_loss += loss.item()
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch >100:
        scheduler.step()
    print('Train Epoch: {}, {} '.format(
                epoch, epoch_loss/(batch_idx+1)))
    
    return epoch_loss/(batch_idx+1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recostruction ICA')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--embeddings', default=2, type=int,
                        help='number of clusters (default: 2)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
 
    args = parser.parse_args()
    args.cuda = not args.cuda and torch.cuda.is_available()
    #torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    print('Loading the dataset...')
    path = '/home/john/Desktop/Dissertation/data/labels_PCA'
    with open(path, 'rb') as f:
        labels_dict = pickle.load(f)
    labels_ID = list(labels_dict.keys())

    path = '/home/john/Desktop/Dissertation/data/Dataset_1.npy'
    df_train = np.load(path)
    labels = np.array(list(labels_dict.values()))

    print('Creating DataLoader...')
    train_dataset = Dataset(data=df_train,labels=labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            **kwargs)
    
    train_loss = []
    model = RICA(df_train.shape[1],n_clusters=256,penalty=1.2).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    for epoch in range(1,args.epochs+1):
        print('Executing Epoch...',epoch)
        train_loss.append(train(epoch,model,optimizer,scheduler))
    
    path = '/home/john/Desktop/Dissertation/TrainingError/RICA_ADAM_loss' 
    with open(path, 'wb') as f:
        pickle.dump(train_loss, f)
    path_to_save_params = '/home/john/Desktop/Dissertation/Pretrained Weights/RICA_ADAM'
    torch.save(model.state_dict(), path_to_save_params)


