from __future__ import print_function
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from Networks import  AE, IDEC
from Datasets import Dataset
from tqdm import tqdm
import numpy as np
import pickle
import json

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
parser.add_argument('--log-interval', type=int, default=22, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pretrain_path', type=str, default='/home/john/Desktop/Dissertation/Pretrained Weights/pretrained_AE_cor',
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
path = '/home/john/Desktop/Dissertation/data/labels_PCA'
with open(path, 'rb') as f:
    labels_dict = pickle.load(f)
labels_ID = list(labels_dict.keys())

path = '/home/john/Desktop/Dissertation/data/Dataset_1.npy'
df_train = np.load(path)
#df_train = df_train.T
labels = np.array(list(labels_dict.values()))

train_dataset = Dataset(data=df_train,labels=labels)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)

print('Created the DataLoader')
input_dims = df_train.shape[1]
encoder_dims = [400,200,100]
latent_space = 50
decoder_dims = encoder_dims[::-1]
model = IDEC(input_dims,
            encoder_dims,
            latent_space,
            decoder_dims,
            args.clusters,
            args.pretrain_path).to(device)


def loss_function(approximation, input_matrix):
    #Minimizing the Frobenius norm
    Frobenius = 0.5*(torch.norm(input_matrix-approximation)**2)
    return Frobenius

def pretrain_ae(ae_model,epochs):
    '''
    pretrain autoencoder
    '''
    train_loss = []
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
    for epoch in range(1, epochs + 1):
        print("Executing",epoch,"...")
        ae_model.train()
        total_loss = 0
        loop = tqdm(train_loader) 
        for batch_idx, (data, _, _) in enumerate(loop):
            data = data.to(device)
            optimizer.zero_grad()
            x_bar, _ = ae_model(data)
            loss = loss_function(x_bar, data)
            
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if epoch>=50:
            scheduler.step()
        epoch_loss = total_loss / (batch_idx + 1)
        train_loss.append(epoch_loss)
        print("epoch {} loss={:.4f}".format(epoch,
                                            epoch_loss))
    torch.save(ae_model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))
    path = '/home/john/Desktop/Dissertation/TrainingError/AE_loss_cor'
    with open(path, 'wb') as f:
        pickle.dump(train_loss, f)


def target_distribution(batch_q):
    numerator = (batch_q**2) / torch.sum(batch_q,0)
    return numerator / torch.sum(numerator,dim=1, keepdim=True)

def train_IDEC(epochs,AE_epochs=100,AE_path=''):
    #pretrain auto_encoder as no path to load pretrained 
    #weights has been specified
    model.pretrain(pretrain_ae,AE_epochs,AE_path)
    device = torch.device('cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
    data = train_dataset.data
    data = torch.from_numpy(np.array(data, dtype='float32')).to(device)
    _ , latent = model.ae(data)
    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    y_pred = kmeans.fit_predict(latent.data.cpu().numpy())
    
    latent = None
    y_pred_last = y_pred
    
    model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_).to(device)
    model.train()
    train_loss = []
    convergence_iter = 0
    for epoch in range(epochs):
        print("Executing IDEC",epoch,"...")
        if epoch % args.update_target == 0:
            _, tmp_q, _ = model(data)

            #update target distribution
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            #evaluate clustering performance

            labels_changed = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            
            if epoch > 0 and labels_changed < args.tol:
                convergence_iter+=1
            else:
                convergence_iter = 0 
            if convergence_iter >10:
                print('percentage of labels changed {:.4f}'.format(labels_changed), '< tol',args.tol)
                print('Reached Convergence threshold. Stopping training.')
                break
        loop = tqdm(train_loader)
        total_loss = 0
        for batch_idx, (x, _, idx) in enumerate(loop):

            x = x.to(device)
            idx = idx.to(device)

            x_bar, q, _ = model(x)

            reconstr_loss = loss_function(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = args.gamma * kl_loss + reconstr_loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = total_loss / (batch_idx + 1)
        train_loss.append(epoch_loss)
        print("epoch {} loss={:.4f}".format(epoch,
                                            epoch_loss))
        scheduler.step()

if __name__ == "__main__":
    train_IDEC(args.epochs,args.AE_epochs)
    path = '/home/john/Desktop/Dissertation/Pretrained Weights/IDEC_3_cor'
    torch.save(model.state_dict(), path)
