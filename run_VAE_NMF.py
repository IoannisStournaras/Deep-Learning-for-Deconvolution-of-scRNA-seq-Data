from __future__ import print_function
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from Networks import VAE
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
path = '/home/john/Desktop/Dissertation/data/labels_PCA'
with open(path, 'rb') as f:
    labels_dict = pickle.load(f)
labels_ID = list(labels_dict.keys())

path = '/home/john/Desktop/Dissertation/data/Corrupted_1.npy'
df_train = np.load(path)
df_train = df_train.T
labels = np.array(list(labels_dict.values()))

print('Creating DataLoader...')
train_dataset = Dataset(data=df_train,labels=labels)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **kwargs)

print('Created the DataLoader')

#Initializing model-optimizer-scheduler
model = VAE(df_train.shape[1],args.embeddings).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)

def loss_function(approximation, input_matrix, loglamda, logkappa):
    #Calculation of the Frobenius norm
    Frobenius = 0.5*(torch.norm(input_matrix-approximation)**2)
    
    E = 0.5772 #Euler-Mascheroni constant
    kappa = logkappa.exp()
    lamda = loglamda.exp()
    gamma_input = 1+(1/kappa) 
    i = torch.lgamma(gamma_input).exp()

    #see PAE-NMF by Steven Squires, Adam Prugel-Bennett, Mahesan Niranjan, ICLR 2019
    #https://openreview.net/forum?id=BJGjOi09t7
    #derivation of KL divergence for Weibull distributions
    KLD = torch.sum(logkappa - kappa*loglamda +(kappa-1)*(loglamda-E/kappa)+lamda*i-1)
    return Frobenius + KLD
    
def train(epoch):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader)
    for batch_idx, (data, _, _) in enumerate(loop):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, loglamba, logkappa, _ = model(data)
        loss = loss_function(recon_batch, data, loglamba, logkappa)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        model.fcout.weight.data.clamp_(0)
        #model.fouter.weight.data.clamp_(0)
        model.H.weight.data.clamp_(0)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    if epoch>50:            
        scheduler.step()
    aver_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, aver_loss))
    return aver_loss


if __name__ == "__main__":
    train_loss = []
    for epoch in range(1, args.epochs + 1):
        print("Executing",epoch,"...")
        train_loss.append(train(epoch))
        
    path = '/home/john/Desktop/Dissertation/TrainingError/loss_2HL_corrupted'
    with open(path, 'wb') as f:
        pickle.dump(train_loss, f)
    path_to_save_params = '/home/john/Desktop/Dissertation/Pretrained Weights/2HL_corrupted'
    torch.save(model.state_dict(), path_to_save_params)

    print('Executing Binary Matrix Factorization...')
    model = model.to('cpu')
    row = torch.from_numpy(np.array(df_train, dtype='float32'))
    row = row.to('cpu')
    recon_batch, loglamba, logkappa, H = model(row)
    third = model.fcout.weight.detach().numpy()
    second = model.fouter.weight.detach().numpy()
    first = model.H.weight.detach().numpy()
    W_learned = third@second@first
    H_learned = H.detach().numpy().T

    model_1 = BMF(W_init=W_learned,H_init=H_learned,tol=1e-5)
    W_final, H_final = model_1.fit_transform(df_train.T)
    path = '/home/john/Desktop/Dissertation/Decompositions/3HL_BMF_basis.npy'
    np.save(path, W_final)
    path = '/home/john/Desktop/Dissertation/Decompositions/3HL_BMF_coeff.npy'
    np.save(path, H_final)

