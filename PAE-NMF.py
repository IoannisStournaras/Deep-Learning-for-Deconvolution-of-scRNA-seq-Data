import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from scipy.special import gamma
from tqdm import tqdm
import pickle

def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = max(init_lr * (0.9 ** (epoch//10)), 0.000001)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class VAE(nn.Module):
    def __init__(self, n_input, encode, rank, decode):
        super(VAE, self).__init__()

        # encoder
        self.enc_Dims = encode.copy()
        self.enc_Dims.insert(0,n_input)    
        self.encode_layers = nn.ModuleList()        
        for idx in range(len(self.enc_Dims)-1):
            self.encode_layers.append(nn.Linear(self.enc_Dims[idx],self.enc_Dims[idx+1]))
        self.L = nn.Linear(self.enc_Dims[-1], rank)
        self.K = nn.Linear(self.enc_Dims[-1], rank)

        # decoder
        self.decode_layers = nn.ModuleList()
        self.dec_Dims = decode.copy() 
        self.dec_Dims.append(n_input) 
        self.H = nn.Linear(rank, self.dec_Dims[0],bias=False)
        for idx in range(len(self.dec_Dims)-1):
            self.decode_layers.append(nn.Linear(self.dec_Dims[idx],self.dec_Dims[idx+1],bias=False))
        
    def encode(self, inputs):
        z = inputs
        for layers in self.encode_layers:
            z = layers(z)
            z = F.relu(z)
        return self.L(z), self.K(z)

    def reparameterize(self, loglamba, logkappa):
        samples = torch.rand_like(loglamba)
        L = loglamba.exp()
        K = logkappa.exp()
        return L*(-samples.log())**(1/K)

    def decode(self, inputs):
        z = inputs
        z = self.H(z)
        for layers in self.decode_layers:
            z = layers(z)
        return z

    def forward(self, x):
        loglamba, logkappa = self.encode(x)
        z = self.reparameterize(loglamba, logkappa)
        return self.decode(z), loglamba, logkappa, z

    def loss_function(self,approximation, input_matrix, loglamda, logkappa):
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

    def fit(self,epochs, train_loader, lr=1e-3, scheduler_act=False, log_interval=23, device='cpu',
                path='/home/john/Desktop/Dissertation/Dataset3/TrainingError/loss_3HL'):
        train_loss = []
        epoch_lr = lr
        optimizer = optim.Adam(self.parameters(), lr=lr)
        if scheduler_act:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
        for epoch in range(1, epochs + 1):
            print("Executing",epoch,"...")  
            self.train()
            epoch_loss = 0
            loop = tqdm(train_loader)
            for batch_idx, (data, _, _) in enumerate(loop):
                data = data.to(device)
                optimizer.zero_grad()
                recon_batch, loglamba, logkappa, _ = self.forward(data)
                loss = self.loss_function(recon_batch, data, loglamba, logkappa)
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
                self.H.weight.data.clamp_(0)
                for layer in self.decode_layers:
                    layer.weight.data.clamp_(0)
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / len(data)))
            if (epoch>50 and scheduler_act):            
                scheduler.step()
            else:
                epoch_lr = adjust_learning_rate(lr, optimizer, epoch)
            aver_loss = epoch_loss / len(train_loader.dataset)
            train_loss.append(aver_loss)
            print('====> Epoch: {} Average loss: {:.4f}, Learning rate: {}'.format(
                epoch, aver_loss, epoch_lr))
        with open(path, 'wb') as f:
            pickle.dump(train_loss, f)