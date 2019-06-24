import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from scipy.special import gamma
from tqdm import tqdm
import json
from typing import Optional
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

class AE(nn.Module):

    def __init__(self, n_input, encode, latent, decode):
        """
        n_input: Int -> Dimensions of the dataset
        encode:  List -> Length of the list determines the 
                number of hidden layers for the encoder.
                Each element of the list determines the 
                size of each hidden layer
        latent: Int -> Size of latent feature space
        decode: List ->  Length of the list determines the 
                number of hidden layers for the decoder.
                Each element of the list determines the 
                size of each hidden layer
        """
        super(AE, self).__init__()
        
        self.enc_layers = nn.ModuleList()
        self.enc_Dims = encode.copy()
        self.enc_Dims.insert(0,n_input)

        self.dec_layers = nn.ModuleList()
        self.dec_Dims = decode.copy()
        self.dec_Dims.insert(0,latent)
        
        # encoder        
        for idx in range(len(self.enc_Dims)-1):
            self.enc_layers.append(nn.Linear(self.enc_Dims[idx],self.enc_Dims[idx+1]))

        self.z_layer = nn.Linear(self.enc_Dims[-1], latent)

        # decoder
        for idx in range(len(self.dec_Dims)-1):
            self.dec_layers.append(nn.Linear(self.dec_Dims[idx],self.dec_Dims[idx+1]))
        
        self.fcout = nn.Linear(self.dec_Dims[-1],n_input)

    def forward(self, inputs):

        # encoder
        z = inputs
        for layers in self.enc_layers:
            z = layers(z)
            z = F.relu(z)

        z = self.z_layer(z)
        dec = z

        # decoder
        for layers in self.dec_layers:
            dec = layers(dec)
            dec = F.relu(dec)
        dec = self.fcout(dec)
        x_bar = F.sigmoid(dec)

        return x_bar, z


class IDEC(nn.Module):
    def __init__(self, n_input, encode, latent, decode,
                 n_clusters, pretrain_path, alpha=1,
                 cluster_centers: Optional[torch.Tensor] = None):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters


        self.ae = AE(n_input, encode,
                    latent, decode)

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                latent,
                dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def pretrain(self,pretrain_ae,epochs ,path=''):
        if path == '':
            pretrain_ae(self.ae,epochs)
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained AE from', path)  

    def forward(self,batch):
        """
        Compute soft assignment for an input, returning the input's assignments
        for each cluster.

        batch: [batch_size, input_dimensions]
        output: [batch_size, number_of_clusters]
        """
        x_bar, z = self.ae(batch)

        Frobenius_squared = torch.sum((z.unsqueeze(1)-self.cluster_centers)**2,2)
        numerator = 1.0 / (1.0+(Frobenius_squared/self.alpha))
        power = float(self.alpha+1)/2
        numerator = numerator**power
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return x_bar, q, z        

