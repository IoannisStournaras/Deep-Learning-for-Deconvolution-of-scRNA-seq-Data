import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from scipy.special import gamma
import tqdm
import json
from typing import Optional

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __getitem__(self, index):
        row = self.data[index]
        row = torch.from_numpy(np.array(row, dtype='float32'))
        label = self.labels[index]
        label = torch.from_numpy(np.array(label))
        return row, label, torch.from_numpy(np.array(index))

    def __len__(self):
        n, _ = self.data.shape
        return n



class VAE(nn.Module):
    def __init__(self,dimensions,rank):
        super(VAE, self).__init__()

        self.fcin = nn.Linear(dimensions, 400)
        self.finter = nn.Linear(400,100)
        self.L = nn.Linear(100, rank)
        self.K = nn.Linear(100, rank)
        self.H = nn.Linear(rank, 100,bias=False)
        self.fouter = nn.Linear(100, 400,bias=False)
        self.fcout = nn.Linear(400, dimensions,bias=False)
        
    def encode(self, x):
        h1 = F.relu(self.fcin(x))
        h2 = F.relu(self.finter(h1))
        return self.L(h2), self.K(h2)

    def reparameterize(self, loglamba, logkappa):
        samples = torch.rand_like(loglamba)
        L = loglamba.exp()
        K = logkappa.exp()
        return L*(-samples.log())**(1/K)

    def decode(self, z):
        h3 = self.H(z)
        h4 = self.fouter(h3)
        return self.fcout(h4)

    def forward(self, x):
        loglamba, logkappa = self.encode(x)
        z = self.reparameterize(loglamba, logkappa)
        return self.decode(z), loglamba, logkappa

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

