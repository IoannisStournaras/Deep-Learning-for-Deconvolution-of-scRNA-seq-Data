import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)


class GaussianNoise(nn.Module):
    def __init__(self,sigma=0.1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0).to('cuda')
    
    def forward(self,x):
        x = x.to('cuda')
        if self.training and self.sigma!=0:
            scale = self.sigma*x.detach()
            sampled_noise = self.noise.repeat(*x.size()).normal_()*scale
            x = x + sampled_noise
        return x

class AAE(nn.Module):
    def __init__(self, input_dim, z_dim=10, y_dim = 2, h_dim=500):
        super(AAE, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,h_dim),
            nn.ReLU()
        )
        self.encode_y = nn.Linear(h_dim,y_dim)
        self.encode_z = nn.Linear(h_dim,z_dim)
        
        self.decode_y = nn.Linear(y_dim,h_dim,bias=False)
        self.decode_z = nn.Linear(z_dim,h_dim,bias=False)
        self.merged_bias = nn.Parameter(torch.ones(h_dim))

        self.decoder = nn.Sequential(
            nn.Linear(h_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,input_dim),
            nn.Sigmoid()
        )

        self.discriminator_z = nn.Sequential(
            GaussianNoise(),
            nn.Linear(z_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,2)
        )

        self.discriminator_y = nn.Sequential(
            GaussianNoise(),
            nn.Linear(y_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,2)
        )   

    def encode_yz(self, x, apply_softax_y = True):
        internal = self.encoder(x)
        y = self.encode_y(internal)
        z = self.encode_z(internal)
        if apply_softax_y:
            y = nn.functional.softmax(y,dim=1)
        return y,z

    def discriminate_z(self,z,apply_softax=False):
        logit = self.discriminator_z(z)
        if apply_softax:
            return nn.functional.softmax(logit)
        return logit     

    
    def discriminate_y(self,y,apply_softax=False):
        logit = self.discriminator_y(y)
        if apply_softax:
            return nn.functional.softmax(logit)
        return logit

    def decode_yz(self,y,z):
        merged = self.decode_y(y) 
        merged += self.decode_z(z)
        merged += self.merged_bias
        return self.decoder(merged)
    