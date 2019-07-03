import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from torch.autograd import Variable
import pickle
from utils import adjust_learning_rate,init_weights,cluster_acc

def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise


class StackedDAE(nn.Module):
    def __init__(self, input_dim, z_dim=10, binary=True,
        encodeLayer=[400,250,100], decodeLayer=[100,250,400], activation="relu", 
        dropout=0):
        super(StackedDAE, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)

        return z, self.decode(z)

    def fit(self, trainloader, lr=0.001, num_epochs=10, corrupt=0.1, loss_type="cross-entropy",use_cuda=False):
        """
        data_x: FloatTensor
        """
        #use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Stacked Denoising Autoencoding layer=======")
        optimizer = optim.Adam(self.parameters(), lr=lr)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
        
        if loss_type=="mse":
            criterion = nn.MSELoss()
        elif loss_type=="cross-entropy":
            criterion = nn.BCELoss()

        train_error = []
        for epoch in range(1,num_epochs+1):
            train_loss = 0.0
            loop = tqdm(trainloader)
            for batch_idx, (inputs, _, _) in enumerate(loop):
                inputs_corr = masking_noise(inputs, corrupt)
                if use_cuda:
                    inputs = inputs.cuda()
                    inputs_corr = inputs_corr.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                inputs_corr = Variable(inputs_corr)

                _, outputs = self.forward(inputs_corr)
                recon_loss = criterion(outputs, inputs)
                train_loss += recon_loss.item()*len(inputs)
                recon_loss.backward()
                optimizer.step()
            #scheduler.step()
            train_error.append(train_loss / len(trainloader.dataset))
            print("#Epoch %3d: Reconstruct Loss: %.3f " % 
                (epoch, train_loss))
       # path = '/home/john/Desktop/Dissertation/Dataset1/TrainingError/SDAE'
       # with open(path, 'wb') as f:
        #    pickle.dump(train_error, f)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
