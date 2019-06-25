import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from Denoising_AE import StackedDAE, buildNetwork
import numpy as np
import math
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import pickle
from utils import cluster_acc,adjust_learning_rate, init_weights

class VaDE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_centroids=10, binary=True,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500]):
        super(VaDE, self).__init__()
        
        #Parameters used for the pretraining
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.binary = binary
        self.encodeLayer = encodeLayer
        self.decodeLayer = decodeLayer

        #Parameters used in VaDE
        self.n_centroids = n_centroids
        #Encoder 
        self.encoder = buildNetwork([input_dim] + encodeLayer)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)
        #Decoder
        self.decoder = buildNetwork([z_dim] + decodeLayer)    
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        #if dataset binary enable sigmoid in last layer
        if binary:
            self._dec_act = nn.Sigmoid()

        #Create Gaussian Parameters
        self.create_gmmparam(n_centroids, z_dim)

    def create_gmmparam(self, n_centroids, z_dim):
        self.class_prop = nn.Parameter(torch.ones(n_centroids)/n_centroids, requires_grad=False)
        self.gmm_mean = nn.Parameter(torch.zeros(z_dim, n_centroids))
        self.gmm_cov = nn.Parameter(torch.ones(z_dim, n_centroids))

    def initialize_gmm(self, dataloader):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Initializing Gaussian Mixture Parameters=======")
        self.eval()
        data = []
        loop = tqdm(dataloader)
        for batch_idx, (inputs, _,_) in enumerate(loop):
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, _, _, _ = self.forward(inputs)
            data.append(z.data.cpu().numpy())
        data = np.concatenate(data)
        gmm = GaussianMixture(n_components=self.n_centroids,covariance_type='diag')
        gmm.fit(data)
        self.gmm_mean.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.gmm_cov.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

    def pretrain(self, train_loader, path, dropout=0.1, lr=0.001, num_epochs=50, corrupt=0.3, cuda=False):

        sdae = StackedDAE(input_dim=self.input_dim, z_dim=self.z_dim, binary=self.binary, encodeLayer=self.encodeLayer,
                        decodeLayer=self.decodeLayer,activation="relu", dropout=dropout)

        sdae.apply(init_weights)
        if self.binary == True:
            criterion = "binary-cross-entropy"
        else: criterion = 'mse'
        sdae.fit(train_loader, lr=lr, num_epochs=num_epochs, corrupt=corrupt, loss_type=criterion, use_cuda=cuda)
        sdae.save_model(path)

    def reparameterize(self, mu, logvar):
        #Reparamererization trick
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def compute_gamma(self, z):
        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
        mean_NDK = self.gmm_mean.unsqueeze(0).expand(z.size()[0], self.gmm_mean.size()[0], self.gmm_mean.size()[1]) # NxDxK
        cov_NDK = self.gmm_cov.unsqueeze(0).expand(z.size()[0], self.gmm_cov.size()[0], self.gmm_cov.size()[1])
        prop_NK = self.class_prop.unsqueeze(0).expand(z.size()[0], self.n_centroids) # NxK

        #Gamma as described in the original paper -> p(c|x) approximated by p(c|z)
        p_c_z = torch.exp(torch.log(prop_NK) - torch.sum(0.5*torch.log(2*math.pi*cov_NDK)+\
            (Z-mean_NDK)**2/(2*cov_NDK), dim=1)) + 1e-10 # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma

    def loss_function(self, recon_x, x, z, z_mean, z_log_var):
        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
        z_mean_NDK = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids) # NxDxK
        z_log_var_NDK = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids) # NxDxK
        mean_NDK = self.gmm_mean.unsqueeze(0).expand(z.size()[0], self.gmm_mean.size()[0], self.gmm_mean.size()[1]) # NxDxK
        cov_NDK = self.gmm_cov.unsqueeze(0).expand(z.size()[0], self.gmm_cov.size()[0], self.gmm_cov.size()[1]) # NxDxK
        prop_NK = self.class_prop.unsqueeze(0).expand(z.size()[0], self.n_centroids) # NxK
        
        #Gamma p(c|z)
        gamma = self.compute_gamma(z) #NxK
        
        #First term(reconstruction error)
        BCE = -torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
            (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
        
        ##second term logp(z|c)
        logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(cov_NDK)+\
            torch.exp(z_log_var_NDK)/cov_NDK + (z_mean_NDK-mean_NDK)**2/cov_NDK, dim=1), dim=1)
        
        #third term logq(z|x)        
        qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), 1)
        
        ##fourth term logp(c)
        logpc = -torch.sum(torch.log(prop_NK)*gamma, 1)
        
        #fifth term logq(c|x)
        logqcx = torch.sum(torch.log(gamma)*gamma, 1)

        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(BCE + logpzc + qentropy + logpc + logqcx)
        return loss

    def forward(self, x):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar

    def fit(self, trainloader, validloader ,lr=0.001,  num_epochs=10, anneal=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Fitting the model......Patience=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        train_error = []
        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            if anneal:
                epoch_lr = adjust_learning_rate(lr, optimizer, epoch)
            train_loss = 0
            loop = tqdm(trainloader)
            for batch_idx, (inputs, _, _) in enumerate(loop):
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                z, outputs, mu, logvar = self.forward(inputs)
                loss = self.loss_function(outputs, inputs, z, mu, logvar)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                # print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                #     batch_idx, recon_loss.data[0]))

            self.eval()
            Y = []
            Y_pred = []
            probabilities =[]
            for batch_idx, (inputs, labels,_) in enumerate(validloader):
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                z, outputs, mu, logvar = self.forward(inputs)

                q_c_x = self.compute_gamma(z).data.cpu().numpy()
                probabilities.append(q_c_x)
                Y.append(labels.numpy())
                Y_pred.append(np.argmax(q_c_x, axis=1))

            Y = np.concatenate(Y)
            Y_pred = np.concatenate(Y_pred)
            acc = cluster_acc(Y_pred, Y)
            # valid_loss = total_loss / total_num
            print("#Epoch %3d: lr: %.5f, Train Loss: %.5f, acc: %.5f" % (
                epoch, epoch_lr, train_loss / len(trainloader.dataset), acc[0]))
            train_error.append(train_loss / len(trainloader.dataset))


        probabilities = np.concatenate(probabilities)    
        path = '/home/john/Desktop/Dissertation/Dataset1/TrainingError/VaDE'
        with open(path, 'wb') as f:
            pickle.dump(train_error, f)
        path = '/home/john/Desktop/Dissertation/Dataset1/Decompositions/VaDE_labels.npy'
        np.save(path,Y_pred)
        path = '/home/john/Desktop/Dissertation/Dataset1/Decompositions/VaDE_probs.npy'
        np.save(path,probabilities)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

