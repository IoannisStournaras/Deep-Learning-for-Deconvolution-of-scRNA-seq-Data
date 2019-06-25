import sys
sys.path.insert(0,'..')
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import pickle
from utils import adjust_learning_rate,init_weights,cluster_acc
from typing import Optional
from sklearn.cluster import KMeans
from Stacked_AE import AE 
print(sys.path)
class IDEC(nn.Module):
    def __init__(self, n_input, encode, latent, 
                 decode, n_clusters, alpha=1,
                 cluster_centers: Optional[torch.Tensor] = None):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.n_clusters = n_clusters
        self.ae = AE(n_input, encode,latent, decode)
        self.y_pred_last = None
        self.convergence_iter = 0
        self.prop = None

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

    def pretrain(self, train_loader, path, lr=0.001, num_epochs=50, cuda=False):

        self.ae.apply(init_weights)
        if self.binary == True:
            criterion = "cross-entropy"
        else: criterion = 'mse'
        self.ae.fit(train_loader, lr=lr, num_epochs=num_epochs, loss_type=criterion)
        self.ae.save_model(path)

    def target_distribution(self,batch_q):
        numerator = (batch_q**2) / torch.sum(batch_q,0)
        return numerator / torch.sum(numerator,dim=1, keepdim=True)

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

    def initialize_kmeans(self,train_dataset):
        print("=====Initializing KMeans Centers=======")
        data = train_dataset.data
        data = torch.from_numpy(np.array(data, dtype='float32'))
        use_cuda = torch.cuda.is_available()
        use_cuda =False
        if use_cuda:
            self.cuda()
            data = data.cuda()

        _ , latent = self.ae.forward(data)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(latent.data.cpu().numpy())
        
        self.y_pred_last = y_pred
        
        self.cluster_centers.data = torch.tensor(kmeans.cluster_centers_)

    def update_target_distribution(self,data,labels,tol):
        _, tmp_q, _ = self.forward(data)
        tmp_q = tmp_q.data
        self.prop = self.target_distribution(tmp_q)

        #evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)
        print(y_pred.shape)
        labels_changed = np.sum(y_pred != self.y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        self.y_pred_last = y_pred
        print(labels.shape)
        if labels_changed < tol:
            self.convergence_iter+=1
        else:
            self.convergence_iter = 0 
        return labels_changed, cluster_acc(labels,y_pred)[0]

    def fit(self,train_dataset,train_loader,num_epochs=100,lr=1e-4,update_target=1,gama=0.1,tolerance=1e-4,loss_type="cross-entropy"):
        data = train_dataset.data
        data = torch.from_numpy(np.array(data, dtype='float32'))
        labels = train_dataset.labels
        
        use_cuda =False
        #use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            data = data.cuda()
        if loss_type=="mse":
            criterion = nn.MSELoss()
        elif loss_type=="cross-entropy":
            criterion = nn.BCELoss()
        elif loss_type == 'Frobenius':
            criterion = self.Frobenius_norm()

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
        self.train()
        train_loss = []

        for epoch in range(num_epochs):
            print("Executing IDEC",epoch,"...")
            if epoch % update_target == 0:
                labels_changed, acc = self.update_target_distribution(data,labels,tol=tolerance)
            if self.convergence_iter >=10:
                print('percentage of labels changed {:.4f}'.format(labels_changed), '< tol',tolerance)
                print('Reached Convergence threshold. Stopping training.')
                break
            loop = tqdm(train_loader)
            total_loss = 0
            for batch_idx, (x, _, idx) in enumerate(loop):
                if use_cuda:
                    x = x.cuda()
                    idx = idx.cuda()

                x_bar, q, _ = self.forward(x)
                reconstr_loss = criterion(x_bar,x)
                kl_loss = F.kl_div(q.log(), self.prop[idx])
                loss = gama * kl_loss + reconstr_loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss = total_loss / (batch_idx + 1)
            train_loss.append(epoch_loss)
            print("epoch {} loss={:.4f}, accuracy={:.5f}".format(epoch,epoch_loss,acc))
            scheduler.step()

    def Frobenius_norm(self,approximation=None, input_matrix=None):
        #Minimizing the Frobenius norm
        Frobenius = 0.5*(torch.norm(input_matrix-approximation)**2)
        return Frobenius

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
