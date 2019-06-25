import sys
sys.path.insert(0,'..')
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from Datasets import Dataset
from Networks.AAE import AAE

def check_grads(model, model_name):
    grads = []
    for p in model.parameters():
        if not p.grad is None:
            grads.append(float(p.grad.mean()))

    grads = np.array(grads)
    if grads.any() and grads.mean() > 5:
        print(f"WARNING! gradients mean is over 100 ({model_name})")
    if grads.any() and grads.max() > 5:
        print(f"WARNING! gradients max is over 100 ({model_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SDAE Pretrain')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dropout', type=int, default=0.1, metavar='N',
                        help='input dropout for training (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--clusters', default=2, type=int,
                        help='number of clusters (default: 2)')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--clip', default=5, type=int,
                        help='Clipping gradient (default: 5)')
    parser.add_argument('--pretrain', type=str, default="/home/john/Desktop/Dissertation/Dataset1/Pretrained Weights/pretrained_SDAE1.pt",
                        help='Path to load pretrained weights for VaDE')


    args = parser.parse_args()
    print(torch.cuda.is_available())
    args.cuda = not args.cuda and torch.cuda.is_available()
    print(args.cuda)
    device = torch.device("cuda" if args.cuda else "cpu")
    
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    
    print('Loading the dataset...')
    path = '/home/john/Desktop/Dissertation/Dataset1/labels_PCA'
    with open(path, 'rb') as f:
        labels_dict = pickle.load(f)
    labels_ID = list(labels_dict.keys())

    path = '/home/john/Desktop/Dissertation/Dataset1/Dataset_1.npy'
    df_train = np.load(path)
    labels = np.array(list(labels_dict.values()))

    print('Creating DataLoader...')
    train_dataset = Dataset(data=df_train,labels=labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            **kwargs)
    print('Created the DataLoader')

    aae = AAE(input_dim=df_train.shape[1], z_dim=2, y_dim=args.clusters ,h_dim=700).to(device)
    optim_encoder = torch.optim.Adam(aae.encoder.parameters(),lr=1e-4)
    optim_decoder = torch.optim.Adam(aae.decoder.parameters(),lr=1e-4)

    optim_discriminator_z = torch.optim.Adam(aae.discriminator_z.parameters(),lr=0.0001)
    optim_discriminator_y = torch.optim.Adam(aae.discriminator_y.parameters(),lr=0.0001)

    gaussian = torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([1.0]))
    categorical = torch.distributions.OneHotCategorical(torch.tensor([0.5,0.5]))
    class_true = torch.zeros(args.batch_size).long().to(device)
    class_fake = torch.ones(args.batch_size).long().to(device)
    criterion = nn.CrossEntropyLoss()

if __name__=='__main__':
    for epoch in range(1,args.epochs+1):
        print("Executing",epoch,"...")
        aae.train()
        loop = tqdm(train_loader)
        recon = 0
        discr = 0
        gener = 0
        for batch_idx, (data, _, _) in enumerate(loop):
            #reconstruction phase
            class_true = torch.zeros(data.shape[0]).long().to(device)
            class_fake = torch.ones(data.shape[0]).long().to(device)

            data = data.to(device)
            optim_encoder.zero_grad()
            optim_decoder.zero_grad()

            y_ohe, z = aae.encode_yz(data,apply_softax_y=True)
            x_bar = aae.decode_yz(y_ohe, z)
            recon_loss = F.binary_cross_entropy(x_bar,data)
            recon += recon_loss.item()
            recon_loss.backward()
            
            torch.nn.utils.clip_grad_norm(aae.parameters(), args.clip)
            optim_decoder.step()
            optim_encoder.step()

            #adversarial phase
            optim_discriminator_y.zero_grad()
            optim_discriminator_z.zero_grad()
            y_ohe_fake, z_fake = aae.encode_yz(data,apply_softax_y=True)
            z_true = gaussian.sample((data.shape[0],aae.z_dim))
            z_true = z_true.squeeze(2)
            y_ohe_true = categorical.sample((data.shape[0],))
            z_true.to(device)
            y_ohe_true.to(device) 

            dz_true = aae.discriminate_z(z_true, apply_softax=False)
            dz_fake = aae.discriminate_z(z_fake, apply_softax=False)
            dy_true = aae.discriminate_y(y_ohe_true, apply_softax=False)
            dy_fake = aae.discriminate_y(y_ohe_fake, apply_softax=False)

            loss_discriminator_z = criterion(dz_true,class_true) + criterion(dz_fake,class_fake)
            loss_discriminator_y = criterion(dy_true,class_true) + criterion(dy_fake,class_fake)
            loss_disc = loss_discriminator_z + loss_discriminator_y
            discr+=loss_disc.item()
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm(aae.parameters(), args.clip)
            optim_discriminator_z.step()
            optim_discriminator_y.step()

            #generator phase
            optim_encoder.zero_grad()
            y_ohe_fake, z_fake = aae.encode_yz(data, apply_softax_y=True)
            dz_fake = aae.discriminate_z(z_fake,apply_softax=False)
            dy_fake = aae.discriminate_y(y_ohe_fake, apply_softax=False)
            loss_gen = criterion(dz_fake,class_true)+criterion(dy_fake,class_true)
            gener += loss_gen.item() 
            loss_gen.backward()

            torch.nn.utils.clip_grad_norm(aae.parameters(), args.clip)
            optim_encoder.step()
        print("epoch {} recon_loss={:.4f} discriminator_loss={:.4f} generator_loss={:.4f}"
                .format(epoch, recon, discr, gener))
    path = '/home/john/Desktop/Dissertation/Dataset1/Pretrained Weights/AAE.pt'
    torch.save(aae.state_dict(), path)

