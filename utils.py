import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp
import scipy.fftpack
from scipy.interpolate import interp2d
from torch.utils.data import DataLoader, Dataset
import torch
import yaml

<<<<<<< HEAD
def prepare_target(target, n_func_pred, device, xp_type='ADR'):
    
    if xp_type=='ADR':
        t_ = target.clone()
        t_[:,:n_func_pred] = 0.
        t = target
        
    elif xp_type=='MNIST':
        t = torch.nn.functional.one_hot(target, num_classes=10)
        t_ = t.clone()
        t_[:,:n_func_pred] = 0.
        
=======

def prepare_target(target, n_func_pred, device, xp_type='ADR'):
    if xp_type == 'ADR':
        t_ = target.clone()
        t_[:, :n_func_pred] = 0.
        t = target

    elif xp_type == 'MNIST':
        t = torch.nn.functional.one_hot(target, num_classes=10)
        t_ = t.clone()
        t_[:, :n_func_pred] = 0.

>>>>>>> eeca32a21b016bf0af68c229ad6355c21c35b32d
    return t.to(device).float(), t_.to(device).float()


def sample_batch_ADR(s, n, num):
    perm = torch.randperm(n)[:num]
    idx_u = 0
    idx_s = -1
    x = s[perm,:, idx_u]
    y = s[perm,:, idx_s]
    return x, y


def sample_batch_Burgers(s, n, num):
    perm = torch.randperm(200)[:n]
    idx_s = torch.randint(1, 21, (1,)).item()
    idx_u = idx_s - torch.randint(0, idx_s, (1,)).item() - 1
    # idx_s = np.min([idx_u+torch.randint(1,20,(1,)).item(),100])
    u_ = torch.permute(s[perm, idx_u].float(), (0, -1, 1, 2))
    s_ = torch.permute(s[perm, idx_s].float(), (0, -1, 1, 2))
    # s_bis = s[idx_sbis,perm].float()
    return u_, s_  # , s_bis

<<<<<<< HEAD
=======

>>>>>>> eeca32a21b016bf0af68c229ad6355c21c35b32d
def sample_batch_MNIST(s, n, num):
    perm = torch.randperm(n)[:num]
    idx_u = 0
    idx_s = -1
    x = s[perm, :, idx_u]
    y = s[perm, :, idx_s]
    return x, y

<<<<<<< HEAD
def generate_batch(meta_bs):
    b_, t_ = [],[]
    for i, b in enumerate(train_loader):
        ba,target = generate_sample(b,bs)
        b_.append(ba)
        t_.append(target)
        if i==meta_bs-1:
            return torch.stack(b_,dim=0),torch.stack(t_,dim=0)
=======

def generate_batch(meta_bs):
    b_, t_ = [], []
    for i, b in enumerate(train_loader):
        ba, target = generate_sample(b, bs)
        b_.append(ba)
        t_.append(target)
        if i == meta_bs - 1:
            return torch.stack(b_, dim=0), torch.stack(t_, dim=0)
>>>>>>> eeca32a21b016bf0af68c229ad6355c21c35b32d


def get_sampler(XP_type):
    if XP_type == 'ADR':
        return sample_batch_ADR
    elif XP_type == 'Burgers':
        return sample_batch_Burgers
    elif XP_type == 'MNIST':
        return sample_batch_Burgers
<<<<<<< HEAD
        
=======

>>>>>>> eeca32a21b016bf0af68c229ad6355c21c35b32d

def get_loss(XP_type):
    if XP_type in ['ADR', 'Burgers', 'MNIST']:
        def loss_function(y, y_hat):
            return torch.pow(y - y_hat, 2).mean()

<<<<<<< HEAD
=======
        return loss_function


>>>>>>> eeca32a21b016bf0af68c229ad6355c21c35b32d
def loss_function(y, y_hat):
    return torch.pow(y - y_hat, 2).mean()  # /n_func_pred


def rmse(y_hat, y_gt):
    return torch.mean(torch.pow(y_hat - y_gt, 2).sum(-1) / torch.pow(y_gt, 2).sum(-1))


def mse(y_hat, y_gt):
    return torch.pow(y_hat - y_gt, 2).sum(-1).mean()


#### MNIST utils

def shuffle(x):
    # ps = np.random.randint(2)+1
    ps = int(np.random.choice([1, 2, 4, 7]))
    # ps = 1
    idx = torch.randperm(int(784 / (ps * ps)))
    # if torch.rand(1).item()>0.5:
    #    x = x.transpose(-2,-1)
    u = F.unfold(x, kernel_size=ps, stride=ps, padding=0)[:, :, idx]
    f = F.fold(u, x.shape[-2:], kernel_size=ps, stride=ps, padding=0)
    return f


def shuffle_label(y):
    idx = np.random.permutation(10)
    y_ = torch.tensor([idx[a] for a in y])
    # y = torch.nn.functional.one_hot(y_, num_classes=10)
    return y_


def generate_sample(batch, bs):
    x = shuffle(batch[0])  # .reshape(bs,-1)
    # x = project(batch[0].reshape(bs,-1))
    y = shuffle_label(batch[1])
    # print(x.shape,x.reshape(bs,-1).shape,batch[1].shape)
    return x.reshape(bs,-1), y  # torch.cat([x.reshape(bs,-1),y.unsqueeze(-1)/10],dim=-1)# x.reshape(bs,-1), y #torch.cat([x.reshape(bs,-1),y],dim=-1), y


def generate_sample_noperm(batch, bs):
    # idx = torch.randperm(784)
    b_ = batch[0].reshape(bs, -1)  # [:,idx]
    return b_, batch[1]


# def generate_sample_noperm(batch,bs):
#     #idx = torch.randperm(784)
#     b_ = batch[0].reshape(bs,-1)#[:,idx]
#     return torch.cat([b_,batch[1].unsqueeze(-1)/10],dim=-1)

def generate_batch(meta_bs):
    b_, t_ = [], []
    for i, b in enumerate(train_loader):
        ba, target = generate_sample(b, bs)
        b_.append(ba)
        t_.append(target)
        if i == meta_bs - 1:
            return torch.stack(b_, dim=0), torch.stack(t_, dim=0)
        
def get_config(XP_type):
    if XP_type=='ADR':
        config_path = './config_ADR.yaml'
    elif XP_type=='Burgers':
        config_path = './config_Burgers.yaml'
    elif XP_type=='MNIST':
        config_path = './config_MNIST.yaml'
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def get_config(XP_type):
    if XP_type == 'ADR':
        config_path = './config_ADR.yaml'
    elif XP_type == 'Burgers':
        config_path = './config_Burgers.yaml'
    elif XP_type == 'MNIST':
        config_path = './config_MNIST.yaml'
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


