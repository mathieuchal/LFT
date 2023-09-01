import os
import math
impor time
import numpy as np
import torch
import torch.nn as nn
import scipy


import matplotlib.pyplot as plt
from scipy import linalg, interpolate
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import scipy
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp
import scipy.fftpack
from scipy.interpolate import interp2d
from torch.utils.data import DataLoader, Dataset

class MyDataset_full(Dataset):
    def __init__(self, source_file='Train_data_N=500_T=1.npz'):
        data = np.load(source_file)
        #sensors, y_train, coeffs = data['sensors'], data['y_train'], data['coeffs']
        self.y_train = data['y_train'].reshape(500,100,100,100)
        #self.sensors = data['sensors']
        #self.coeffs  = data['coeffs']

    def torch_wrapper(self, array):
        tensor = torch.from_numpy(array).to('cuda').float()
        return tensor

    def __getitem__(self, index):
        y = self.torch_wrapper(self.y_train[index])
        #u = self.torch_wrapper(self.sensors[index])
        #c = self.coeffs[index]
        return y#,c#, u#, c

    def __len__(self):
        return len(self.y_train)

path = '/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/PDE_reg/ADR_l=0.2.npz'#{}/'.format(params['version'])
dataset = MyDataset_full(path)

# path = '/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/PDE_reg/ADR_l=0.2_bs=500_test.npz'#{}/'.format(params['version'])
# dataset = MyDataset_full(path)
# dataloader = DataLoader(dataset, batch_size=500, shuffle=False)

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange





class FFT1(nn.Module):
    def __init__(self, modes):
        super().__init__()
        self.modes=modes
    def forward(self, x):
        return torch.view_as_real(torch.fft.rfft(x,dim=-1,norm="ortho")[:,:,:self.modes]).flatten(start_dim=-2)

class IFFT1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.view_as_complex(x.reshape(x.size(0),-1,2)) #   complex X up/down X num_vectors
        return torch.fft.irfft(x,dim=-1,n=(100),norm="ortho")


class Attention_transduction_Fourier(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,modes=12):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        #self.scale = dim_head ** -0.5
        self.modes1 = modes
        self.scale = nn.Parameter(torch.ones(1))
        self.attend = nn.Softmax(dim=-1)
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None):
        # print(x.shape)
        x1, x2 = torch.chunk(x, 2, dim=0)
        q = rearrange(self.to_q(x1), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(x1), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x2), 'b n (h d) -> b h n d', h=self.heads)
        if mask == None:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + mask.unsqueeze(0).unsqueeze(0).repeat(1,self.heads,1, 1)
            #dots = torch.pow(q.unsqueeze(-2) - k.unsqueeze(-3),2).sum(-1) * self.scale + mask.unsqueeze(0).unsqueeze(0).repeat(1,self.heads,1, 1)
            #dots = torch.pow(q.unsqueeze(-2) - k.unsqueeze(-3),2).sum(-1) * self.scale * torch.exp(mask.unsqueeze(0).unsqueeze(0).repeat(1,self.heads,1, 1))

        # print(dots.shape,mask.shape)
        attn = self.attend(dots)
        #attn = torch.nn.functional.normalize(dots,p=2.0,dim=-1)
        out = torch.matmul(attn, v)
        # print(out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return torch.cat([x1, x2 + self.to_out(out)], dim=0)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x) + x


class Trans_Fourier(nn.Module):
    def __init__(self, modes, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.dim = modes*2
        self.modes=modes
        self.layers = nn.ModuleList([])
        self.fft = FFT1(modes)
        self.ifft = IFFT1()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_transduction_Fourier(self.dim, heads=heads, dim_head=dim_head,modes=modes),
                FeedForward(self.dim, mlp_dim)
            ]))

    def forward(self, xy, mask):
        #inp = torch.view_as_real(torch.fft.rfft(xy, dim=-1)[:, :, :self.modes]).reshape(2,xy.size(1),-1)
        inp = self.fft(xy)

        for attn, ff in self.layers:
            for i in range(1):
                inp = ff(inp)
                inp = attn(inp, mask)
        #inp = torch.view_as_complex(inp.reshape(2,xy.size(1),self.modes,2))
        #inp = torch.fft.irfft(inp, dim=-1,n=xy.size(2))

        return self.ifft(inp[-1])


class Transducer(nn.Module):
    def __init__(self, *, modes, depth, heads, mlp_dim, dim_head=32):
        super().__init__()
        #self.encoder = nn.Linear(2, 1)
        #self.encoder_target = nn.Linear(2, 1)
        self.transformer = Trans_Fourier(modes, depth, heads, dim_head, mlp_dim)
        #self.linear_head = nn.Linear(100, 100)

    def forward(self, x, y, mask):
        # x = torch.stack([x,torch.linspace(0,1,100).unsqueeze(0).repeat(x.shape[0],1).cuda()],dim=-1)
        # y = torch.stack([y, torch.linspace(0,1,100).unsqueeze(0).repeat(y.shape[0],1).cuda()],dim=-1)
        # x = self.encoder(x)
        # y = self.encoder_target(y)
        xy = torch.stack([x,y],dim=0)#.squeeze(-1)
        x_out = self.transformer(xy, mask)
        return x_out #self.linear_head(x_out)




name = 'ADR_v0_u1_30_100ex_8layers_L2kernel'
path = '/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/ADR_regression/'
if not os.path.exists(path):
    os.makedirs(path)

# path = '/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/PDE_reg/NEW_VITLIKE_RESNET_Multiphysics_27_10'
# if not os.path.exists(path):
#     os.makedirs(path)

model = Transducer(
    modes=50,
    depth=8,
    heads=32,
    mlp_dim=100,
    dim_head=16,
).cuda()


bs = 100
n = bs
n_func_pred = 10
n_iter = 200000
lr = 2e-4  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5)



def generate_square_subsequent_mask(sz: int, n_func_pred: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mat = torch.zeros(sz, sz)
    mat[:, :n_func_pred] = float('-inf')
    #mat[range(n_func_pred), range(n_func_pred)] = 0
    # mat[:n_func_pred,:n_func_pred]=(torch.ones(n_func_pred,n_func_pred)-torch.eye(n_func_pred,n_func_pred))* float('-inf')
    return mat

def sample_batch(s,n,num):
    perm = torch.randperm(n)[:num]
    idx_u = 0
    idx_s = -1
    x = s[perm,:,idx_u]
    y = s[perm,:,idx_s]
    return x, y

def loss_function(y,y_hat):
    return torch.pow(y-y_hat,2).mean()#/n_func_pred

def rmse(y_hat,y_gt):
    return torch.mean(torch.pow(y_hat - y_gt, 2).sum(-1)/ torch.pow(y_gt, 2).sum(-1))

def mse(y_hat,y_gt):
    return torch.pow(y_hat-y_gt,2).sum(-1).mean()


model.train()  # turn on train mode
total_loss = 0.
losses = []
log_interval = 10
save_interval = 1000
start_time = time.time()

for i in range(n_iter):
    k = np.random.choice(len(dataset))
    num = np.random.randint(30,101)
    src_mask = generate_square_subsequent_mask(num, n_func_pred).cuda()
    x, y = sample_batch(dataset[k],n,num)
    y_ = y.clone().cuda()
    y_[:n_func_pred] = 0
    output = model(x, y_, src_mask)[:n_func_pred]

    loss = loss_function(output, y[:n_func_pred])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
    optimizer.step()
    scheduler.step()

    total_loss += loss.item()
    losses.append(loss.item())
    with torch.no_grad():
        out = output.flatten().detach().cpu()
        y_hat = y[:n_func_pred].flatten().detach().cpu()
        RMSE = rmse(out, y_hat)
        MSE  = mse(out, y_hat)
    if i % log_interval == 0 and i > 0:
        lr = scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - start_time) * 1000 / log_interval
        cur_loss = total_loss / log_interval
        # ppl = math.exp(cur_loss)
        print(f'| iter {i:3d} | {i:5d}/{n_iter:5d} n_iter | '
              f'lr {lr:2.5f} | ms/batch {ms_per_batch:5.3f} | '
              f'loss {cur_loss:5.6f} | RMSE {RMSE:5.6f} | MSE {MSE:5.6f}')
        start_time = time.time()
        total_loss=0
    if i % save_interval == 0 and i > 0:
        np.save(path + name + '_loss.npy',np.array(losses))
        torch.save(model.state_dict(), path + name + '.pt')


torch.save(model.state_dict(), path + name + '.pt')
np.save(path + name + '_loss.npy',np.array(losses))
