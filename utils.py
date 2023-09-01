import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp
import scipy.fftpack
from scipy.interpolate import interp2d
from torch.utils.data import DataLoader, Dataset


class Dataset_ADR(Dataset):
    def __init__(self, source_file='Train_data_N=500_T=1.npz'):
        data = np.load(source_file)
        # sensors, y_train, coeffs = Data['sensors'], Data['y_train'], Data['coeffs']
        self.y_train = data['y_train'].reshape(500, 100, 100, 100)
        # self.sensors = Data['sensors']
        # self.coeffs  = Data['coeffs']

    def torch_wrapper(self, array):
        tensor = torch.from_numpy(array).to('cuda').float()
        return tensor

    def __getitem__(self, index):
        y = self.torch_wrapper(self.y_train[index])
        # u = self.torch_wrapper(self.sensors[index])
        # c = self.coeffs[index]
        return y  # ,c#, u#, c

    def __len__(self):
        return len(self.y_train)

class Dataset_Burger(Dataset):
    def __init__(self, source_file='Train_data_N=500_T=1.npz'):
        self.y_train = np.load(source_file)

    def torch_wrapper(self, array):
        tensor = torch.from_numpy(array).to('cuda').float()
        return tensor

    def __getitem__(self, index):
        y = self.torch_wrapper(self.y_train[index])

        return y

    def __len__(self):
        return len(self.y_train)


def sample_batch_ADR(s, n, num):
    perm = torch.randperm(n)[:num]
    idx_u = 0
    idx_s = -1
    x = s[perm, :, idx_u]
    y = s[perm, :, idx_s]
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

def get_sampler(XP_type):
    if XP_type == 'ADR':
        return sample_batch_ADR
    elif XP_type == 'Burgers':
        return sample_batch_Burgers



def loss_function(y, y_hat):
    return torch.pow(y - y_hat, 2).mean()  # /n_func_pred


def rmse(y_hat, y_gt):
    return torch.mean(torch.pow(y_hat - y_gt, 2).sum(-1) / torch.pow(y_gt, 2).sum(-1))


def mse(y_hat, y_gt):
    return torch.pow(y_hat - y_gt, 2).sum(-1).mean()