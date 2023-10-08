from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import torch


class Dataset_ADR(Dataset):
    def __init__(self, source_file='./Train_data_N=500_T=1.npz'):
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

class Dataset_Burgers(Dataset):
    def __init__(self, source_file='./Train_data_N=500_T=1.npz'):
        self.y_train = np.load(source_file)

    def torch_wrapper(self, array):
        tensor = torch.from_numpy(array).to('cuda').float()
        return tensor

    def __getitem__(self, index):
        y = self.torch_wrapper(self.y_train[index])

        return y

    def __len__(self):
        return len(self.y_train)



def Meta_dataset(path, type, batch_size=100):
    if type == 'ADR':
        return Dataset_ADR(path +'/ADR_l=0.2.npz')
    elif type == 'Burgers':
        return Dataset_Burgers(path +'/Burgers.npz')
    elif type == 'MNIST_like':
        dataset = datasets.MNIST('/media/data_cifs_lrs/projects/prj_control/mathieu/unsupervised/MNIST/', train=True,
                                 transform=transforms.ToTensor(), download=False)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader


