from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn.functional as F



def shuffle(x):
    #ps = np.random.randint(2)+1
    ps = int(np.random.choice([1,2,4,7]))
    #ps = 1
    idx = torch.randperm(int(784/(ps*ps)))
    #if torch.rand(1).item()>0.5:
    #    x = x.transpose(-2,-1)
    u = F.unfold(x, kernel_size=ps, stride=ps, padding=0)[:,:,idx]
    f = F.fold(u, x.shape[-2:], kernel_size=ps, stride=ps, padding=0)
    return f

def project(x):
    with torch.no_grad():
        A = torch.empty(1,784,784,device='cuda').normal_(mean=0,std=np.sqrt(1/784)).repeat(x.shape[0],1,1)
    return torch.bmm(A,x.unsqueeze(-1).cuda()).squeeze(-1)

def shuffle_label(y):
    idx = np.random.permutation(10)
    y_ = torch.tensor([idx[a] for a in y])
    #y = torch.nn.functional.one_hot(y_, num_classes=10)
    return y_

def generate_sample(b,t):
    bs = b.shape[0]
    x = shuffle(b.unsqueeze(1)).reshape(bs,-1)
    #x = project(b)
    y = shuffle_label(t)
    #print(x.shape,x.reshape(bs,-1).shape,batch[1].shape)
    return x, y #torch.cat([x.reshape(bs,-1),y.unsqueeze(-1)/10],dim=-1)# x.reshape(bs,-1), y #torch.cat([x.reshape(bs,-1),y],dim=-1), y

def generate_sample_noperm(batch,bs):
    #idx = torch.randperm(784)
    b_ = batch[0].reshape(bs,-1)#[:,idx]
    return b_, batch[1]






class Dataset_ADR(Dataset):
    def __init__(self, source_file='./Train_data_N=500_T=1.npz'):
        data = np.load(source_file)
        # sensors, y_train, coeffs = Data['sensors'], Data['y_train'], Data['coeffs']
        self.y_train = data['y_train'].reshape(500, 100, 100, 100)
        self.data_lenght = len(self.y_train)
        # self.sensors = Data['sensors']
        # self.coeffs  = Data['coeffs']

    def torch_wrapper(self, array, device):
        tensor = torch.from_numpy(array).to(device).float()
        return tensor
    
    def sample_batch(self,data,n,device):
        idx = torch.randperm(len(data))[:n]
        t_target = np.random.randint(30,100)
        x, y = data[idx,:,0],data[idx,:,t_target]
        return self.torch_wrapper(x,device), self.torch_wrapper(y,device)

    def meta_batch(self,N,n,device):
        b_, t_ = [],[]
        for k in range(N):
            c = np.random.choice(range(self.data_lenght))
            b, target = self.sample_batch(self.y_train[c],n,device)
            b_.append(b)
            t_.append(target) 
        return torch.stack(b_,dim=0),torch.stack(t_,dim=0)   

    
    def __getitem__(self, index, device):
        return self.y_train[index] 

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
    
    
class Dataset_MNIST_Like(object):
    """
    Class for MNIST_like data
    """
    def __init__(self,dataset):
        self.data_lenght = len(dataset.data)
        self.data = dataset.data
        self.target = dataset.targets
        
    def sample_batch(self,n,device):
        idx = torch.randperm(self.data_lenght)[:n]
        d, t = generate_sample(self.data[idx].float(),self.target[idx])
        return d.reshape(n,-1).to(device), t.to(device)
    
    def meta_batch(self,N,n,device):
        b_, t_ = [],[]
        for k in range(N):
            b, target = self.sample_batch(n,device)
            b_.append(b)
            t_.append(target) 
        return torch.stack(b_,dim=0),torch.stack(t_,dim=0)
    
    def __len__(self):
        return self.data_lenght



def Meta_dataset(path, type_, batch_size=100):
    if type_ == 'ADR':
        return Dataset_ADR(path +'ADR_l=0.2.npz')
    elif type_ == 'Burgers':
        return Dataset_Burgers(path +'Burgers.npz')
    elif type_ == 'MNIST':
        dataset = datasets.MNIST(path, train=True, transform=transforms.ToTensor(), download=False)
    return Dataset_MNIST_Like(dataset)


