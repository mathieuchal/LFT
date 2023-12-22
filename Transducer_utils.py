


import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


class FFT1(nn.Module):
    def __init__(self, modes):
        super().__init__()
        self.modes = modes

    def forward(self, x):
        return torch.view_as_real(torch.fft.rfft(x, dim=-1, norm="ortho")[:,:,:,:self.modes]).flatten(start_dim=-2)

class IFFT1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.view_as_complex(x.reshape(x.size(0),x.size(1), -1, 2))  # complex X up/down X num_vectors
        return torch.fft.irfft(x, dim=-1, n=(100), norm="ortho")

class FFT2(nn.Module):
    def __init__(self, d=10):
        super().__init__()
        self.d = d

    def forward(self, x):
        d = self.d
        f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        coeffs = torch.stack([f[:, :, :, :d, :d].flatten(3), f[:, :, :, -d:, :d].flatten(3)], dim=-2)
        return torch.view_as_real(coeffs).flatten(2)

class IFFT2(nn.Module):
    def __init__(self, d=10):
        super().__init__()
        self.d = d

    def forward(self, x):
        d = self.d
        d_s = int(d * d)
        x = torch.view_as_complex(x.reshape(x.size(0), 2, 2, d_s, 2)) 
        f = torch.zeros((x.size(0), 2, 64, 33), dtype=torch.complex64).cuda()
        f[:, :, :d, :d] = x[:, :, 0, :].reshape(-1, 2, d, d)
        f[:, :, -d:, :d] = x[:, :, 1, :].reshape(-1, 2, d, d)
        return torch.fft.irfft2(f, dim=(-2, -1), s=(64, 64), norm="ortho")

class Transduction_layer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, transform_u=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        #self.modes1 = modes
        self.scale = nn.Parameter(torch.ones(1),requires_grad=True)
        self.kernel = nn.Softmax(dim=-1)
        self.proj_V1 = nn.Linear(dim, inner_dim, bias=False)
        self.proj_V2 = nn.Linear(dim, inner_dim, bias=False)
        self.proj_U = nn.Linear(dim, inner_dim, bias=False)
        #self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.transform_u = transform_u
        if self.transform_u:
            self.proj_U = nn.Linear(dim, inner_dim, bias=False)
            self.to_out = nn.Sequential(Rearrange('b h n d -> b n (h d)'), nn.Linear(inner_dim, dim, bias=False))
        else:
            self.proj_U = nn.Identity()
            #self.to_out = nn.Identity()
            #self.to_out = lambda x : x.sum(axis=1)
            self.to_out = nn.Sequential(Rearrange('b h n d -> b n (h d)'), nn.Linear(heads*10, 10, bias=False))

    def forward(self, inp , mask=None):
        x,y = inp
        q = rearrange(self.proj_V1(x), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.proj_V2(x), 'b n (h d) -> b h n d', h=self.heads)
        if self.transform_u:
            v = rearrange(self.proj_U(y), 'b n (h d) -> b h n d', h=self.heads)
        else:
            v = y.unsqueeze(1).repeat(1, self.heads, 1, 1)
        if mask is None:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + mask.unsqueeze(1).repeat(1,self.heads,1, 1)

        k_dots = self.kernel(dots)
        out = torch.matmul(k_dots, v)
        #out = rearrange(out, 'b h n d -> b n (h d)')
        return (x, y + self.to_out(out))

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
    
class FeedForward_id_u(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net_v = FeedForward(dim, hidden_dim)

    def forward(self, inp):
        return (self.net_v(inp[0]),inp[1])
    
class FeedForward_shared(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net_vu = FeedForward(dim, hidden_dim)

    def forward(self, inp):
        return (self.net_vu(inp[0]),self.net_vu(inp[1]))
    
class FeedForward_unshared(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net_u = FeedForward(dim, hidden_dim)
        self.net_v = FeedForward(dim, hidden_dim)

    def forward(self, inp):
        return (self.net_v(inp[0]),self.net_u(inp[1]))


class Transducer(nn.Module):
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, freq_transform=True, fflayer=False, transform_u=False):
        super().__init__()
        
        self.dim = dim #modes * 2
        self.heads = heads
        self.dim_head = dim_head
        #self.modes = modes
        self.layers = nn.ModuleList([])
        
        self.freq_transform = freq_transform
        if self.freq_transform:
            self.fft = FFT1(int(dim/2))
            self.ifft = IFFT1()
            
        if fflayer=='both':
            FF = FeedForward_unshared
        elif fflayer=='shared':
            FF = FeedForward_shared
        elif fflayer=='no_output':
            FF = FeedForward_id_u
            
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                FF(self.dim, mlp_dim),
                Transduction_layer(self.dim, heads=self.heads, dim_head=self.dim_head, transform_u=transform_u)
            ]))

    def forward(self, x, y, mask):
        inp = (x,y)
        if self.freq_transform:
            inp = torch.stack([x, y], dim=0)
            inp = self.fft(inp)
        
        for ff, attn in self.layers:
            inp = ff(inp)
            inp = attn(inp, mask)
                
        if self.freq_transform:
            inp = self.ifft(inp[1])
        else:
            inp = inp[1]
            
        return inp

def generate_mask(sz: int, n_func_pred: int, device: str):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mat = torch.zeros(sz, sz)
    mat[:, :n_func_pred] = float('-inf')
    return mat.to(device)


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data, gain=0.001)
        if module.bias is not None:
            module.bias.data.zero_()

def get_model(XP_type, device):
    if XP_type == 'ADR':
        model = Transducer(dim=100,
                           depth=8,
                           heads=32,
                           mlp_dim=100,
                           dim_head=16,
                           freq_transform=True,
                           fflayer='shared',
                           transform_u=True)
    
    elif XP_type == 'Burgers':
        model = Transducer(dim=100,
                           depth=5,
                           heads=16,
                           mlp_dim=100,
                           dim_head=32,
                           freq_transform=True,
                           fflayer='shared',
                           transform_u=True)
    elif XP_type == 'MNIST':
        model = Transducer(dim=784,
                           depth=2,
                           heads=32,
                           mlp_dim=512,
                           dim_head=32,
                           freq_transform=False,
                           fflayer='no_output',
                           transform_u=False)
    return model.to(device)


    
    
#class Transducer(nn.Module):
#    def __init__(self, *, dim_in, dim, dimout, depth, heads, mlp_dim, dim_head=32, modes=50):
#        super().__init__()
#        self.transformer = Trans(dim, depth, heads, dim_head, mlp_dim)
#        self.linear_head = nn.Linear(dim_in, dim)

#    def forward(self, x, y, mask):
#        xy = (x, y)
#        x_out = self.transformer(xy, mask)[1]
#        return x_out 

