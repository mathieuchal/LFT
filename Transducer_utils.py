


import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


class FFT1(nn.Module):
    def __init__(self, modes):
        super().__init__()
        self.modes = modes

    def forward(self, x):
        return torch.view_as_real(torch.fft.rfft(x, dim=-1, norm="ortho")[:, :, :self.modes]).flatten(start_dim=-2)

class IFFT1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.view_as_complex(x.reshape(x.size(0), -1, 2))  # complex X up/down X num_vectors
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
        x = torch.view_as_complex(x.reshape(x.size(0), 2, 2, d_s, 2))  # torch.stack(torch.chunk(x, 2, dim=-1),dim=-1))
        f = torch.zeros((x.size(0), 2, 64, 33), dtype=torch.complex64).cuda()
        f[:, :, :d, :d] = x[:, :, 0, :].reshape(-1, 2, d, d)
        f[:, :, -d:, :d] = x[:, :, 1, :].reshape(-1, 2, d, d)
        return torch.fft.irfft2(f, dim=(-2, -1), s=(64, 64), norm="ortho")



class Transduction_layer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, modes=12):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.modes1 = modes
        self.scale = nn.Parameter(torch.ones(1))
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None):
        x1, x2 = torch.chunk(x, 2, dim=0)
        q = rearrange(self.to_q(x1), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(x1), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x2), 'b n (h d) -> b h n d', h=self.heads)
        if mask is None:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + mask.unsqueeze(0).unsqueeze(0).repeat(1,self.heads,1, 1)

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
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


class Transducer(nn.Module):
    def __init__(self, modes, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.dim = modes * 2
        self.modes = modes
        self.layers = nn.ModuleList([])
        self.fft = FFT1(modes)
        self.ifft = IFFT1()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_transduction_Fourier(self.dim, heads=heads, dim_head=dim_head, modes=modes),
                FeedForward(self.dim, mlp_dim)
            ]))

    def forward(self, x, y, mask):
        # inp = torch.view_as_real(torch.fft.rfft(xy, dim=-1)[:, :, :self.modes]).reshape(2,xy.size(1),-1)
        xy = torch.stack([x, y], dim=0)
        inp = self.fft(xy)

        for attn, ff in self.layers:
            for i in range(1):
                inp = ff(inp)
                inp = attn(inp, mask)
                # inp = torch.view_as_complex(inp.reshape(2,xy.size(1),self.modes,2))
        # inp = torch.fft.irfft(inp, dim=-1,n=xy.size(2))

        return self.ifft(inp[-1])

def generate_mask(sz: int, n_func_pred: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mat = torch.zeros(sz, sz)
    mat[:, :n_func_pred] = float('-inf')
    return mat

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data, gain=0.001)
        if module.bias is not None:
            module.bias.data.zero_()

def get_model(XP_type):
    if XP_type == 'ADR':
        model = Transducer(modes=50,
        depth=8,
        heads=32,
        mlp_dim=100,
        dim_head=16)
    elif XP_type == 'Burgers':
        model = Transducer(
            dim=1152,
            depth=10,
            heads=64,
            mlp_dim=1200,
            dim_head=8,
            d=12
        )

    return model.cuda()

