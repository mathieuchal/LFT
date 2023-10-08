import os
import copy
import math
import torch
import scipy
import time
import numpy as np

import torch.nn as nn
from scipy.integrate import solve_ivp
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange



# Generate path for experiment
XP_type = 'MNIST'
path = './Data/'
name = XP_type + '_test'

if not os.path.exists(path):
    os.makedirs(path)

# Get data
dataset = Meta_dataset(path, XP_type)

# Get training device and config
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


# Model init
model = get_model(XP_type)
sample_batch = get_sampler(XP_type, cfg)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['reduce_n_step'], gamma=cfg['reduce_coeff'])
train_loader = Meta_dataset(path, type, batch_size=cfg['bs'])
loss_function = get_loss(XP_type)

# Accounting
model.train()  # turn on train mode
total_loss = 0.
losses = []
start_time = time.time()


best_val_loss = float('inf')
best_model = None
device = 'cuda'
losses = []
acc = []

for epoch in range(1, cfg['epochs'] + 1):
    epoch_start_time = time.time()
    loss, target, inp, out = train_epoch(model)
    losses += loss
    elapsed = time.time() - epoch_start_time
    print('-' * 70)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ')
    print('-' * 70)
    scheduler.step()
    acc.append(((target[:, :n_func_pred].flatten() == torch.argmax(out[:, :n_func_pred],
                                                                   dim=-1).flatten().detach().cpu()).sum() / (
                            n_func_pred * meta_bs)).item())

    if (epoch % cfg['save_interval'] == 0) or (epoch == cfg['epochs']):
        torch.save(model.state_dict(),
                   '/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/VR/' + name + '.pt')
        np.save('/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/VR/' + name + '_loss.npy',
                np.array(losses))