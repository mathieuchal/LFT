import os
import copy
import math
import torch
import scipy
import time
import numpy as np
# import tqdm

import torch.nn as nn
from scipy.integrate import solve_ivp
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from Datasets import Meta_dataset
from utils import get_loss, prepare_target, get_config, mse, rmse
from Transducer_utils import get_model, generate_mask

### Get XP change
XP_type = 'ADR'  # Must be an element of ['ADR','Burgers','MNIST']
name = XP_type + '_test'
cfg = get_config(XP_type)

# Generate path for experiment
if not os.path.exists(cfg['save_path']):
    os.makedirs(cfg['save_path'])

# Get data
dataset = Meta_dataset(cfg['data_path'], XP_type)

# Get training device and config
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Model init
model = get_model(XP_type, device)
optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg['reduce_n_step']),
                                            gamma=float(cfg['reduce_lr_coeff']))
loss_function = get_loss(XP_type)

# Accounting
model.train()  # turn on train mode
total_loss = 0.
losses = []
start_time = time.time()
best_val_loss = float('inf')
best_model = None
losses = []
acc = [0]
n_func_pred = int(cfg['n_func_pred'])
N = int(cfg['meta_batch_size'])
n = int(cfg['bs'])

# Meta training loop
for i in range(cfg['n_iter']):
    v, u = dataset.meta_batch(N, n, device)
    u_target, u_ = prepare_target(u, n_func_pred, device, xp_type=XP_type)
    mask = generate_mask(n, n_func_pred, device).unsqueeze(0).repeat(N, 1, 1).to(device)
    out = model(v, u_, mask)
    loss = loss_function(out[:, :n_func_pred], u_target[:, :n_func_pred])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip']))
    optimizer.step()
    scheduler.step()

    total_loss += loss.item()
    losses.append(loss.item())

    if i % int(cfg['log_interval']) == 0:
        elapsed = time.time() - start_time
        scheduler.step()
        if XP_type == 'MNIST':
            acc.append(((u[:, :n_func_pred].flatten().detach().cpu() == torch.argmax(out[:, :n_func_pred],
                                                                                     dim=-1).flatten().detach().cpu()).sum() / (
                                    n_func_pred * N)).item())
            print(f'| Iter {i:3d} | time: {elapsed:5.2f}s | acc: {acc[-1]:3.3f} | loss: {losses[-1]:3.3f} |  ')
        elif XP_type == 'ADR':
            with torch.no_grad():
                out_sliced = out[:, :n_func_pred].flatten().detach().cpu()
                u_sliced = u_target[:, :n_func_pred].flatten().detach().cpu()
                RMSE = rmse(out_sliced, u_sliced)
                MSE = mse(out_sliced, u_sliced)
            print(
                f'| Iter {i:3d} | time: {elapsed:5.2f}s  | Loss : {loss:3.6f} | MSE: {MSE:3.6f} | RMSE: {RMSE:3.6f} |  ')

    if (i % int(cfg['save_interval']) == 0) or (i == cfg['n_iter'] - 1):
        torch.save(model.state_dict(), cfg['save_path'] + name + '.pt')
        np.save(cfg['save_path'] + name + '_loss.npy', np.array(losses))



