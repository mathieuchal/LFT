import os
import math
import time
import numpy as np
import torch
import torch.nn as nn

from Datasets import Meta_dataset
from utils import sample_batch, loss_function, rmse, mse

from Transducer_utils import Transducer, generate_mask


# Generate path for experiment
XP_type = 'ADR'
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
sample_batch = get_sampler(XP_type)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['reduce_n_step'], gamma=cfg['reduce_coeff'])


# Accounting
model.train()  # turn on train mode
total_loss = 0.
losses = []
log_interval = 10
save_interval = 1000
start_time = time.time()


# Meta-training loop
for i in range(cfg['n_iter']):
    # Sampling and preparing dataset
    k = np.random.choice(len(dataset))
    num = np.random.randint(cfg['N_min'], cfg['N_max'])
    src_mask = generate_mask(num, cfg['n_func_pred']).to(device)
    x, y = sample_batch(dataset[k], cfg['n'], num)
    y_ = y.clone().to(device)
    y_[:n_func_pred] = 0.

    #Inference and loss
    output = model(x, y_, src_mask)[:cfg['n_func_pred']]
    loss = loss_function(output, y[:cfg['n_func_pred']])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()
    losses.append(loss.item())

    #Compute stats
    with torch.no_grad():
        out = output.flatten().detach().cpu()
        y_hat = y[:cfg['n_func_pred']].flatten().detach().cpu()
        RMSE = rmse(out, y_hat)
        MSE = mse(out, y_hat)
    # Logs
    if i % log_interval == 0 and i > 0:
        lr = scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - start_time) * 1000 / log_interval
        cur_loss = total_loss / log_interval
        print(f'| iter {i:3d} | {i:5d}/{n_iter:5d} n_iter | '
              f'lr {lr:2.5f} | ms/batch {ms_per_batch:5.3f} | '
              f'loss {cur_loss:5.6f} | RMSE {RMSE:5.6f} | MSE {MSE:5.6f}')
        start_time = time.time()
        total_loss = 0
    # Saving
    if (i % cfg['save_interval'] == 0 and i > 0) or (i==n_iter-1):
        np.save(path + name + '_loss.npy', np.array(losses))
        torch.save(model.state_dict(), path + name + '.pt')

