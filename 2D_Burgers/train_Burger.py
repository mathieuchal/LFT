

















################################################################
# training and evaluation
################################################################

# name = 'Navier_stokes_NUat0.5max_sharedff_3smax_fullu_warmstart_NOselftattention_sharedKeyquey_32heads_NEWmodes25_fullres_4layers_NEWGRFAT8'
name = 'NEW_Burger_u20min_s20max_input_NUfrom0.1_STARTGRF6_NOWARMSTART_12times12_MLP1200_90examplessampled_learnablescaleQKVDIFFTRUE_10layers_64heads_dim8_SINGLEFF_NOWARMUP_NOCLAMP_REDUCEDLR1e_4_NEWDATA_NOAUGMENTATION'  # GOOD_spatialattentionbefore_sharedffnets_bs10_FFT2_selfattentionok'#pressure_to_temp'
model.train()  # turn on train mode
total_loss = 0.
losses = []
save_interval = 1000
log_interval = 1
start_time = time.time()
num_batches = 0

# idx = np.load('/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/PHIFLOW/trainset_BURGER_NU_leq05_GRFALPHA_at_6TRUE_nb_datasets=50_bs_per_dataset=200_zscored_NULIST.npy') > 0.1
# indices = np.where(idx)[0]

for i in range(n_iter):



    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    scheduler.step()

    total_loss += loss.item()
    losses.append(loss.item())
    if i % log_interval == 0 and i > 0:
        lr = scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - start_time) * 1000 / log_interval
        cur_loss = total_loss / log_interval
        # ppl = math.exp(cur_loss)
        s_hat = out.flatten(1).detach().cpu().numpy()
        s_gt = s[:n_func_pred].flatten(1).detach().cpu().numpy()
        RMSE = np.mean(np.power(s_hat - s_gt, 2).sum(1) / np.power(s_gt, 2).sum(1))
        MSE = np.power(s_hat - s_gt, 2).mean()
        print(f'| iter {i:3d} | {i:5d}/{num_batches:5d} batches | '
              f'lr {lr:2.5f} | ms/batch {ms_per_batch:5.3f} | '
              f'loss {cur_loss:5.6f} | RMSE {RMSE:5.6f} | MSE {MSE:5.6f}')
        total_loss = 0
        start_time = time.time()
    if i % save_interval == 0 and i > 0:
        np.save('/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/PHIFLOW/' + name + '_loss.npy',
                np.array(losses))
        torch.save(model.state_dict(),
                   '/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/PHIFLOW/' + name + '.pt')

torch.save(model.state_dict(),
           '/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/PHIFLOW/' + name + '.pt')
np.save('/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/Climate/' + name + '_loss.npy',
        np.array(losses))
# model_VIT.eval()
# with torch.no_grad():
#     path = '/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/Climate/Data'
#     dataset = MyDataset_LARGE(path,train=False,map_reversed=map_reversed)
#     testloader = DataLoader(dataset, batch_size=bs, shuffle=False)
#     for k, batch in enumerate(testloader):
#         u, s, y = batch
#         break

#     errors = []
#     n = 10
#     n_func_pred = 1
#     for i in range(10,1825):
#          #[e for e in range(i-50,i)] #[int(np.clip(i%365 + 365*k + p ,0,1824)) for k in range(5) for p in range(-scope,scope+1)]  #[i%365 +365*k + p for k in range(5) for p in range(5)] + [i]
#         u, s = sample_batch(i, dataset.u_train, dataset.s_train,n)
#         inp = torch.cat([u.reshape(-1,1,720,720),s.reshape(-1,1,720,720)],dim=1)
#         inp[:n_func_pred,1] = 0  #caching the first function result

#         src_mask = generate_square_subsequent_mask(n, n_func_pred).to('cuda')
#         out = model_VIT(inp,src_mask)[:n_func_pred].reshape(n_func_pred,720,720)
#         loss = torch.pow(out-s[:n_func_pred].reshape(n_func_pred,720,720),2).sum()/n_func_pred


#         s_hat = out.flatten().detach().cpu().numpy()*9.130892637850938 + 96.815196053188
#         s_gt = s[:n_func_pred].flatten().detach().cpu().numpy()*9.130892637850938 + 96.815196053188
#         err = np.linalg.norm(s_hat - s_gt, 2)/np.linalg.norm(s_gt,2)
#         errors.append(err)
#         if i%100==0:
#             print(err)

# print("The average test relative error is {} the standard deviation is {} the min error is {} and the max error is {}".format(np.mean(errors),np.std(errors),np.min(errors),np.max(errors)))
# np.save('/media/data_cifs_lrs/projects/prj_control/mathieu/Operator/Transfo/Climate/'+ name + '.npy',errors)

