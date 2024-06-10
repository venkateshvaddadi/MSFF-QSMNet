#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:49:39 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:02:28 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:24:27 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:44:32 2021

@author: venkatesh
"""


import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn



import numpy as np
import time
import scipy.io
import tqdm
import matplotlib.pyplot as plt
import scipy.io
import os
#%%
from loss import*
from MSFF_QSMNet_Model import MSFF_QSMNet

#%%
epoch=5
device_id=1
data_source_no=1
is_data_normalized=True


#%%
input_dir="data/qsm_2016_recon_challenge/input/"
out_dir="data/qsm_2016_recon_challenge/output/"
#%%

try:
    os.makedirs(out_dir)
except:
    print("Exception...")
#%%
dw = MSFF_QSMNet()
model_path="savedModels/MSFF_QSMNet/ckpt.t7"
checkpoint = torch.load(model_path)
dw.load_state_dict(checkpoint['net'])
dw.eval()
dw = dw.cuda(device_id)
print("model loaded")

#%%

# save_path = "savedModels/MSFF_QSMNet/ckpt.t7"
# torch.save({'net': dw.state_dict()}, save_path)



#%%
# define the train data stats
stats = scipy.io.loadmat('csv_files/tr-stats.mat')


if(not is_data_normalized):
    sus_mean=0
    sus_std=1
    print('\n\n data is not normalized..................\n\n ')

else:
    stats = scipy.io.loadmat('csv_files/tr-stats.mat')
    sus_mean= torch.tensor(stats['out_mean']).cuda(device_id)
    sus_std = torch.tensor(stats['out_std' ]).cuda(device_id)
    phs_mean= torch.tensor(stats['inp_mean']).cuda(device_id)
    phs_std = torch.tensor(stats['inp_std' ]).cuda(device_id)

    print(phs_mean,phs_std)
    print(sus_mean,sus_std)

#%%

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("\nToc: start time not set")

#%%

def padding_data(input_field):
    N = np.shape(input_field)
    N_16 = np.ceil(np.divide(N,16.))*16
    N_dif = np.int16((N_16 - N) / 2)
    npad = ((N_dif[0],N_dif[0]),(N_dif[1],N_dif[1]),(N_dif[2],N_dif[2]))
    pad_field = np.pad(input_field, pad_width = npad, mode = 'constant', constant_values = 0)
    # pad_field = np.expand_dims(pad_field, axis=0)
    # pad_field = np.expand_dims(pad_field, axis=0)
    return pad_field, N_dif, N_16

def crop_data(result_pad, N_dif):
    result_pad = result_pad.squeeze()
    N_p = np.shape(result_pad)
    result_final  = result_pad[N_dif[0]:N_p[0]-N_dif[0],N_dif[1]:N_p[1]-N_dif[1],N_dif[2]:N_p[2]-N_dif[2]]
    return result_final

#%%

with torch.no_grad():


    phs=scipy.io.loadmat(input_dir+'/phs1.mat')['phs_tissue']
    sus=scipy.io.loadmat(input_dir+'/cos1.mat')['cos']
    msk=scipy.io.loadmat(input_dir+'/msk1.mat')['msk']

    print(phs.shape)
    print(msk.shape)
    print(sus.shape)
    phs, N_dif_phs, N_16_phs=padding_data(phs)
    msk, N_dif_msk, N_16_msk=padding_data(msk)
    sus, N_dif_sus, N_16_sus=padding_data(sus)
    print(phs.shape)
    print(msk.shape)
    print(sus.shape)


    
    
    phs=torch.unsqueeze(torch.unsqueeze(torch.tensor(phs),0),0)
    sus=torch.unsqueeze(torch.unsqueeze(torch.tensor(sus),0),0)
    msk=torch.unsqueeze(torch.unsqueeze(torch.tensor(msk),0),0)

    phs=phs*msk
    sus=sus*msk
    
    phs=phs.cuda(device_id).float()
    sus=sus.cuda(device_id).float()
    msk=msk.cuda(device_id).float()

    tic()

    phs=(phs-phs_mean)/phs_std
    output=dw(phs)
    sus_cal=output[0]

    sus_cal=sus_cal*sus_std+sus_mean
    sus_cal=sus_cal*msk

    sus_cal_cpu=sus_cal.detach().cpu().numpy()
    sus_given_cpu=sus.cpu().numpy()
    
    sus_cal_cpu=crop_data(sus_cal_cpu, N_dif_sus)
    sus_cal_cpu=sus_cal_cpu
    mdic  = {"sus_cal" : sus_cal_cpu}
    filename  = out_dir +"sus_cal.mat"
    scipy.io.savemat(filename, mdic)

        
    toc()
#%%














