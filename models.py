
import torch
import torch.nn as nn
import numpy as np

from MSFF_QSMNet_Model import U2NETP_for_3D_cus_ch
from MSFF_QSMNet_Model import MSFF_QSMNet


#%%
def loadmodel(model):
    if model == 'U2NETP_for_3D_cus_ch':
        net = U2NETP_for_3D_cus_ch(in_ch=1,out_ch=1,cus_ch=64,mid_ch=16)
    # proposed model
    elif model =='MSFF_QSMNet':
        net = MSFF_QSMNet()
    else:
        print('Model Not Identified..!')
        net = None
    return net



 
if __name__ == "__main__":

    net = loadmodel('MSFF_QSMNet')
    print(net)