import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import torchvision.transforms as transforms
import scipy.io
from torch.autograd import Variable
from padutils import*

class qsmdata(Dataset):
    
    def __init__(self, csv_file_path, root_dir, training = True):
        
        self.names     = pd.read_csv(csv_file_path)
        self.root_dir  = root_dir         
        self.training = training       
        
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        
        if self.training==True:            
            file_name = os.path.join(self.root_dir,self.names['FileName'][idx])             
            data = scipy.io.loadmat(file_name)
            #print(list(data.keys()))
            phs  = torch.tensor(data['phs']).unsqueeze(dim=0)     
            sus  = torch.tensor(data['susc']).unsqueeze(dim=0) 
            msk = torch.tensor(data['msk']).unsqueeze(dim=0)       
            return phs, sus, msk         

        else:
            file_name = os.path.join(self.root_dir,self.names['FileName'][idx])  
            p_id = self.names['Label'][idx]
            data = scipy.io.loadmat(file_name)
            #msk, N_dif, N_16 = padding_data(data['msk'])
            #msk = torch.tensor(msk).squeeze(dim=0) 
            phs, N_dif, N_16 = padding_data(data['phs'])
            phs = torch.tensor(phs).squeeze(dim=0) 
            
            return phs, N_dif, p_id
            


## Check the dataloader

if __name__=="__main__":
    
    loader = qsmdata('./val.csv', './Data/Validation Data/')
    trainloader = DataLoader(loader, batch_size = 1, shuffle=False, num_workers=1)
    print(len(trainloader))
    for i, data in enumerate(trainloader): 
        Xtkd, Stkd, msk = data[0], data[1], data[2]
        print(i)
        print(Xtkd.size())
        print(Stkd.size())
        print('-------------------------------------------------------------------');
        
