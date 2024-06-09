from __future__ import print_function

import argparse
import csv
import os, logging
import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from models import loadmodel
from utils import progress_bar, set_logging_defaults
from myDataset import qsmdata
import pandas as pd
#from loss import loss_gradient
from loss import total_loss_l1
from loss import sobel_kernel
from loss import dipole_kernel
from loss import total_loss_l1_with_model_loss
import scipy.io as io

#%%
parser = argparse.ArgumentParser(description='Susceptibility Mapping')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--name', default='proposed_model_MSFF_QSMNet',type=str, help='model name should assign')
parser.add_argument('--batch-size', default=8, type=int, help='batch size')
parser.add_argument('--epoch', default=50, type=int, help='total epochs to run')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--ngpu', default=2, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--saveroot', default='./savedModels', type=str, help='save directory')
parser.add_argument('--model', default='MSFF_QSMNet',  type=str, help='which model : model')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_val = 10000
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

cudnn.benchmark = True

print('==> Preparing The Dataloaders ... !')

# make the data iterator for training data
train_data = qsmdata('./csv_files/train.csv', './Data/Training Data/')

trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle = True, num_workers=2)

# make the data iterator for validation data
val_data = qsmdata('./csv_files/val.csv', './Data/Validation Data/')
valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, num_workers=2)

print('Number of Mini Batches In Training  : ' ,len(trainloader))
print('Number of QSM  Volumes In Training  : ' ,train_data.__len__())
print('Number of Mini Batches In Validation: ' ,len(valloader))
print('Number of QSM  Volumes In Validation: ' ,val_data.__len__())

# Model
print('==> Building model .. !')

net = loadmodel(args.model)

# print(net)

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    net.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')


if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=[0,1])


optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)

logdir = os.path.join(args.saveroot, args.model, args.name)
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')
logname = os.path.join(logdir, 'log.csv')


# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(logdir, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_val = checkpoint['val']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

#define sobel kernel
ss = sobel_kernel()
ss = ss.cuda()

#define dipole kernel
matrix_size = [64, 64, 64]
voxel_size = [1,  1,  1]

dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk=dk.float()
dk = torch.unsqueeze(dk, dim=0)

dk=dk.cuda()

criterion = nn.MSELoss()

# load train stats

data = io.loadmat('./csv_files/tr-stats.mat')

#print(data.keys())

if use_cuda:
    a  = torch.tensor(data['inp_mean']).cuda()
    b  = torch.tensor(data['out_mean']).cuda()
    x  = torch.tensor(data['inp_std' ]).cuda()
    y  = torch.tensor(data['out_std' ]).cuda()   
else:
    a  = torch.tensor(data['inp_mean'])
    b  = torch.tensor(data['out_mean'])
    x  = torch.tensor(data['inp_std' ])
    y  = torch.tensor(data['out_std' ])


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    for batch_idx, data in enumerate(trainloader):

        if use_cuda:
            inp, out,msk = data[0].cuda(), data[1].cuda(), data[2].cuda()
            inp = (inp-a)/x    
            out = (out-b)/y                     
        else:           
            inp, out, msk = data[0], data[1], data[2]
            inp = (inp-a)/x  
            out = (out-b)/y 


        d0,d1,d2,d3,d4,d5,d6,d7 = net(inp)

        loss=total_loss_l1_with_model_loss(chi=d0, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d1, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d2, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d3, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d4, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d5, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d6, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d7, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)


        train_loss += loss.item()        

        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        progress_bar(batch_idx, len(trainloader),
           'Loss: %.5f'
           % (train_loss/(batch_idx+1)))

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.5f}]'.format(
        epoch,
        train_loss/(batch_idx+1)
        ))
    torch.save(net.state_dict(), os.path.join(logdir, 'model_'+str(epoch)+'.pth'))
    return (train_loss/batch_idx+1)

def val(epoch):

    global best_val
    net.eval()
    val_loss = 0.0   

    with torch.no_grad():

        for batch_idx, data in enumerate(valloader):

            if use_cuda:
                inp, out,msk = data[0].cuda(), data[1].cuda(), data[2].cuda()
                inp = (inp-a)/x              
                out = (out-b)/y                        
            else:           
                inp, out, msk = data[0], data[1], data[2]
                inp = (inp-a)/x                        
                out = (out-b)/y 

            d0,d1,d2,d3,d4,d5,d6,d7 = net(inp)
            loss=total_loss_l1_with_model_loss(chi=d0, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d1, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d2, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d3, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d4, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d5, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d6, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)+total_loss_l1_with_model_loss(chi=d7, y=out, b=inp, d=dk, m=msk, sobel=ss,sm=b,ssd=y,pm=a,psd=x)


            val_loss += loss.item()

            progress_bar(batch_idx, len(valloader),
               'Loss: %.5f'
               % (val_loss/(batch_idx+1)))


    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [Loss {:.5f}] '.format(
        epoch,
        val_loss/(batch_idx+1)
        ))

    val = val_loss/(batch_idx+1)

    if val < best_val:
        best_val = val
        checkpoint(val, epoch)
    return val


def checkpoint(val, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
    'net': net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'val': val,
    'epoch': epoch,
    'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(logdir, 'ckpt.t7'))

def adjust_learning_rate(optimizer, epoch):
    #decrease the learning rate at 100 and 150 epoch
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Logs
for epoch in range(start_epoch, args.epoch):
    train_loss = train(epoch)
    val_loss   = val(epoch)
    adjust_learning_rate(optimizer, epoch)
    
print("Best Val : {}".format(best_val))
logger = logging.getLogger('best')
logger.info('[Val {:.5f}]'.format(best_val))

