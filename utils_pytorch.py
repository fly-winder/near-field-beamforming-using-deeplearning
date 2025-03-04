import torch
import torch.nn as nn
import scipy.io as sio
import math


Nt = 256

def trans_Vrf(temp):
    v_real = torch.cos(temp * math.pi)
    v_imag = torch.sin(temp * math.pi)
    vrf = torch.complex(v_real, v_imag)
    return vrf

def Rate_func(h, v, SNR_input):
    v = trans_Vrf(v)

    h = h.unsqueeze(1)
    v = v.unsqueeze(2)
    hv = torch.bmm(h.to(torch.complex64), v)
    hv = hv.squeeze(dim=-1)
    rate = torch.log2(1 + SNR_input / Nt * torch.pow(torch.abs(hv), 2))

    return -rate



def mat_load(path):
    print('loading data...')
    h = sio.loadmat(path + '/pcsi.mat')['pcsi']
    h_est = sio.loadmat(path + '/ecsi.mat')['ecsi']
    print('loading complete')
    print('The shape of CSI is: ', h_est.shape)
    return h, h_est
