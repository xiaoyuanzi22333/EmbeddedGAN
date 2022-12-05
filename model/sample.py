import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 3*256*256
batch_size = 32
channel_size = 1

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
   
    
class Unflatten(nn.Module):
    def __init__(self, N=-1, C=channel_size, H=256, W=256):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)


def build_dc_classifier():

    return nn.Sequential(
        #  input = 64* 3* 256* 256
        Unflatten(N=batch_size,C=1,H=256,W=256),
        nn.Conv2d(1,32,kernel_size=5, stride=1),             #  64* 32* 252* 252
        nn.LeakyReLU(negative_slope=0.01, inplace=True),     #  64* 32* 252* 252
        nn.MaxPool2d(kernel_size=2, stride=2),               #  64* 32* 126* 126
        nn.Conv2d(32,64,kernel_size=5, stride=1),            #  64* 64* 122* 122
        nn.LeakyReLU(negative_slope=0.01, inplace=True),     #  64* 64* 122* 122
        nn.MaxPool2d(kernel_size=2, stride=2),               #  64* 64* 61* 61
        Flatten(),
        nn.Linear(64* 61* 61,1024),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Linear(1024,1)
    )



def build_dc_generator():

    return nn.Sequential(
        #    input = 64* 3* 256* 256
        nn.Linear(256, 1024),                                         #    64* 1024
        nn.ReLU(inplace=True),                                                 #    64* 1024
        nn.BatchNorm1d(1024),                                               #    64* 1024
        nn.Linear(1024,32* 64* 64),                                          #    64* (32*64*64)
        nn.ReLU(inplace=True),                                                 #    64* (32*64*64)
        nn.BatchNorm1d(32* 64* 64),                                             #    64* (32*64*64)
        Unflatten(batch_size, 32, 64, 64),                                    #    64* 32* 64* 64
        nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),       #    64* 16* 128* 128
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),         #    64* 3* 256* 256
        nn.Tanh(),
        Flatten()
    )