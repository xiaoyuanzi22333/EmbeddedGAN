import numpy as np
import torch
import os
from torch.autograd import Variable

dtype = torch.FloatTensor

def bce_loss(input, target):

    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminator_loss(logits_real, logits_fake):

    N = logits_fake.size()
    true_labels = Variable(torch.ones(N).type(dtype))
    true_labels = true_labels.cuda()
    
    real_loss = bce_loss(logits_real, true_labels)
    fake_loss = bce_loss(logits_fake, 1-true_labels)
    
    loss = real_loss + fake_loss
    
    return loss

def generator_loss(logits_fake):
    
    N = logits_fake.size()
    true_labels = Variable(torch.ones(N).type(dtype))
    true_labels = true_labels.cuda()
    
    loss = bce_loss(logits_fake, true_labels)

    return loss

def sample_noise(batch_size, dim):

    a = torch.rand(batch_size, dim)
    noise = a*2 - 1
    
    return noise
