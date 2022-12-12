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
from torchvision import transforms
from tqdm import tqdm
from datasets.data import MyDatasets
from model.pix2pix import pix2pix_Discriminator, pix2pix_Generator
import os
import cv2
from torch.utils.tensorboard import SummaryWriter
import random

from option.options import BaseOption


opt = BaseOption().parse()


noise_path = opt.noise_path
encode_path = opt.encode_path
truth_path = opt.truth_path

model_path = opt.model_path
output_img_path = opt.output_img_path
writer_dir = opt.writer_dir

num_epochs = opt.num_epochs
batch_size = opt.batch_size
noise = False
itr_save_img = opt.itr_save_img
epoch_save_model = opt.epoch_save_model
cuda_device = opt.cuda_device



def addNoise():
    if not os.path.exists('./mix'):
        os.mkdir('./mix')
    for i in range(5000):
        randi = random.randint(100,199)
        in1_path = './encode/'+str(i+1)+'_AB.jpg'
        in2_path = './output_noise/hbx_epoch'+str(randi)+'_1.jpg'
        out_path = './mix/'+str(i+1)+'_AB.jpg'
        
        in1 = cv2.imread(in1_path)
        in2 = cv2.imread(in2_path)
        
        in1 = 255-in1
        in2 = 255-in2
        
        out = in1+in2
        out = 255-out
        
        cv2.imwrite(out_path,out)
    return

        




def train_pix2pix():
    print("======= start =======")
    train_data = MyDatasets(encode_path, truth_path, transform=transforms.ToTensor(), step=False, add=False)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    
    print("total epoch = ",num_epochs)
    print("batch size = ", batch_size)
    print("model path is: ", model_path)
    print("output image path is: ", output_img_path)
    print("summary writer directory is: ", writer_dir)
    print("gpu id is: ", cuda_device)
    
    in_ch = 3
    out_ch = 3
    dim = 64
    lamb = 100
    G = pix2pix_Generator(in_ch,out_ch,dim)
    G = G.cuda(cuda_device)
    D = pix2pix_Discriminator(in_ch,out_ch,dim)
    D = D.cuda(cuda_device)
    
    print("G's device is: ", next(G.parameters()).device)
    print("D's device is: ", next(G.parameters()).device)
    
    G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    BCELoss = nn.BCELoss()
    BCELoss.cuda(cuda_device)
    L1 = nn.L1Loss()
    L1 = L1.cuda(cuda_device)
    
    G.train()
    D.train()
    
    writer = SummaryWriter(log_dir=writer_dir)
    
    print("=============== start training ===================")
    for epoch in range(num_epochs):
        itr = 0
        GAN_L1 = 0
        GAN_G_error = 0
        GAN_D_error = 0
        print("start training epoch = ", epoch)
        for edge, label in tqdm(train_loader):
            edge = edge.cuda(cuda_device)
            label = label.cuda(cuda_device)
            
            #    train the discriminator
            if len(edge) != batch_size:
                continue
            
            D.zero_grad()
            D_optimizer.zero_grad()
            
            xy = torch.cat([edge,label], dim=1)
            D_out_real = D(xy).squeeze()
            D_real_loss = BCELoss(D_out_real, torch.ones(D_out_real.size()).to('cuda:'+str(cuda_device)))
            
            D_out = G(edge).detach()
            fake_out = torch.cat([edge,D_out], dim=1)
            D_out_fake = D(fake_out).squeeze()
            D_fake_loss = BCELoss(D_out_fake, torch.ones(D_out_fake.size()).to('cuda:'+str(cuda_device)))
            
            D_loss = 0.5*(D_real_loss + D_fake_loss)
            GAN_D_error = D_loss
            D_loss.backward()
            D_optimizer.step()
            
            
            #   train the generator
            G_optimizer.zero_grad()
            
            G_out = G(edge)
            G_fake = torch.cat([edge,G_out], dim=1)
            D_fake_out = D(G_fake).squeeze()
            G_BCE_loss = BCELoss(D_fake_out, torch.ones(D_fake_out.size()).to('cuda:'+str(cuda_device)))
            G_L1_loss = L1(G_out, label)
            GAN_L1 = G_L1_loss
            
            G_loss = G_BCE_loss + lamb*G_L1_loss
            GAN_G_error = G_loss
            G_loss.backward()
            G_optimizer.step()
            
            if itr % itr_save_img == 0:
                label_numpy = (label.view(batch_size,3,256,256)).data.cpu().numpy()
                label = label_numpy[0].transpose(1,2,0)
                imgs_numpy = (G_out.view(batch_size, 3, 256, 256)).data.cpu().numpy()
                img = imgs_numpy[0].transpose(1,2,0)
                edge_numpy = (edge.view(batch_size,3,256,256)).data.cpu().numpy()
                edge = edge_numpy[0].transpose(1,2,0)
                if not os.path.exists(output_img_path):
                    os.makedirs(output_img_path)
                cv2.imwrite(output_img_path+'/mask_epoch'+str(epoch)+'_'+str(itr)+'.jpg',edge*256)
                cv2.imwrite(output_img_path+'/hbx_epoch'+str(epoch)+'_'+str(itr)+'.jpg',img*256)
                cv2.imwrite(output_img_path+'/truth_epoch'+str(epoch)+'_'+str(itr)+'.jpg',label*256)
                
            itr += 1

        if epoch % 5 == 0:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            print("saving the model at epoch = ",epoch)
            torch.save(D,model_path+'/epoch'+str(epoch)+'_D_save.pth')
            torch.save(G,model_path+'/epoch'+str(epoch)+'_G_save.pth')
        
        writer.add_scalar("GAN_L1",GAN_L1,epoch)
        writer.add_scalar("GAN_G_error",GAN_G_error,epoch)
        writer.add_scalar("GAN_D_error",GAN_D_error,epoch)
    
    writer.close()
    print("=============== training end ================")
        
    
    return
    
    




if __name__ == '__main__':
    # addNoise()
    train_pix2pix()