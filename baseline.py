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
from torch.utils.data import DataLoader
from utils.losses import discriminator_loss, generator_loss, sample_noise
from datasets.data import MyDatasets
from model.sample import build_dc_classifier, build_dc_generator
from tqdm import tqdm
from torch.autograd import Variable
import os
import cv2
from torch.utils.tensorboard import SummaryWriter

source_path = './sample1/'
target_path = './encode1b/'

target_path1 = './encode/'
target_path2 = './encode1b/'

mask_path = './mask/'
model_path = './log1_2(1b)'
output_img_path = './output_trail(1b)'
writer_dir = './runs1_100(1b)'


NUM_TRAIN = 50000
NUM_VAL = 5000
batch_size = 32

def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(),lr = 1e-3, betas = (0.5, 0.999))
    return optimizer



def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)


def init_img():
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    for filename in os.listdir(source_path):
        path = os.path.join(source_path,filename)
        input = cv2.imread(path)
        W,H,C = input.shape
        #
        output1 = input[:,:H//2,:]
        target1 = os.path.join(target_path,filename)
        cv2.imwrite(target1,output1)
        #
        output2 = input[:,H//2:,:]
        target2 = os.path.join(mask_path,filename)
        cv2.imwrite(target2,output2)
        
        print(filename)
        
def b2w():
    if not os.path.exists(target_path2):
        os.mkdir(target_path2)
    for filename in os.listdir(target_path):
        path = os.path.join(target_path,filename)
        input = cv2.imread(path)
        output = 256-input
        target = os.path.join(target_path2,filename)
        cv2.imwrite(target,output)
        
        pass
        

def train_noise():
    print("======= start training =======")
    train_data = MyDatasets(target_path, mask_path, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    
    num_epochs = 200
    iter_count = 0
    noise_size = 256
    dtype = torch.FloatTensor
    print("total epoch = ",num_epochs)
    print("batch size = ", batch_size)
    
    print("======= init the model =======")
    D = build_dc_classifier().cuda().type(dtype)
    D.apply(initialize_weights)
    D = D.cuda()
    G = build_dc_generator().cuda().type(dtype)
    G.apply(initialize_weights)
    G = G.cuda()
    
    print("D's device = ",next(D.parameters()).device)
    print("G's device = ",next(G.parameters()).device)

    print("======= init the solver =======")
    D_solver = get_optimizer(D)
    G_solver = get_optimizer(G)
    
    d_total_error = 0
    g_error = 0
    
    writer = SummaryWriter(log_dir=writer_dir)
    
    print("start here")
    for epoch in range(num_epochs):
        print("it is at epoch: ", epoch)
        itx = 0
        for x, label in tqdm(train_loader):
            # print(x.size())
            # print(label.size())
            # print("start")
            # print(len(x))
            if len(x) != batch_size:
                continue
            
            # print(x.device)
            D_solver.zero_grad()
            real_data = Variable(x).type(dtype)
            real_data = real_data.cuda()
            logits_real = D(2* (real_data - 0.5)).type(dtype)
            logits_real = logits_real.cuda()
            # print("logits_real device: ", logits_real.device)
            # print(x.size())
            # x = x.data.cpu().numpy()
            # x = x.reshape(64,3,256,256)
            # print(x[0].shape)
            # img = x[0].transpose(1,2,0)
            # print(img.shape)
            # cv2.imwrite('./output_trial/epoch'+str(epoch)+'_'+str(itx)+'.jpg',img*256)

            # print("batchsize =",batch_size)
            g_fake_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
            g_fake_seed = g_fake_seed.cuda()
            # print("g_fake_seed device: ", g_fake_seed.device)
            
            # print("g_size = ", g_fake_seed.size())
            fake_images = G(g_fake_seed).detach()
            # print("fake_images device: ", fake_images.device)
            logits_fake = D(fake_images.view(batch_size, 1, 256, 256))
            # print('logits_fake device: ', logits_fake.device)

            # print("end1")
            
            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()        
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
            g_fake_seed = g_fake_seed.cuda()
            # print("g_fake_seed device: ", g_fake_seed.device)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 256, 256))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()
            
            if itx%1000 == 0:
                imgs_numpy = (fake_images.view(batch_size, 1, 256, 256)).data.cpu().numpy()
                itx += 1
                img = imgs_numpy[0].transpose(1,2,0)
                print(img.shape)
                if not os.path.exists(output_img_path):
                    os.makedirs(output_img_path)
                cv2.imwrite(output_img_path+'/hbx_epoch'+str(epoch)+'_'+str(itx)+'.jpg',img*256)
        
        writer.add_scalar('d_total_error', d_total_error, epoch)
        writer.add_scalar('g_error', g_error, epoch)
        
        print('===============================')
        print('d_total_error', d_total_error)
        print('g_error', g_error)
        
        if epoch%1 == 0:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            print("saving the model at epoch = ",epoch)
            torch.save(D,model_path+'/epoch'+str(epoch)+'_D_save.pth')
            torch.save(G,model_path+'/epoch'+str(epoch)+'_G_save.pth')


if __name__ == '__main__':
    # init_img() 
    # b2w()
    train_noise()