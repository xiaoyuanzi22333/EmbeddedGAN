import numpy as np
import os
import torch
import cv2
import random


cuda_device = 1


class MyDatasets():
    def __init__(self, target_path, mask_path, transform=None,step=False, add=False):
        self.img = os.listdir(target_path)
        self.label = os.listdir(mask_path)
        self.i = 0
        self.step = step
        self.add = add # which step
        
        assert  len(self.img) == len(self.label)
        
        self.transform = transform
        
        self.images_and_labels = []
        for i in range(len(self.img)):
            self.images_and_labels.append(
                # (target_path + self.img[i], mask_path + self.label[i])
                (target_path + str(i+1) + '_AB.jpg', mask_path + str(i+1) + '_AB.jpg')
            )

        
    def __getitem__(self, index):
        image_path, label_path = self.images_and_labels[index]
        
        # print(image_path)
        # print(label_path)
        
        randi = random.randint(100,199)
        # print(image_path)
        # print(label_path)
        
        img = cv2.imread(image_path)
        img = cv2.resize(img,(256,256))
        if self.step:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # cv2.imwrite('./output_trial/test1.jpg',img)
        # print(img.shape)
        
        label = cv2.imread(label_path)
        label = cv2.resize(label,(256,256))
        
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
            
        img = img.cuda(cuda_device)
        label = label.cuda(cuda_device)
        
        if self.add:
            noise_path = './output_noise/hbx_epoch'+str(randi)+'_1.jpg'
            noise = cv2.imread(noise_path)
            noise = cv2.cvtColor(noise,cv2.COLOR_RGB2GRAY)
            noise = cv2.resize(noise,(256,256))
            noise = self.transform(noise)
            noise = noise.cuda(cuda_device)
            img = torch.cat([img,noise],dim=0)
        
        
        # if self.i == 0:
        #     x = img.data.cpu().numpy()
        #     x = x.transpose(1,2,0)
        #     print(x.shape)
        #     cv2.imwrite('./output_trial/test3.jpg',x*256)
        #     print("=============")
        #     self.i = 1
        
        return img, label
    
    def __len__(self):
        return len(self.img)