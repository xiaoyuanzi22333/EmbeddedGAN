import os
import numpy as np
import torch
import rembg
import random



noise_path = './output_noise'


ran = random.randint(100,200)
img_path = './hbx_epoch'+str(ran)+'_1.jpg'

print(img_path)
