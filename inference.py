# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 18:57:57 2019

@author: Nabilla Abraham
"""

from torchvision import transforms
from torch.autograd import Variable
from unet_model import unet
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import cv2

transforms = transforms.Compose([transforms.Resize(128),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0,0,0], 
                                                      std=[0.5,0.5,0.5])])
img = Image.open('test.jpg')
orig = plt.imread('test.jpg')

img = transforms(img)
img = Variable(img).cpu()
img = img.unsqueeze(0)
model = unet(1)
model.load_state_dict(torch.load('unet_brats.pth', map_location='cpu'))
model.cpu()
model.eval()
pred = model(img)


pred = pred.squeeze(0).squeeze(0)
pred = pred.detach().numpy()
pred_up = cv2.resize(pred, dsize=(240,240))
out = pred_up*orig
cv2.imwrite('output.jpg', out)
#edges = cv2.Canny(pred_up,100,200)
