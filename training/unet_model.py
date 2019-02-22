# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 00:21:23 2018
@author: Nabilla Abraham
"""

import torch
import torch.nn as nn
from torchvision import models

base_model = models.resnet18(pretrained=False)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        
        self.base_layers = list(base_model.children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 256, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu(256, 256, 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu(512, 512, 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu(1024, 512, 1, 0)  
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(2048, 1024, 1, 0)  
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up2 = convrelu(512 + 512, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
        
        return nn.Sigmoid(out)
    
def conv_relu_bn(in_channels, out_channels, kernel=3):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
            )

class unet(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
        self.img_channels = img_channels
        
        self.conv1 = conv_relu_bn(self.img_channels,64)
        self.conv2 = conv_relu_bn(64,128)
        self.conv3 = conv_relu_bn(128,256)
        self.conv4 = conv_relu_bn(256,512)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.up3 = conv_relu_bn(256 + 512, 256)
        self.up2 = conv_relu_bn(128 + 256, 128)
        self.up1 = conv_relu_bn(64 + 128, 64)
        
        self.output = nn.Sequential(nn.Conv2d(64, 1, 1),
                                    nn.Sigmoid())
        
    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.conv2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.conv3(x)
        x = self.maxpool(conv3)
        
        x = self.conv4(x)
        
        x = self.upsample(x)
        x = torch.cat([conv3, x], dim=1)
        x = self.up3(x)
        
        x = self.upsample(x)
        x = torch.cat([conv2, x], dim=1)
        x = self.up2(x)
        
        x = self.upsample(x)
        x = torch.cat([conv1, x], dim=1)
        x = self.up1(x)
        out = self.output(x)
        return out

class unet_vae(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
    
        self.img_channels = img_channels
        self.conv1 = conv_relu_bn(self.img_channels,64)
        self.conv2 = conv_relu_bn(64,128)
        self.conv3 = conv_relu_bn(128,256)
        self.conv4 = conv_relu_bn(256,512)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.up3 = conv_relu_bn(256 + 512, 256)
        self.up2 = conv_relu_bn(128 + 256, 128)
        self.up1 = conv_relu_bn(64 + 128, 64)
        
        self.output = nn.Sequential(nn.Conv2d(64, 1, 1),
                                    nn.Sigmoid())
        
        self.encode = nn.Sequential( 
                        conv_relu_bn(self.img_channels,64),nn.MaxPool2d(2),
                        conv_relu_bn(64,128), nn.MaxPool2d(2),
                        conv_relu_bn(128,256), nn.MaxPool2d(2),
                        conv_relu_bn(256,512)
                        )
        
        self.decode = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        conv_relu_bn(256 + 512, 256),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        conv_relu_bn(128 + 256, 128),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        conv_relu_bn(64 + 128, 64), 
                        nn.Conv2d(64, 1, 1),
                        nn.Sigmoid()
                       )
        
        
    def forward(self, x):
        x = self.encode(x)
        out = self.decode(x)
        return out
       
class unet_pyramid(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = conv_relu_bn(3,64)
        self.conv2 = conv_relu_bn(64,128)
        self.conv3 = conv_relu_bn(128+32,256)
        self.conv4 = conv_relu_bn(256+32,512)
        self.pyramid = conv_relu_bn(3,32)
        
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.up3 = conv_relu_bn(256 + 512 + 32, 256)    #skip + conv4 out + conv from pyramid  
        self.up2 = conv_relu_bn(128 + 256, 128)
        self.up1 = conv_relu_bn(64 + 128 , 64)
        
        self.output = nn.Sequential(nn.Conv2d(64, 1, 1),
                                    nn.Sigmoid())
        
    def forward(self, x):
        scale_1 = x
        scale_2 = self.avgpool(scale_1)
        scale_3 = self.avgpool(scale_2)
        scale_4 = self.avgpool(scale_3)
        
        conv1 = self.conv1(scale_1)         #chan = 64
        x = self.maxpool(conv1)
        
        conv2 = self.conv2(x)               #chan = 128
        inp2 = self.pyramid(scale_2)        #chan = 32
        x = torch.cat([conv2,inp2], dim=1)  #channels = 32 + 128
        x = self.maxpool(x)
        
        conv3 = self.conv3(x)               #chan = 256
        inp3 = self.pyramid(scale_3)        #chan = 32
        x = torch.cat([conv3, inp3], dim=1) #chan = 256 + 32
        x = self.maxpool(x)
        
        conv4 = self.conv4(x)               #chan = 512
        inp4 = self.pyramid(scale_4)        #chan = 32
        x = torch.cat([conv4, inp4], dim=1) #chan = 512 + 32
        
        x = self.upsample(x)
        x = torch.cat([conv3, x], dim=1)
        x = self.up3(x)
        
        x = self.upsample(x)    #(N, 256, 128,128)
        x = torch.cat([conv2, x], dim=1)
        x = self.up2(x)
        
        x = self.upsample(x)
        x = torch.cat([conv1, x], dim=1)
        x = self.up1(x)
        out = self.output(x)
        return out
        