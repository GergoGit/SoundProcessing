# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 07:18:44 2022

@author: bonnyaigergo

https://www.kaggle.com/code/adinishad/urbansound-classification-with-pytorch-and-fun
https://docs.microsoft.com/en-us/learn/modules/intro-audio-classification-pytorch/4-speech-model

"""
import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\SoundProcessing')

from torch import nn
# from torchsummary import summary

import torch
from torch.nn.utils import weight_norm
# import torchinfo
import utils

class NetConv2D(nn.Module):
    def __init__(self,
                 n_timesteps: int,
                 n_freq: int,
                 n_classes: int, 
                 batch_size: int,
                 params: dict):
        super().__init__()
        self.n_classes = n_classes 
        self.batch_size = batch_size
        self.conv_block_output_size = utils.calculate_conv2d_block_output_size(n_timesteps,
                                                                               n_freq, 
                                                                               params)       
        self.multi_block = params['multi_block']
        self.params = params
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=params['conv1_ch_in'], 
                                            out_channels=params['conv1_ch_out'], 
                                            kernel_size=params['conv1_kernel_size'], 
                                            stride=params['conv1_stride'], 
                                            padding=params['conv1_padding']),
                                   nn.BatchNorm2d(params['conv1_ch_out']),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.MaxPool2d(kernel_size=params['pool1_kernel_size'], 
                                                stride=params['pool1_stride'], 
                                                padding=params['pool1_padding'])
                                   )
        if self.multi_block:
            self.conv2 = nn.Sequential(nn.Conv2d(in_channels=params['conv1_ch_out'], 
                                                out_channels=params['conv2_ch_out'], 
                                                kernel_size=params['conv2_kernel_size'], 
                                                stride=params['conv2_stride'], 
                                                padding=params['conv2_padding']),
                                       nn.BatchNorm2d(params['conv2_ch_out']),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.MaxPool2d(kernel_size=params['pool2_kernel_size'], 
                                         stride=params['pool2_stride'], 
                                         padding=params['pool2_padding'])
            )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.conv_block_output_size, self.n_classes)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        if self.multi_block:
            x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.relu(x)
        x = self.softmax(x)
        return x
    
    
class NetConv2D_v2(nn.Module):
    def __init__(self,
                 n_timesteps: int,
                 n_freq: int,
                 n_classes: int, 
                 batch_size: int,
                 params: dict):
        super().__init__()
        self.n_classes = n_classes 
        self.batch_size = batch_size
        self.conv_block_output_size = utils.calculate_conv2d_block_output_size(n_timesteps,
                                                                               n_freq, 
                                                                               params)
        self.multi_block = params['multi_block']
        self.params = params
        
        self.conv1 = nn.Conv2d(in_channels=params['conv1_ch_in'], 
                                            out_channels=params['conv1_ch_out'], 
                                            kernel_size=params['conv1_kernel_size'], 
                                            stride=params['conv1_stride'], 
                                            padding=params['conv1_padding']
                                )
        self.bn1 = nn.BatchNorm2d(params['conv1_ch_out'])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=params['pool1_kernel_size'], 
                                                stride=params['pool1_stride'], 
                                                padding=params['pool1_padding'])
                                   
        if self.multi_block:
            self.conv2 = nn.Conv2d(in_channels=params['conv1_ch_out'], 
                                                out_channels=params['conv2_ch_out'], 
                                                kernel_size=params['conv2_kernel_size'], 
                                                stride=params['conv2_stride'], 
                                                padding=params['conv2_padding'])
            self.bn2 = nn.BatchNorm2d(params['conv2_ch_out'])
            self.pool2 = nn.MaxPool2d(kernel_size=params['pool2_kernel_size'], 
                                         stride=params['pool2_stride'], 
                                         padding=params['pool2_padding'])
            
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.conv_block_output_size, self.n_classes)
        # self.relu = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.reshape((self.batch_size, self.n_features, self.n_timesteps))
        print(f"input {x.shape}")
        x = self.conv1(x)
        print(f"conv1 out {x.shape}")
        x = self.bn1(x)
        print(f"b1 out {x.shape}")
        x = self.relu(x)
        x = self.pool1(x)
        print(f"pool1 out {x.shape}")
        if self.multi_block:
            x = self.conv2(x)
            print(f"conv2 out {x.shape}")
            x = self.bn2(x)
            print(f"b2 out {x.shape}")
            x = self.relu(x)
            x = self.pool2(x)
            print(f"pool2 out {x.shape}")
        x = self.flatten(x)
        print(f"flatten out {x.shape}")
        x = self.fc(x)
        print(f"fc out {x.shape}")
        return x
  
# seq_true, target = next(iter(train_dataloader))
# seq_true.shape

# net = NetConv2D(n_timesteps=seq_true.shape[2],
#                 n_freq=seq_true.shape[3],
#                 n_classes=10, 
#                 batch_size=64,
#                 params=model_params['arch_params'])


# net.forward(seq_true)

# calc_conv_or_pool_output_size(input_size=64, kernel_size=5, stride=1, padding=0)
# calc_conv_or_pool_output_size(input_size=60, kernel_size=2, stride=2, padding=0)

# calc_conv_or_pool_output_size(input_size=30, kernel_size=5, stride=1, padding=0)
# calc_conv_or_pool_output_size(input_size=26, kernel_size=2, stride=2, padding=0)

class CNNNetwork(nn.Module):

    def __init__(self, n_classes: int):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.pooling = nn.AdaptiveAvgPool2d((8, 8)) # extended
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.pooling(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 64, 44))