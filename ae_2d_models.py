# -*- coding: utf-8 -*-
"""
AE:
https://www.kaggle.com/code/ljlbarbosa/convolution-autoencoder-pytorch/notebook
https://www.kaggle.com/code/nathra/fashion-mnist-convolutional-autoencoder/notebook
https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
https://github.com/ShashiChilukuri/Deep-Learning-Projects/blob/master/CNN(PyTorch)%20-%20MNIST%20Convolutional%20Autoencoder/CNN(PyTorch)%20-%20MNIST%20Transpose%20Convolutional%20Autoencoder.ipynb
https://github.com/GuitarsAI/MLfAS/blob/master/MLAS_06_Convolutional_Autoencoder.ipynb
https://datahacker.rs/003-gans-autoencoder-implemented-with-pytorch/

VAE:
https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/
https://github.com/yjlolo/vae-audio

output size:
http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html
"""

import torch
from torch import nn
import torch.nn.functional as F

import utils


class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x