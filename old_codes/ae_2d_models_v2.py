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
from math import floor
import optuna
import config
# from config import arch_params_dict as arch_par
# from config import optim_params_dict as optim_par

def get_encoder1_conv2d_arch_params(trial: optuna.Trial, dataset: str):
    
    encoder1_conv2d_arch_params = {
        'conv1_ch_out': trial.suggest_int(name="conv1_ch_out", # 8
                                          low=arch_par[dataset].encoder1_conv2d.conv1_ch_out.low, 
                                          high=arch_par[dataset].encoder1_conv2d.conv1_ch_out.high, 
                                          step=arch_par[dataset].encoder1_conv2d.conv1_ch_out.step
                                          ),
        'conv1_kernel_size': (3,3),
        'conv1_stride': (2,2),
        'conv1_padding': (1,1),
        'conv1_dilation': (1,1),
       
        'conv2_ch_out': trial.suggest_int(name="conv2_ch_out", # 16
                                          low=arch_par[dataset].encoder1_conv2d.conv2_ch_out.low, 
                                          high=arch_par[dataset].encoder1_conv2d.conv2_ch_out.high, 
                                          step=arch_par[dataset].encoder1_conv2d.conv2_ch_out.step
                                          ),
        'conv2_kernel_size': (3,3),
        'conv2_stride': (2,2),
        'conv2_padding': (1,1),
        'conv2_dilation': (1,1),
        
        'conv3_ch_out': trial.suggest_int(name="conv3_ch_out", # 32
                                          low=arch_par[dataset].encoder1_conv2d.conv3_ch_out.low, 
                                          high=arch_par[dataset].encoder1_conv2d.conv3_ch_out.high, 
                                          step=arch_par[dataset].encoder1_conv2d.conv3_ch_out.step
                                          ),
        'conv3_kernel_size': (3,3),
        'conv3_stride': (2,2),
        'conv3_padding': (0,0),
        'conv3_dilation': (1,1),
    
        'fc1_out': trial.suggest_int(name="fc1_out", # 128
                                     low=arch_par[dataset].encoder1_conv2d.fc1_out.low, 
                                     high=arch_par[dataset].encoder1_conv2d.fc1_out.high, 
                                     step=arch_par[dataset].encoder1_conv2d.fc1_out.step
                                     ),
        }
    
    return encoder1_conv2d_arch_params

def get_encoder2_conv2d_arch_params(trial: optuna.Trial, dataset: str):
    
    encoder2_conv2d_arch_params = {
        'conv1_ch_out': trial.suggest_int(name="conv1_ch_out", # 8
                                          low=arch_par[dataset].encoder2_conv2d.conv1_ch_out.low, 
                                          high=arch_par[dataset].encoder2_conv2d.conv1_ch_out.high, 
                                          step=arch_par[dataset].encoder2_conv2d.conv1_ch_out.step
                                          ),
        'conv1_kernel_size': (3,3),
        'conv1_stride': (2,2),
        'conv1_padding': (1,1),
        'conv1_dilation': (1,1),
       
        'conv2_ch_out': trial.suggest_int(name="conv2_ch_out", # 16
                                          low=arch_par[dataset].encoder2_conv2d.conv2_ch_out.low, 
                                          high=arch_par[dataset].encoder2_conv2d.conv2_ch_out.high, 
                                          step=arch_par[dataset].encoder2_conv2d.conv2_ch_out.step
                                          ),
        'conv2_kernel_size': (3,3),
        'conv2_stride': (2,2),
        'conv2_padding': (1,1),
        'conv2_dilation': (1,1),
       
        'enc_maxpool': True, #trial.suggest_categorical("enc_maxpool", [True, False]), 
        'pool1_kernel_size': (2,2),
        'pool1_stride': (1,1), 
        'pool1_padding': (1,1),
    
        'fc1_out': trial.suggest_int(name="fc1_out", # 128
                                     low=arch_par[dataset].encoder2_conv2d.fc1_out.low, 
                                     high=arch_par[dataset].encoder2_conv2d.fc1_out.high, 
                                     step=arch_par[dataset].encoder2_conv2d.fc1_out.step
                                     ),
        }
    
    return encoder2_conv2d_arch_params





    
# encoder = Encoder2Conv2D(n_timesteps, 
#                         n_features, 
#                         embedding_size, 
#                         batch_size,
#                         enc_arch_params)
# input_vector = torch.ones((batch_size, 1, n_timesteps, n_features))
# encoder.forward(input_vector)



class Encoder1Conv2D(nn.Module):
    
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int,
                 enc_arch_params: dict
                 ):
        super().__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.batch_size = batch_size
        self.conv2d_block_output_size = utils.calc_conv2d_block_output_size(input_size=(n_timesteps, n_features), 
                                                                            params=enc_arch_params)

        self.c1 = nn.Conv2d(in_channels=1, 
                            out_channels=enc_arch_params['conv1_ch_out'], 
                            kernel_size=enc_arch_params['conv1_kernel_size'], 
                            stride=enc_arch_params['conv1_stride'], 
                            padding=enc_arch_params['conv1_padding'],
                            dilation=enc_arch_params['conv1_dilation']
                            )
        self.c2 = nn.Conv2d(in_channels=enc_arch_params['conv1_ch_out'], 
                            out_channels=enc_arch_params['conv2_ch_out'], 
                            kernel_size=enc_arch_params['conv2_kernel_size'], 
                            stride=enc_arch_params['conv2_stride'], 
                            padding=enc_arch_params['conv2_padding'],
                            dilation=enc_arch_params['conv2_dilation']
                            )
        self.bn1 = nn.BatchNorm2d(num_features=enc_arch_params['conv2_ch_out'])
        self.c3 = nn.Conv2d(in_channels=enc_arch_params['conv2_ch_out'], 
                            out_channels=enc_arch_params['conv3_ch_out'], 
                            kernel_size=enc_arch_params['conv3_kernel_size'], 
                            stride=enc_arch_params['conv3_stride'], 
                            padding=enc_arch_params['conv3_padding'],
                            dilation=enc_arch_params['conv3_dilation']
                            )
        
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features=self.conv2d_block_output_size, 
                             out_features=enc_arch_params['fc1_out']
                             )
        self.fc2 = nn.Linear(in_features=enc_arch_params['fc1_out'], 
                             out_features=embedding_size
                             )
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        x = x.reshape((x.shape[0], 1, self.n_features, self.n_timesteps))
        # x = self.encoder_cnn(x)
        x = self.c1(x)
        # print(f"c1 output {x.shape}")
        x = self.relu(x)
        x = self.c2(x)
        # print(f"c2 output {x.shape}")
        x = self.bn1(x)
        # print(f"bn1 output {x.shape}")
        x = self.relu(x)
        x = self.c3(x)
        # print(f"c3 output {x.shape}")
        x = self.relu(x)        
        x = self.flatten(x)
        # print(f"flatten output {x.shape}")
        # x = self.encoder_lin(x)
        x = self.fc1(x)
        # print(f"fc1 output {x.shape}")
        x = self.relu(x)
        x = self.fc2(x)
        # print(f"fc2 output {x.shape}")        
        return x
    
# encoder = EncoderConv2D(n_timesteps, 
#                         n_features, 
#                         embedding_size, 
#                         batch_size)
# input_vector = torch.ones((batch_size, 1, n_timesteps, n_features))
# encoder.forward(input_vector)
# x = input_vector.reshape((input_vector.shape[0], 1, n_features, n_timesteps))
# x = c1(x)
# x = c2(x)
# x = bn1(x)
# x = c3(x)
# print(f"c3 output {x.shape}")
# x = relu(x)        
# x = flatten(x)
    
class Encoder2Conv2D(nn.Module):
    
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int,
                 enc_arch_params: dict
                 ):
        super().__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.batch_size = batch_size
        self.conv2d_block_output_size = utils.calc_conv2d_with_pool_block_output_size(input_size=(n_timesteps, n_features), 
                                                                                      params=enc_arch_params)

        self.c1 = nn.Conv2d(in_channels=1, 
                            out_channels=enc_arch_params['conv1_ch_out'], 
                            kernel_size=enc_arch_params['conv1_kernel_size'], 
                            stride=enc_arch_params['conv1_stride'], 
                            padding=enc_arch_params['conv1_padding'],
                            dilation=enc_arch_params['conv1_dilation']
                            )
        self.c2 = nn.Conv2d(in_channels=enc_arch_params['conv1_ch_out'], 
                            out_channels=enc_arch_params['conv2_ch_out'], 
                            kernel_size=enc_arch_params['conv2_kernel_size'], 
                            stride=enc_arch_params['conv2_stride'], 
                            padding=enc_arch_params['conv2_padding'],
                            dilation=enc_arch_params['conv2_dilation']
                            )
        self.bn1 = nn.BatchNorm2d(num_features=enc_arch_params['conv2_ch_out'])
        if enc_arch_params['enc_maxpool']:
            self.pool1 = nn.MaxPool2d(kernel_size=enc_arch_params['pool1_kernel_size'], 
                                      stride=enc_arch_params['pool1_stride'], 
                                      padding=enc_arch_params['pool1_padding']
                                      )
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=enc_arch_params['pool1_kernel_size'], 
                                      stride=enc_arch_params['pool1_stride'], 
                                      padding=enc_arch_params['pool1_padding']
                                      )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features=self.conv2d_block_output_size, 
                             out_features=enc_arch_params['fc1_out']
                             )
        self.fc2 = nn.Linear(in_features=enc_arch_params['fc1_out'], 
                             out_features=embedding_size
                             )
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        x = x.reshape((x.shape[0], 1, self.n_features, self.n_timesteps))
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.relu(x)        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)     
        return x
    


dec_arch_params = {
    
   'fc1_out': 128,
   
   'width_in': 30,
   'hight_in': 10,

   'convtr1_ch_in': 32,
   'convtr1_ch_out': 16,
   'convtr1_kernel_size': (5,4),
   'convtr1_stride': (2,2),
   'convtr1_padding': (0,0),
   'convtr1_output_padding': (1,1),
   
   'convtr2_kernel_size': (4,2),
   'convtr2_stride': (2,2),
   'convtr2_padding': (1,1),
   'convtr2_output_padding': (0,0),
   }
    
class DecoderConv2D(nn.Module):
    
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int,
                 dec_arch_params: dict
                 ):
        super().__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.batch_size = batch_size
        self.dec_arch_params = dec_arch_params
        
        self.fc1 = nn.Linear(in_features=embedding_size, 
                             out_features=dec_arch_params['fc1_out']
                             )
        self.fc2 = nn.Linear(in_features=dec_arch_params['fc1_out'], 
                             out_features=dec_arch_params['convtr1_ch_in'] * \
                                 dec_arch_params['width_in'] * dec_arch_params['hight_in']
                             )
        self.ct1 = nn.ConvTranspose2d(in_channels=dec_arch_params['convtr1_ch_in'], 
                                     out_channels=dec_arch_params['convtr1_ch_out'], 
                                     kernel_size=dec_arch_params['convtr1_kernel_size'],
                                     stride=dec_arch_params['convtr1_stride'], 
                                     padding=dec_arch_params['convtr1_padding'],
                                     output_padding=dec_arch_params['convtr1_output_padding'])
        self.bn1 = nn.BatchNorm2d(num_features=dec_arch_params['convtr1_ch_out'])
        self.relu = nn.ReLU(True)
        self.ct2 = nn.ConvTranspose2d(in_channels=dec_arch_params['convtr1_ch_out'], 
                                       out_channels=1, 
                                       kernel_size=dec_arch_params['convtr2_kernel_size'],
                                       stride=dec_arch_params['convtr2_stride'], 
                                       padding=dec_arch_params['convtr2_padding'], 
                                       output_padding=dec_arch_params['convtr2_output_padding'])
        # self.bn2 = nn.BatchNorm2d(num_features=1)
        # self.ct3 = nn.ConvTranspose2d(in_channels=8, 
        #                                out_channels=1, 
        #                                kernel_size=3, 
        #                                stride=2, 
        #                                padding=1, 
        #                                output_padding=1)
        
    def forward(self, x):
        x = x.reshape((x.shape[0], self.embedding_size))
        x = self.fc1(x)
        # print(f'fc1 output: {x.shape}')
        x = self.relu(x)
        x = self.fc2(x)
        # print(f'fc2 output: {x.shape}')
        # print(f'relu2 output: {x.shape}')
        x = self.relu(x) 
        x = x.view(-1, 
                   self.dec_arch_params['convtr1_ch_in'], 
                   self.dec_arch_params['width_in'], 
                   self.dec_arch_params['hight_in'])
        # print(f'unflatten output: {x.shape}')
        x = self.ct1(x)
        # print(f'ct1 output: {x.shape}')
        x = self.bn1(x)
        # print(f'bn1 output: {x.shape}')
        x = self.relu(x) 
        x = self.ct2(x)
        x = torch.sigmoid(x)
        # print(f'final output: {x.shape}')
        return x

# decoder = DecoderConv2D(n_timesteps, 
#                         n_features, 
#                         embedding_size, 
#                         batch_size)
# latent_vector = torch.ones((batch_size, embedding_size))
# decoder.forward(latent_vector)

# unflatten = nn.Unflatten(0, (32, 3, 3))
# x = unflatten(torch.ones(288))
# ct1(x)

class Autoencoder(nn.Module):
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
    def encoding(self, x):
        return self.encoder(x)
    

    
if __name__ == "__main__":
    
    import datasets as ds
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from tqdm import tqdm
    import numpy as np
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, \
        LambdaLR, MultiplicativeLR, ExponentialLR, CosineAnnealingLR, CyclicLR
    
    DATASET = "SpeechCommands" # 'AudioMNIST' # "SpeechCommands" # "UrbanSound"
    
    X, X_min, X_max, y = ds.prepare_inputdata(dataset_name=DATASET, transform_type=ds.TransfType.MFCC)
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)
    
    train_set = ds.CustomDataset(X_train, y_train)
    val_set = ds.CustomDataset(X_val, y_val)
    
    # X = torch.squeeze(X)
    
    # Set parameters
    n_timesteps = X.shape[-2]
    n_features = X.shape[-1]
    
    N_EPOCH = 5
    BATCH_SIZE = 64
    EMBEDDING_SIZE = 28
    batch_size = 64
    embedding_size = 28
    
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
    # model = build_ae_model(n_timesteps, 
    #                         n_features, 
    #                         enc_arch_params=model_params['enc_arch_params'],
    #                         dec_arch_params=model_params['dec_arch_params'],
    #                         embedding_size=EMBEDDING_SIZE, 
    #                         batch_size=BATCH_SIZE,
    #                         encoder_model=model_params['encoder'],
    #                         decoder_model=model_params['decoder'],
    #                         ae_model=model_params['autoencoder'])
    
    encoder = Encoder1Conv2D(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size,
                            enc_arch_params)
    
    decoder = DecoderConv2D(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size,
                            dec_arch_params)
    model = Autoencoder(encoder, decoder)
                        
    model = model.to(device)
    
    # opt_params = model_params['optimization']
    # optimizer = getattr(optim, opt_params['optimizer'])(model.parameters(), lr=opt_params['learning_rate'])
    # scheduler = StepLR(optimizer, 
    #                    step_size=opt_params['scheduler_step_size'],
    #                    gamma=opt_params['gamma'])         
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, 
                        step_size=10, #kwargs['scheduler_step_size']
                        gamma=0.1)
    reconstruction_loss_fn = nn.MSELoss(reduction='mean').to(device)
    
    # checkpoint_path='runs/'+dataset+"_"+model_type+'_model_'+str(trial.number)+'_'+run_id+'.pt'
    
    history = dict(train=[], val=[], loss_diff=[])
    
    from utils import EarlyStopping    
    early_stopping = EarlyStopping(patience=7, delta=0, verbose=True)
    
    for epoch in tqdm(range(1, N_EPOCH + 1)):
      
           train_loss_batches = []
           val_loss_batches = []
           
           # Training
           model = model.train()
           for i, (seq_true, label) in enumerate(train_dataloader):
               # seq_true, _ = next(iter(train_dataloader))
               optimizer.zero_grad()
               seq_true = seq_true.to(device)
               seq_pred = model(seq_true)
               loss = reconstruction_loss_fn(seq_pred, seq_true)
               loss.backward()
               optimizer.step()
               train_loss_batches.append(loss.item())
                           
          # Validation
           model = model.eval()
           with torch.no_grad():
               for i, (seq_true, label) in enumerate(val_dataloader):
                   seq_true = seq_true.to(device)
                   seq_pred = model(seq_true)
                   loss = reconstruction_loss_fn(seq_pred, seq_true)
                   val_loss_batches.append(loss.item())
           
           train_loss_epoch = np.mean(train_loss_batches)
           val_loss_epoch = np.mean(val_loss_batches)
           loss_diff_epoch = np.abs(train_loss_epoch - val_loss_epoch)
           
           history['train'].append(train_loss_epoch)
           history['val'].append(val_loss_epoch)
           history['loss_diff'].append(loss_diff_epoch)
           
           # mlflow.log_params(trial.params)
           # mlflow.log_metric("train_loss", train_loss_epoch)
           # mlflow.log_metric('val_loss', val_loss_epoch)
           # mlflow.log_metric('epoch', epoch)
           # mlflow.log_metric('train_and_diff', np.min(history['train'] + history['loss_diff']))
       
           print(f'Epoch {epoch}: train loss {train_loss_epoch} val loss {val_loss_epoch}')
           
           early_stopping(val_loss_epoch, model)
           if early_stopping.meet_criterion:
               break
           
           # Learning rate decay
           scheduler.step()