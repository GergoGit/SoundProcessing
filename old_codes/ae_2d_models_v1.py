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


class EncoderConv2D(nn.Module):
    
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int
                 # enc_arch_params: dict
                 ):
        super().__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.batch_size = batch_size        
        ### Convolutional section
        # self.encoder_cnn = nn.Sequential(
        self.c1 = nn.Conv2d(in_channels=1, 
                              out_channels=8, 
                              kernel_size=3, 
                              stride=2, 
                              padding=1)
        self.relu = nn.ReLU(True)
        self.c2 = nn.Conv2d(in_channels=8, 
                      out_channels=16, 
                      kernel_size=3, 
                      stride=2, 
                      padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
            # nn.ReLU(True),
        self.c3 = nn.Conv2d(in_channels=16, 
                      out_channels=32, 
                      kernel_size=3, 
                      stride=2, 
                      padding=0)
            # nn.ReLU(True)
        # )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        # self.encoder_lin = nn.Sequential(
        self.fc1 = nn.Linear(2400, 128)
            # nn.ReLU(True),
        self.fc2 = nn.Linear(128, embedding_size)
        # )
        
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
    
class DecoderConv2D(nn.Module):
    
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int
                 # dec_arch_params: dict
                 ):
        super().__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.batch_size = batch_size   
        # self.decoder_lin = nn.Sequential(
        self.fc1 = nn.Linear(embedding_size, 128)
            # nn.ReLU(True),
        self.fc2 = nn.Linear(128, 32 * 30 * 10)
            # nn.ReLU(True)
        # )

        # self.unflatten = nn.Unflatten(dim=0, unflattened_size=(32, 32, 12))

        # self.decoder_conv = nn.Sequential(
        self.ct1 = nn.ConvTranspose2d(in_channels=32, 
                                     out_channels=16, 
                                     kernel_size=(5,4),
                                     stride=2, 
                                     padding=(0,0),
                                     output_padding=(1,1))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(True)
        self.ct2 = nn.ConvTranspose2d(in_channels=16, 
                                       out_channels=1, 
                                       kernel_size=(4,2),
                                       stride=2, 
                                       padding=(1,1), 
                                       output_padding=(0,0))
        # self.bn2 = nn.BatchNorm2d(num_features=1)
            # nn.ReLU(True),
        # self.ct3 = nn.ConvTranspose2d(in_channels=8, 
        #                                out_channels=1, 
        #                                kernel_size=3, 
        #                                stride=2, 
        #                                padding=1, 
        #                                output_padding=1)
        # )
        
    def forward(self, x):
        x = x.reshape((x.shape[0], self.embedding_size))
        # x = self.decoder_lin(x)
        x = self.fc1(x)
        # print(f'fc1 output: {x.shape}')
        x = self.relu(x)
        x = self.fc2(x)
        # print(f'fc2 output: {x.shape}')
        # print(f'relu2 output: {x.shape}')
        x = self.relu(x) 
        # x = self.unflatten(x)
        x = x.view(-1, 32, 30, 10)
        # print(f'unflatten output: {x.shape}')
        # x = self.decoder_conv(x)
        x = self.ct1(x)
        # print(f'ct1 output: {x.shape}')
        x = self.bn1(x)
        # print(f'bn1 output: {x.shape}')
        x = self.relu(x) 
        x = self.ct2(x)
        # print(f'ct2 output: {x.shape}')
        # x = self.bn2(x)
        # print(f'bn2 output: {x.shape}')
        # x = self.relu(x)
        # x = self.ct3(x)     
        # print(f'ct3 output: {x.shape}')
        # TODO: Scaling
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
    
class EncoderStacked(nn.Module):
    
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int
                 # enc_arch_params: dict
                 ):
        super().__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.batch_size = batch_size
        
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(n_timesteps * n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        x = x.reshape((x.shape[0], self.n_features, self.n_timesteps))
        # x = self.decoder_lin(x)
        x = self.fc1(x)
        # print(f'fc1 output: {x.shape}')
        x = self.relu(x)
        x = self.fc2(x)
        # print(f'fc2 output: {x.shape}')
        x = self.relu(x) 
        x = self.fc3(x)
        # print(f'fc3 output: {x.shape}')
        return x

class DecoderStacked(nn.Module):
    
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int
                 # dec_arch_params: dict
                 ):
        super().__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.batch_size = batch_size
        
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, n_timesteps * n_features)
        

    def forward(self, x):
        x = x.reshape((x.shape[0], self.embedding_size))
        # x = self.decoder_lin(x)
        x = self.fc1(x)
        # print(f'fc1 output: {x.shape}')
        x = self.relu(x)
        x = self.fc2(x)
        # print(f'fc2 output: {x.shape}')
        x = self.relu(x) 
        x = self.fc3(x)
        # print(f'fc3 output: {x.shape}')
        return x
    
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
    
    encoder = EncoderConv2D(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size)
    
    decoder = DecoderConv2D(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size)
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