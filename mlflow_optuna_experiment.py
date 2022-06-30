# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:58:12 2022

@author: bonnyaigergo
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import accuracy
import pandas as pd
import torchaudio

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
import optuna
# MLFlow
import mlflow
import mlflow.tensorflow
import mlflow.pytorch
from mlflow import pytorch

import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\SoundProcessing')

from datasets import create_train_val_dataset
from utils import plot_loss, embedding_histogram, EarlyStopping
import model_params as mpar


model_list = ['Conv2D', 'RNN']
DATASET = "UrbanSound"
MODEL_TYPE = 'Conv2D'
# TRACKING_URI = 'http://127.0.0.1:5000'


def build_classifier_model(n_timesteps: int, 
                            n_freq: int,  
                            n_classes: int,
                            batch_size: int,
                            model: nn.Module,
                            params: dict):
        
    classifier_model = model(n_timesteps, 
                            n_freq, 
                            n_classes, 
                            batch_size,
                            params)
    
    return classifier_model


def objective(trial: optuna.Trial, dataset: str=DATASET, model_type: str=MODEL_TYPE):
    
     # mlflow.set_tracking_uri(TRACKING_URI)
     mlflow.set_experiment(experiment_name=dataset+"_"+model_type+"_exp")
     
     with mlflow.start_run(run_name=DATASET+"_"+MODEL_TYPE+str(trial.number)):
          
         model_params = mpar.get_model_params(trial, model_type)
         
         BATCH_SIZE = 64 #trial.suggest_categorical("batch_size", [16, 32, 64])
         N_EPOCH = 3
         
         train_set, val_set = create_train_val_dataset(dataset)
    
         # Create data loader for pytorch
         train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
         val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
         
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
         conv2d_arch_params = {
               'conv1_ch_in': 1, 
               'conv1_ch_out': 4, # trial.suggest_int(name="conv1_ch_out", low=3, high=6, step=1),
               'conv1_kernel_size': 5,
               'conv1_stride': 1,
               'conv1_padding': 0,
               'pool1_kernel_size': 2,
               'pool1_stride': 2, # default value is kernel_size
               'pool1_padding': 0,
               'multi_block': True, # trial.suggest_categorical("multi_block", [True, False]),
               'conv2_ch_out': 8,
               'conv2_kernel_size': 5,
               'conv2_stride': 1,
               'conv2_padding': 0,
               'pool2_kernel_size': 2,
               'pool2_stride': 2, # default value is kernel_size
               'pool2_padding': 0
               }
         
         model_params = {'classifier': NetConv2D,
                    'arch_params': conv2d_arch_params,
                    }
         
         n_timesteps = 64
         n_freq = 44
         n_classes = 10
         
         model = build_classifier_model(n_timesteps, 
                                        n_freq,  
                                        n_classes,
                                        batch_size=BATCH_SIZE,
                                        model=model_params['classifier'],
                                        params=model_params['arch_params'])
                            
         model = model.to(device)
         
         # optimizer = getattr(optim, kwargs['optimizer'])(model.parameters(), lr=kwargs['learning_rate'])
         optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
         scheduler = StepLR(optimizer, 
                            step_size=15, #kwargs['scheduler_step_size']
                            gamma=0.1)
         loss_fn = nn.NLLLoss().to(device)
         # loss_fn = nn.CrossEntropyLoss()
         
         checkpoint_path='runs/'+model_type+'model'+str(trial.number)+'.pt'
         
         history = dict(train=[], val=[], loss_diff=[])
         
         early_stopping=EarlyStopping(patience=5, delta=0, verbose=True)
         
         for epoch in tqdm(range(1, N_EPOCH + 1)):
           
                train_loss_batches = []
                val_loss_batches = []
                
                # Training
                model = model.train()
                for seq_true, target in enumerate(train_dataloader):
                    # seq_true, target = next(iter(train_dataloader))
                    optimizer.zero_grad()
                    seq_true = seq_true.to(device)
                    target = target.to(device)
                    pred = model(seq_true)
                    loss = loss_fn(pred, target)
                    loss.backward()
                    optimizer.step()
                    train_loss_batches.append(loss.item())
                                
               # Validation
                model = model.eval()
                with torch.no_grad():
                    for seq_true, target in enumerate(val_dataloader):
                        seq_true = seq_true.to(device)
                        target = target.to(device)
                        pred = model(seq_true)
                        loss = loss_fn(pred, target)
                        val_loss_batches.append(loss.item())
                
                train_loss_epoch = np.mean(train_loss_batches)
                val_loss_epoch = np.mean(val_loss_batches)
                loss_diff_epoch = np.abs(train_loss_epoch - val_loss_epoch)
                
                # mlflow.log_params(trial.params)
                # mlflow.log_metric("train_loss", train_loss_epoch)
                # mlflow.log_metric('val_loss', val_loss_epoch)
                
                history['train'].append(train_loss_epoch)
                history['val'].append(val_loss_epoch)
                history['loss_diff'].append(loss_diff_epoch)
            
                print(f'Epoch {epoch}: train loss {train_loss_epoch} val loss {val_loss_epoch}')
                
                early_stopping(val_loss_epoch, model)
                if early_stopping.meet_criterion:
                    break
                
                # Learning rate decay
                scheduler.step()
                
         torch.save({'model_state_dict': early_stopping.best_model_weights,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'arch_params': model.params,
                    'batch_size': model.batch_size
                    },
                   checkpoint_path)
        
         target_metric = np.min(history['train'] + history['loss_diff'])

     return target_metric

 
if __name__ == "__main__":
        
    N_TRIALS = 1
        
    study = optuna.create_study(direction="minimize", 
                                sampler=optuna.samplers.TPESampler(), 
                                pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=N_TRIALS)

    mlflow.end_run()