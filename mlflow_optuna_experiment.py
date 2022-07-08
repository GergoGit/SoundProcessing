import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from torchmetrics.functional import accuracy
# from torchmetrics import Accuracy
import pandas as pd
import torchaudio

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
import optuna
import mlflow
import mlflow.tensorflow
import mlflow.pytorch
from mlflow import pytorch

import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\SoundProcessing')

from datasets import prepare_inputdata, CustomDataset, mfcc_transform
from utils import plot_loss, embedding_histogram, EarlyStopping
import utils
from model_builder import build_classifier_model
import model_params as mpar

# TODO: accuracy


model_list = ['Conv2D', 'RNN']
DATASET = "UrbanSound"
MODEL_TYPE = 'Conv2D'
# TRACKING_URI = 'http://127.0.0.1:5000'


X, y = prepare_inputdata(dataset_name="UrbanSound", transformation=mfcc_transform)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)

train_set = CustomDataset(X_train, y_train)
val_set = CustomDataset(X_val, y_val)
         
def objective(trial: optuna.Trial, dataset: str=DATASET, model_type: str=MODEL_TYPE):
    
     """
     Objective for optuna hyperparameter optimization
     NN Classifier is trained within the objective
     MLFlow Tracker is set
     """
    
     # mlflow.set_tracking_uri(TRACKING_URI)
     mlflow.set_experiment(experiment_name=dataset+"_"+model_type+"_exp")
     
     with mlflow.start_run(run_name=DATASET+"_"+MODEL_TYPE+str(trial.number)):
          
         model_params = mpar.get_model_params(trial, model_type)
         
         BATCH_SIZE = 64 #trial.suggest_categorical("batch_size", [16, 32, 64])
         N_EPOCH = 20
        
         train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
         val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
         
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
         n_timesteps = X.shape[2]
         n_freq = X.shape[3]
         n_classes = 10
         
         # model_params = {'classifier': ClassifierRNN,
         #                'arch_params': rnn_arch_params}
         
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
                train_accuracy_batches = []
                val_accuracy_batches = []
                
                # Training
                train_size = len(train_dataloader.dataset)
                train_correct = 0
                model = model.train()
                for i, (input_sample, target) in enumerate(train_dataloader):
                    # input_sample, target = next(iter(train_dataloader))
                    optimizer.zero_grad()
                    input_sample, target = input_sample.to(device), target.to(device)
                    pred = model(input_sample)
                    loss = loss_fn(pred, target)
                    loss.backward()
                    optimizer.step()
                    train_loss_batches.append(loss.item())                    
                    train_correct += (pred.argmax(1)==target).type(torch.float).sum().item()
                                
                # Validation
                val_size = len(val_dataloader.dataset)
                val_correct = 0
                model = model.eval()
                with torch.no_grad():
                    for i, (input_sample, target) in enumerate(val_dataloader):
                        input_sample, target = input_sample.to(device), target.to(device)
                        pred = model(input_sample)
                        loss = loss_fn(pred, target)
                        val_loss_batches.append(loss.item())                        
                        val_correct += (pred.argmax(1)==target).type(torch.float).sum().item()
                
                train_loss_epoch = np.mean(train_loss_batches)
                val_loss_epoch = np.mean(val_loss_batches)
                loss_diff_epoch = np.abs(train_loss_epoch - val_loss_epoch)
                train_acc_epoch = train_correct / train_size
                val_acc_epoch = val_correct / val_size
                
                mlflow.log_params(trial.params)
                mlflow.log_metric("train_loss", train_loss_epoch)
                mlflow.log_metric('val_loss', val_loss_epoch)
                mlflow.log_metric("train_acc", train_acc_epoch)
                mlflow.log_metric('val_acc', val_acc_epoch)
                
                history['train'].append(train_loss_epoch)
                history['val'].append(val_loss_epoch)
                history['loss_diff'].append(loss_diff_epoch)
            
                print(f'Epoch {epoch}: \n\
                      train loss {train_loss_epoch} \nval loss {val_loss_epoch} \n\
                      train accuracy {train_acc_epoch} \nval accuracy {val_acc_epoch}')
                
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