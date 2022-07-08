# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\SoundProcessing')

from math import floor
from models import NetConv2D, ClassifierRNN
import optuna

def get_model_params(trial: optuna.Trial, model_type: str):
    
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
         
    rnn_arch_params = {
        'cell_type': 'GRU', # trial.suggest_categorical("enc_multi_block", ['GRU', 'LSTM']),
        'multi_block': False,
        'layer1_hidden_size': 32,
        'layer1_n_layers': 5,
        'layer1_bidirectional': False,
        'layer1_dropout_rate': 0.2,
        'layer2_hidden_size': 16,
        'layer2_n_layers': 2,
        'layer2_bidirectional': False,
        'layer2_dropout_rate': 0.2,
        'before_fc_dropout': 0.4
        }
    
    optim_params = {
        'learning_rate': trial.suggest_loguniform(name='learning_rate', low=1e-5, high=1e-1),
        'optimizer': trial.suggest_categorical(name="optimizer", choices=["Adam", "RMSprop", "SGD"]),
        'scheduler_step_size': trial.suggest_int(name="scheduler_step_size", low=15, high=75, step=15)
        }
    
         
    model_params_dict = {
        'Conv2D': {'classifier': NetConv2D,
                   'arch_params': conv2d_arch_params,
                   # 'optimization': optim_params
                   },
        'RNN': {'classifier': ClassifierRNN,
                'arch_params': rnn_arch_params,
                # 'optimization': optim_params
                }
        }
    
    return model_params_dict[model_type]
