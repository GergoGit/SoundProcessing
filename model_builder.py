import torch.nn as nn
import torch

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



if __name__ == "__main__":
    
    from models import NetConv2D
    
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
    
    model = build_classifier_model(n_timesteps=128, 
                                    n_freq=44,  
                                    n_classes=10,
                                    batch_size=64,
                                    model=model_params['classifier'],
                                    params=model_params['arch_params'])