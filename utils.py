# -*- coding: utf-8 -*-
"""
EarlyStopping:
https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from math import floor

def plot_loss(data: dict, train='train', val="val", log_yscale: bool=True):
    plt.plot(data[train])
    if val != None:
        plt.plot(data[val])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Loss over training epochs')
    if val != None:
        plt.legend(['train','val'])
    else:
        plt.legend([train])
    if log_yscale:
        plt.yscale('log')
    plt.show()
    
def embedding_histogram(model: nn.Module, 
                        dataloader: DataLoader, 
                        n_batch: int, 
                        n_rows: int, 
                        n_cols: int):
    batch_counter = 0
    embeddings = []
    for i, seq in enumerate(dataloader):
        with torch.no_grad():
            embedding = model.encoding(seq).to('cpu')
        
        embeddings.append(embedding)
        batch_counter += 1
        if batch_counter >= n_batch:
            break
    
    result = torch.cat(embeddings, dim=0)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, 
                             sharex=True, sharey=True,
                             # figsize=(15, 15)
                             )
    
    i = 0
    for row in range(len(axes)):
        for col in range(len(axes[0])):
            axes[row][col].hist(result[:, i].numpy())
            i += 1
    plt.show()

def plot_generated_vs_original(dataloader: DataLoader, 
                               model: nn.Module, 
                               seq_number: int):
    """
    Parameters
    ----------
    dataloader : DataLoader
    model : nn.Module
        Defined network.
    seq_number : int
        integer less then or equal to batch_size.

    Returns
    -------
    Line chart of the original and the generated one.
    """
    seq_true = next(iter(dataloader))
    seq_pred = model.forward(seq_true)
    
    plt.plot(seq_true[seq_number].detach().numpy())
    plt.plot(seq_pred[seq_number].detach().numpy())
    plt.ylabel('y')
    plt.xlabel('time')
    plt.title('Sample check')
    plt.legend(['real','generated'])
    plt.show()
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience: int=5, delta: float=0, verbose: bool=False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.meet_criterion = False
        
    def __call__(self, val_loss_epoch, model):
        if self.best_loss == None:
            self.best_loss = val_loss_epoch
        elif self.best_loss - val_loss_epoch > self.delta:
            self.best_loss = val_loss_epoch
            self.best_model_weights = deepcopy(model.state_dict())
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss_epoch < self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print('Early stopping')
                self.meet_criterion = True
                
     
#####################################
# Conv1d and 2d output calculation
#####################################

def calc_conv1d_or_pool1d_output_size(input_size: int, kernel_size: int, stride: int, padding: int, dilation: int=1):
    output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1
    return output_size

def calc_conv2d_or_maxpool2d_output_size(input_size: tuple, kernel_size: tuple, stride: tuple=(1,1), padding: tuple=(0,0), dilation: tuple=(1,1)):
    width_output_size = floor((input_size[0] + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0]) + 1
    hight_output_size = floor((input_size[1] + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1]) + 1
    return width_output_size, hight_output_size

def calc_avgpool2d_output_size(input_size: tuple, kernel_size: tuple, stride: tuple=(1,1), padding: tuple=(0,0)):
    width_output_size = floor((input_size[0] + 2*padding[0] - kernel_size[0])/stride[0]) + 1
    hight_output_size = floor((input_size[1] + 2*padding[1] - kernel_size[1])/stride[1]) + 1
    return width_output_size, hight_output_size

def calculate_conv1d_block_output_size(input_size: int, params: dict):
    """
    Assuming we have 1 or 2 blocks containing conv1d and pooling1d layers after each other
    and we need to calculate the output size which is the input size of a fully connected layer
    after flattenning.

    Parameters
    ----------
    input_size : int
        DESCRIPTION.
    params : dict
        architecture parameters including channel_number, kernel_size, padding, stride

    Returns
    -------
    int
        output size of the last block (pooling layer)

    """
    
    conv1_output_size = calc_conv1d_or_pool1d_output_size(input_size=input_size, 
                                                        kernel_size=params['conv1_kernel_size'], 
                                                        stride=params['conv1_stride'], 
                                                        padding=params['conv1_padding'])
    
    pool1_output_size = calc_conv1d_or_pool1d_output_size(input_size=conv1_output_size, 
                                                        kernel_size=params['pool1_kernel_size'], 
                                                        stride=params['pool1_stride'], 
                                                        padding=params['pool1_padding'])
    
    if params['multi_block'] == False:
        flat_size = pool1_output_size * params['conv1_ch_out']
    else:
        conv2_output_size = calc_conv1d_or_pool1d_output_size(input_size=pool1_output_size, 
                                                            kernel_size=params['conv2_kernel_size'], 
                                                            stride=params['conv2_stride'], 
                                                            padding=params['conv2_padding'])
        
        pool2_output_size = calc_conv1d_or_pool1d_output_size(input_size=conv2_output_size, 
                                                            kernel_size=params['pool2_kernel_size'], 
                                                            stride=params['pool2_stride'], 
                                                            padding=params['pool2_padding'])
            
        flat_size = pool2_output_size * params['conv2_ch_out']
    return int(flat_size)
    
def calculate_conv2d_block_output_size(x_size: int, y_size: int, params: dict):
    """
    Assuming we have 1 or 2 blocks containing conv1d and pooling1d layers after each other
    and we need to calculate the output size which is the input size of a fully connected layer
    after flattenning.

    Parameters
    ----------
    x_size : int
    y_size : int
    params : dict
        architecture parameters including channel_number, kernel_size, padding, stride

    Returns
    -------
    int
        output size of the last block (pooling layer)

    """
    
    conv1_output_x_size = calc_conv1d_or_pool1d_output_size(input_size=x_size, 
                                                        kernel_size=params['conv1_kernel_size'], 
                                                        stride=params['conv1_stride'], 
                                                        padding=params['conv1_padding'])
    
    pool1_output_x_size = calc_conv1d_or_pool1d_output_size(input_size=conv1_output_x_size, 
                                                        kernel_size=params['pool1_kernel_size'], 
                                                        stride=params['pool1_stride'], 
                                                        padding=params['pool1_padding'])
    
    conv1_output_y_size = calc_conv1d_or_pool1d_output_size(input_size=y_size, 
                                                        kernel_size=params['conv1_kernel_size'], 
                                                        stride=params['conv1_stride'], 
                                                        padding=params['conv1_padding'])
    
    pool1_output_y_size = calc_conv1d_or_pool1d_output_size(input_size=conv1_output_y_size, 
                                                        kernel_size=params['pool1_kernel_size'], 
                                                        stride=params['pool1_stride'], 
                                                        padding=params['pool1_padding'])
    
    if params['multi_block'] == False:
        flat_size = pool1_output_x_size * pool1_output_y_size * params['conv1_ch_out']
    else:
        conv2_output_x_size = calc_conv1d_or_pool1d_output_size(input_size=pool1_output_x_size, 
                                                            kernel_size=params['conv2_kernel_size'], 
                                                            stride=params['conv2_stride'], 
                                                            padding=params['conv2_padding'])
        
        pool2_output_x_size = calc_conv1d_or_pool1d_output_size(input_size=conv2_output_x_size, 
                                                            kernel_size=params['pool2_kernel_size'], 
                                                            stride=params['pool2_stride'], 
                                                            padding=params['pool2_padding'])

        conv2_output_y_size = calc_conv1d_or_pool1d_output_size(input_size=pool1_output_y_size, 
                                                            kernel_size=params['conv2_kernel_size'], 
                                                            stride=params['conv2_stride'], 
                                                            padding=params['conv2_padding'])
        
        pool2_output_y_size = calc_conv1d_or_pool1d_output_size(input_size=conv2_output_y_size, 
                                                            kernel_size=params['pool2_kernel_size'], 
                                                            stride=params['pool2_stride'], 
                                                            padding=params['pool2_padding'])
            
        flat_size = pool2_output_x_size * pool2_output_y_size * params['conv2_ch_out']
    return int(flat_size)

def calc_conv2d_with_pool_block_output_size(input_size: tuple, params: dict):
    
    conv1_width_output_size, conv1_hight_output_size = calc_conv2d_or_maxpool2d_output_size(input_size=input_size, 
                                                                                        kernel_size=params['conv1_kernel_size'], 
                                                                                        stride=params['conv1_stride'], 
                                                                                        padding=params['conv1_padding'],
                                                                                        dilation=params['conv1_dilation']
                                                                                        )
    conv2_width_output_size, conv2_hight_output_size = calc_conv2d_or_maxpool2d_output_size(input_size=(conv1_width_output_size, conv1_hight_output_size), 
                                                                                        kernel_size=params['conv2_kernel_size'], 
                                                                                        stride=params['conv2_stride'], 
                                                                                        padding=params['conv2_padding'],
                                                                                        dilation=params['conv2_dilation']
                                                                                        )
    if params['enc_maxpool']:
        pool1_width_output_size, pool1_hight_output_size = calc_conv2d_or_maxpool2d_output_size(input_size=(conv2_width_output_size, conv2_hight_output_size), 
                                                                                            kernel_size=params['pool1_kernel_size'], 
                                                                                            stride=params['pool1_stride'],
                                                                                            padding=params['pool1_padding']
                                                                                            )
    else:
        pool1_width_output_size, pool1_hight_output_size = calc_avgpool2d_output_size(input_size=(conv2_width_output_size, conv2_hight_output_size), 
                                                                                      kernel_size=params['pool1_kernel_size'], 
                                                                                      stride=params['pool1_stride'],
                                                                                      padding=params['pool1_padding']
                                                                                      )
    
    flat_size = pool1_width_output_size * pool1_hight_output_size * params['conv2_ch_out']
    return int(flat_size)      

def calc_conv2d_block_output_size(input_size: tuple, params: dict):
    
    conv1_width_output_size, conv1_hight_output_size = calc_conv2d_or_maxpool2d_output_size(input_size=input_size, 
                                                                                        kernel_size=params['conv1_kernel_size'], 
                                                                                        stride=params['conv1_stride'], 
                                                                                        padding=params['conv1_padding'],
                                                                                        dilation=params['conv1_dilation']
                                                                                        )
    conv2_width_output_size, conv2_hight_output_size = calc_conv2d_or_maxpool2d_output_size(input_size=(conv1_width_output_size, conv1_hight_output_size), 
                                                                                        kernel_size=params['conv2_kernel_size'], 
                                                                                        stride=params['conv2_stride'], 
                                                                                        padding=params['conv2_padding'],
                                                                                        dilation=params['conv2_dilation']
                                                                                        )
    conv3_width_output_size, conv3_hight_output_size = calc_conv2d_or_maxpool2d_output_size(input_size=(conv2_width_output_size, conv2_hight_output_size), 
                                                                                        kernel_size=params['conv3_kernel_size'], 
                                                                                        stride=params['conv3_stride'], 
                                                                                        padding=params['conv3_padding'],
                                                                                        dilation=params['conv3_dilation']
                                                                                        )
    
    flat_size = conv3_width_output_size * conv3_hight_output_size * params['conv3_ch_out']
    return int(flat_size)      
                                                                                  
#####################################
# ConvTranspose output calculation
#####################################
    
def calc_convtranspose1d_output_size(input_size: int, kernel_size: int, stride: int, padding: int, output_padding: int=0, dilation: int=1):
    output_size = (input_size-1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1
    return output_size

def calc_convtranspose2d_output_size(input_size: tuple, kernel_size: tuple, stride: tuple=(1,1), padding: tuple=(0,0), output_padding: tuple=(0,0), dilation: tuple=(1,1)):
    width_output_size = (input_size[0]-1)*stride[0]-2*padding[0]+dilation[0]*(kernel_size[0]-1)+output_padding[0]+1
    hight_output_size = (input_size[1]-1)*stride[1]-2*padding[1]+dilation[1]*(kernel_size[1]-1)+output_padding[1]+1
    return width_output_size, hight_output_size

def calculate_convtranspose1d_block_output_size(input_size: int, **dec_arch_params: dict):
    
    convtranspose1_output_size = calc_convtranspose1d_output_size(input_size=input_size, 
                                                        kernel_size=dec_arch_params['convtr1_kernel_size'], 
                                                        stride=dec_arch_params['convtr1_stride'], 
                                                        padding=dec_arch_params['convtr1_padding'],
                                                        output_padding=dec_arch_params['convtr1_output_padding'],
                                                        dilation=dec_arch_params['convtr1_dilation'])
    
    if dec_arch_params['dec_multi_block'] == False:
        flat_size = convtranspose1_output_size * dec_arch_params['convtr1_ch_out']
        return int(flat_size)
    else:
        conv2_output_size = calc_convtranspose1d_output_size(input_size=convtranspose1_output_size, 
                                                            kernel_size=dec_arch_params['convtr2_kernel_size'], 
                                                            stride=dec_arch_params['convtr2_stride'], 
                                                            padding=dec_arch_params['convtr2_padding'],
                                                            output_padding=dec_arch_params['convtr2_output_padding'],
                                                            dilation=dec_arch_params['convtr2_dilation'])
            
        flat_size = conv2_output_size * dec_arch_params['convtr2_ch_out']
        return int(flat_size)

#####################################
# TCN output calculation
#####################################

def calc_dilated_casual_conv1d_output_size(input_size: int, kernel_size: int, n_layer: int):
    max_dilation = 2**n_layer
    output_size = input_size-(kernel_size*max_dilation)+1
    return output_size

def calculate_tcn_block_output_size(input_size: int, **params: dict):
    
    tcn_block1_output_size = calc_dilated_casual_conv1d_output_size(input_size=input_size, 
                                                                    kernel_size=params['tcn1_kernel_size'], 
                                                                    n_layer=params['tcn1_n_layer'])
    
    pool1_output_size = calc_conv1d_or_pool1d_output_size(input_size=tcn_block1_output_size, 
                                                        kernel_size=params['pool1_kernel_size'], 
                                                        stride=params['pool1_stride'], 
                                                        padding=params['pool1_padding'])
    
    flat_output_size = pool1_output_size * params['tcn1_ch_out']
    return int(flat_output_size)