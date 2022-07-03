# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 03:51:51 2022

@author: bonnyaigergo
"""
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from audio_preprocessing import resample_if_necessary, mix_down_if_necessary, cut_if_necessary, right_pad_if_necessary


def preprocess_and_save_urbansound_dataset():
    
    """
    Preprocess UrbanSound audio data from the downloaded files
    The data is from here:
        https://www.kaggle.com/datasets/chrisfilo/urbansound8k
    1 channel audio data is created with given sample_rate
    signals and labels (tensors) are saved to a data folder
    """
    
    METADATA_FILE = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\metadata\UrbanSound8K.csv"
    AUDIO_DIR = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio"
    
    TARGET_SAMPLE_RATE = 22050
    N_SAMPLES = 22050
    
    metadata = pd.read_csv(METADATA_FILE)
    
    damaged_file_list = ('17853-5-0-15.wav', '174873-3-5-12.wav')
    metadata_good = metadata.loc[~metadata['slice_file_name'].isin(damaged_file_list)]
    
    labels = []
    signals = []
    
    for i in range(len(metadata_good)):
        fold = f"fold{metadata_good.iloc[i, 5]}"
        path = os.path.join(AUDIO_DIR, fold, metadata_good.iloc[i, 0])
        signal, sr = torchaudio.load(path)
        signal = resample_if_necessary(signal, sr, TARGET_SAMPLE_RATE)
        signal = mix_down_if_necessary(signal)
        signal = cut_if_necessary(signal, N_SAMPLES)
        signal = right_pad_if_necessary(signal, N_SAMPLES)
        
        label = metadata_good.iloc[i, 6]
        
        signals.append(signal)
        labels.append(label)
    
    torch.save(signals, r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\signals.pt")
    torch.save(labels, r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\labels.pt")

def prepare_inputdata(dataset_name: str) -> list:
    
    if dataset_name == "UrbanSound":
        X = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\signals.pt")
        y = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\labels.pt")
        
    return X, y

# scale, transform

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(signals, labels, test_size=0.2, shuffle=True, random_state=123)

class CustomDataset(Dataset):
    """Custom dataset loader"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]


# annotations.loc[annotations['slice_file_name']=='17853-5-0-15.wav']['classID'].values

# file_list = []
# for i in range(len(annotations_good)):
#     fold = f"fold{annotations_good.iloc[i, 5]}"
#     path = os.path.join(AUDIO_DIR, fold, annotations_good.iloc[i, 0])
#     file_list.append(path)



# import glob
# file_list = glob.glob(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio\*\*.wav")

# for file in file_list:
#     signal, sr = torchaudio.load(file)
#     signal = _resample_if_necessary(signal, sr)
#     signal = _mix_down_if_necessary(signal)
#     signal = _cut_if_necessary(signal)
#     signal = _right_pad_if_necessary(signal)
    
#     signals.append(signal)
    # signal = transformation(signal)
    
# signal_dict = {'filename': file_list,
#                'signal': signals}
# df = pd.DataFrame(signal_dict)



# import re

# string_to_cut = r'D:\Thesis\Data Sets\Audio\UrbanSound8K\audio\fold1\101415-3-0-2.wav'
# filepattern = r'D:\Thesis\Data Sets\Audio\UrbanSound8K\audio\fold[1-9]+\'
# numberpattern = r'[1-9]+'
# pattern = r'[1-9]+'
# re.sub(pattern=pattern, repl='', string=string_to_cut)
