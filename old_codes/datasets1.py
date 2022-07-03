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


def _cut_if_necessary(signal):
        if signal.shape[1] > n_samples:
            signal = signal[:, :n_samples]
        return signal

def _right_pad_if_necessary(signal):
    length_signal = signal.shape[1]
    if length_signal < n_samples:
        num_missing_samples = n_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def _resample_if_necessary(signal, sr):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

def _mix_down_if_necessary(signal):
    """
    If the number of channels is more than 1 then we need to create a 1 channel signal

    """
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def _get_audio_sample_path(index):
    fold = f"fold{annotations.iloc[index, 5]}"
    path = os.path.join(audio_dir, fold, annotations.iloc[
        index, 0])
    return path

def _get_audio_sample_label(index):
    return annotations.iloc[index, 6]

   
METADATA_FILE = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\metadata\UrbanSound8K.csv"
AUDIO_DIR = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio"

target_sample_rate = 22050
n_samples = 22050

annotations = pd.read_csv(METADATA_FILE)
annotations.columns

damaged_file_list = ('17853-5-0-15.wav', '174873-3-5-12.wav')
metadata_good = annotations.loc[~annotations['slice_file_name'].isin(damaged_file_list)]

labels = []
signals = []

for i in range(len(metadata_good)):
    fold = f"fold{metadata_good.iloc[i, 5]}"
    path = os.path.join(AUDIO_DIR, fold, metadata_good.iloc[i, 0])
    signal, sr = torchaudio.load(path)
    signal = _resample_if_necessary(signal, sr)
    signal = _mix_down_if_necessary(signal)
    signal = _cut_if_necessary(signal)
    signal = _right_pad_if_necessary(signal)
    
    label = metadata_good.iloc[i, 6]
    
    signals.append(signal)
    labels.append(label)

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
