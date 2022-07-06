# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 03:51:51 2022

@author: bonnyaigergo
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Callable
import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\SoundProcessing')

from audio_preprocessing import resample_if_necessary, stereo_to_mono_if_necessary, cut_if_necessary, right_pad_if_necessary

TARGET_SAMPLE_RATE = 22050
N_SAMPLES = 22050
SAMPLE_RATE = 22050
N_FFT = 1024
WIN_LENGTH = None
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 128
N_LFCC = 128


def preprocess_and_save_urbansound_dataset(target_sample_rate: int=TARGET_SAMPLE_RATE,
                                           n_samples: int=N_SAMPLES
                                           ) -> None:
    
    """
    Preprocess UrbanSound audio data from the downloaded files
    The data is from here:
        https://www.kaggle.com/datasets/chrisfilo/urbansound8k
    1 channel (mono) audio data is created with given sample_rate
    signals and labels (tensors) are saved into a data folder
    """
    
    METADATA_FILE = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\metadata\UrbanSound8K.csv"
    AUDIO_DIR = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio"

    metadata = pd.read_csv(METADATA_FILE)
    
    damaged_file_list = ('17853-5-0-15.wav', '174873-3-5-12.wav')
    metadata_good = metadata.loc[~metadata['slice_file_name'].isin(damaged_file_list)]
    
    labels = []
    signals = []
    
    for i in range(len(metadata_good)):
        fold = f"fold{metadata_good.iloc[i, 5]}"
        path = os.path.join(AUDIO_DIR, fold, metadata_good.iloc[i, 0])
        signal, sr = torchaudio.load(path)
        signal = resample_if_necessary(signal, sr, target_sample_rate)
        signal = stereo_to_mono_if_necessary(signal)
        signal = cut_if_necessary(signal, n_samples)
        signal = right_pad_if_necessary(signal, n_samples)
        
        label = metadata_good.iloc[i, 6]
        
        signals.append(signal)
        labels.append(label)
        
    signals = torch.stack(signals, dim=0)
    
    torch.save(signals, r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\signals.pt")
    torch.save(labels, r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\labels.pt")


mel_spectrogram = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
    )


mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_fft": N_FFT,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "mel_scale": "htk",
    },
)

lfcc_transform = T.LFCC(
    sample_rate=SAMPLE_RATE,
    n_lfcc=N_LFCC,
    speckwargs={
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
    },
)



def prepare_inputdata(dataset_name: str,
                      transformation: Callable):
    
    if dataset_name == "UrbanSound":
        X = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\signals.pt")
        y = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\labels.pt")
        
        X_trans = []

        for i in range(len(X)):
            transformed_signal = transformation(X[i])
            X_trans.append(transformed_signal)
            
        X_trans = torch.stack(X_trans, dim=0)
                
    return X_trans, y
    

# TODO: scale

class CustomDataset(Dataset):
    """Custom dataset loader"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]


# from scipy.io import arff
# import pandas as pd

# data = arff.loadarff(r'D:\Thesis\Data Sets\TimeSeries\UrbanSound.arff')
# df = pd.DataFrame(data[0])

# from arff import arffread

#f = open('test.arff')
#name, sparse, alist, m) = arffread(f)

# I tried to unzip here, but the folders contain wrong files
# from zipfile import ZipFile

# with ZipFile(file=r'D:\Thesis\Data Sets\Audio\UrbanSound8k.zip', mode='r') as zipObj:
#    zipObj.extractall(r'D:\Thesis\Data Sets\Audio\UrbanSound8K')


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

if __name__ == "__main__":
    
    X, y = prepare_inputdata(dataset_name="UrbanSound", transformation=mfcc_transform)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)
    
    train_set = CustomDataset(X_train, y_train)
    val_set = CustomDataset(X_val, y_val)
    
    BATCH_SIZE = 64
    
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
