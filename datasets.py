"""
datasets:
https://github.com/Jakobovski/free-spoken-digit-dataset
https://github.com/soerenab/AudioMNIST
https://github.com/jayrodge/AudioMNIST-using-PyTorch
https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
https://github.com/karolpiczak/ESC-50


tried to load arff file without success:
https://folk.ntnu.no/staal/assets/sw/mit/arff/
https://discuss.analyticsvidhya.com/t/loading-arff-type-files-in-python/27419

new data source was found and used:
https://www.kaggle.com/datasets/chrisfilo/urbansound8k
https://www.kaggle.com/code/adinishad/urbansound-classification-with-pytorch-and-fun

https://jovian.ai/charmzshab/urban-sound-dataset
https://github.com/musikalkemist/pytorchforaudio/blob/main/06%20Padding%20audio%20files/urbansounddataset.py

https://www.youtube.com/watch?v=SFBfzr0wZIc&ab_channel=918.software

https://www.youtube.com/watch?v=3mju52xBFK8&ab_channel=AssemblyAI
https://github.com/AssemblyAI/youtube-tutorials/blob/main/torchaudio/torchaudio_tutorial.ipynb

# str(torchaudio.get_audio_backend())
# pip install PySoundFile
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import pandas as pd
from typing import Callable
from enum import Enum
from tqdm import tqdm
import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\SoundProcessing')

from audio_preprocessing import resample_if_necessary, stereo_to_mono_if_necessary, cut_if_necessary, right_pad_if_necessary
from config import audio_transf_params_dict as cfg


dataset_dict = {'UrbanSound': {'n_observations': 8_730,
                                'n_classes': 10,
                                'loc': r'D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data',
                                'class_dict': {0: "air_conditioner",
                                               1: "car_horn",
                                               2: "children_playing",
                                               3: "dog_bark",
                                               4: "drilling",
                                               5: "engine_idling",
                                               6: "gun_shot",
                                               7: "jackhammer",
                                               8: "siren",
                                               9: "street_music"
                                               }
                                },
                'SpeechCommands': {'n_observations': 41_767,
                                    'n_classes': 12,
                                    'loc': r'D:\Thesis\Data Sets\Audio\SpeechCommands\preprocessed_data',
                                    'class_dict': {0: "on",
                                                   1: "off",
                                                   2: "up",
                                                   3: "down",
                                                   4: "right",
                                                   5: "left",
                                                   6: "stop",
                                                   7: "go",
                                                   8: "backward",
                                                   9: "forward",
                                                   10: "no",
                                                   11: "yes"
                                                   }
                                    },
                'AudioMNIST':       {'n_observations': 30_000,
                                    'n_classes': 10,
                                    'loc': r'D:\Thesis\Data Sets\Audio\AudioMNIST\preprocessed_data',
                                    'class_dict': {0: "zero",
                                                   1: "one",
                                                   2: "two",
                                                   3: "three",
                                                   4: "four",
                                                   5: "five",
                                                   6: "six",
                                                   7: "seven",
                                                   8: "eight",
                                                   9: "nine"
                                                   }
                                    }
                }

class TransfType(Enum):
    MEL = "MEL"
    MFCC = "MFCC"
    LFCC = "LFCC"
    

# TARGET_SAMPLE_RATE = 22050
# SAMPLE_SIZE = 22050
# SAMPLE_RATE = 22050
# N_FFT = 1024
# WIN_LENGTH = None
# HOP_LENGTH = 512
# N_MELS = 128
# N_MFCC = 128
# N_LFCC = 128

######################################
# Preprocess and Save Audio Data
######################################

def preprocess_and_save_speechcommands_dataset(target_sample_rate: int=cfg.SpeechCommands.TARGET_SAMPLE_RATE,
                                               sample_size: int=cfg.SpeechCommands.SAMPLE_SIZE
                                               ) -> None:

    import glob
    item_list = ['on','off','up','down','right','left','stop','go','backward','forward','no','yes']
    
    labels = []
    signals = []
        
    for i in tqdm(range(len(item_list))):
        
        file_path_list = glob.glob("D:\Thesis\Data Sets\Audio\SpeechCommands\Audio"+"\\"+item_list[i]+"\*.wav")
        
        for file_path in file_path_list:
            signal, sr = torchaudio.load(file_path)
            signal = resample_if_necessary(signal, sr, target_sample_rate)
            signal = stereo_to_mono_if_necessary(signal)
            signal = cut_if_necessary(signal, sample_size)
            signal = right_pad_if_necessary(signal, sample_size)
            
            label = int(i)
            
            signals.append(signal)
            labels.append(label)
      
    signals = torch.stack(signals, dim=0)
    labels = torch.LongTensor(labels)
    
    torch.save(signals, r"D:\Thesis\Data Sets\Audio\SpeechCommands\preprocessed_data\signals.pt")
    torch.save(labels, r"D:\Thesis\Data Sets\Audio\SpeechCommands\preprocessed_data\labels.pt")
    


def preprocess_and_save_audiomnist_dataset(target_sample_rate: int=cfg.AudioMNIST.TARGET_SAMPLE_RATE,
                                           sample_size: int=cfg.AudioMNIST.SAMPLE_SIZE
                                           ) -> None:

    import glob
    file_path_list = glob.glob(r"D:\Thesis\Data Sets\Audio\AudioMNIST\audio\*\*.wav")
    
    labels = []
    signals = []
    
    for file_path in tqdm(file_path_list):
        signal, sr = torchaudio.load(file_path)
        signal = resample_if_necessary(signal, sr, target_sample_rate)
        signal = stereo_to_mono_if_necessary(signal)
        signal = cut_if_necessary(signal, sample_size)
        signal = right_pad_if_necessary(signal, sample_size)
        
        label = int(file_path.split("\\")[-1][0])
        
        signals.append(signal)
        labels.append(label)
        
    signals = torch.stack(signals, dim=0)
    labels = torch.LongTensor(labels)
    
    torch.save(signals, r"D:\Thesis\Data Sets\Audio\AudioMNIST\preprocessed_data\signals.pt")
    torch.save(labels, r"D:\Thesis\Data Sets\Audio\AudioMNIST\preprocessed_data\labels.pt")




def preprocess_and_save_urbansound_dataset(target_sample_rate: int=cfg.UrbanSound.TARGET_SAMPLE_RATE,
                                           sample_size: int=cfg.UrbanSound.SAMPLE_SIZE
                                           ) -> None:
    
    """
    Preprocess UrbanSound audio data from the downloaded files
    The data is from here:
        https://www.kaggle.com/datasets/chrisfilo/urbansound8k
    1 channel (mono) audio waveform data is created with a given sample_rate
    signals and labels (tensors) are saved into a folder
    """
    
    METADATA_FILE = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\metadata\UrbanSound8K.csv"
    AUDIO_DIR = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio"

    metadata = pd.read_csv(METADATA_FILE)
    
    damaged_file_list = ('17853-5-0-15.wav', '174873-3-5-12.wav')
    metadata_good = metadata.loc[~metadata['slice_file_name'].isin(damaged_file_list)]
    
    labels = []
    signals = []
    
    for i in tqdm(range(len(metadata_good))):
        fold = f"fold{metadata_good.iloc[i, 5]}"
        path = os.path.join(AUDIO_DIR, fold, metadata_good.iloc[i, 0])
        signal, sr = torchaudio.load(path)
        signal = resample_if_necessary(signal, sr, target_sample_rate)
        signal = stereo_to_mono_if_necessary(signal)
        signal = cut_if_necessary(signal, sample_size)
        signal = right_pad_if_necessary(signal, sample_size)
        
        label = metadata_good.iloc[i, 6]
        
        signals.append(signal)
        labels.append(label)
        
    signals = torch.stack(signals, dim=0)
    labels = torch.LongTensor(labels)
    
    torch.save(signals, r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\signals.pt")
    torch.save(labels, r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\labels.pt")

######################################
# Audio Data Transformation
######################################

# mel_spectrogram = T.MelSpectrogram(
#     sample_rate=SAMPLE_RATE,
#     n_fft=N_FFT,
#     hop_length=HOP_LENGTH,
#     n_mels=N_MELS
#     )


# mfcc_transform = T.MFCC(
#     sample_rate=SAMPLE_RATE,
#     n_mfcc=N_MFCC,
#     melkwargs={
#         "n_fft": N_FFT,
#         "n_mels": N_MELS,
#         "hop_length": HOP_LENGTH,
#         "mel_scale": "htk",
#     },
# )

# lfcc_transform = T.LFCC(
#     sample_rate=SAMPLE_RATE,
#     n_lfcc=N_LFCC,
#     speckwargs={
#         "n_fft": N_FFT,
#         "win_length": WIN_LENGTH,
#         "hop_length": HOP_LENGTH,
#     },
# )


######################################
# Prepare Audio Input Data
######################################

# def prepare_inputdata(dataset_name: str,
#                       transformation: Callable):
    
#     if dataset_name == "UrbanSound":
#         X = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\signals.pt")
#         y = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\labels.pt")
        
#         X_trans = []

#         for i in range(len(X)):
#             transformed_signal = transformation(X[i])
#             X_trans.append(transformed_signal)
            
#         X_trans = torch.stack(X_trans, dim=0)
                
#     return X_trans, y

def ts_scaler(X: torch.Tensor) -> list:
    """
    Parameters
    ----------
    X : torch.Tensor
        Time series observations without target variable.

    Returns
    -------
    X_scaled : torch.Tensor
    X_min : float
    X_max : float

    """
    X_min = torch.min(X)
    X_max = torch.max(X)
    X_scaled = (X - X_min)/(X_max - X_min)
            
    return X_scaled, X_min, X_max



def prepare_inputdata(dataset_name: str,
                      transform_type: TransfType):
    
    if transform_type == TransfType.MEL:
        transformation = T.MelSpectrogram(
                            sample_rate=cfg[dataset_name].SAMPLE_RATE,
                            n_fft=cfg[dataset_name].N_FFT,
                            hop_length=cfg[dataset_name].HOP_LENGTH,
                            n_mels=cfg[dataset_name].N_MELS
                            )
    
    elif transform_type == TransfType.MFCC:
        transformation = T.MFCC(
                            sample_rate=cfg[dataset_name].SAMPLE_RATE,
                            n_mfcc=cfg[dataset_name].N_MFCC,
                            melkwargs={
                                "n_fft": cfg[dataset_name].N_FFT,
                                "n_mels": cfg[dataset_name].N_MELS,
                                "hop_length": cfg[dataset_name].HOP_LENGTH,
                                "mel_scale": "htk"
                                }
                            )
    
    elif transform_type == TransfType.LFCC:
        transformation = T.LFCC(
                            sample_rate=cfg[dataset_name].SAMPLE_RATE,
                            n_lfcc=cfg[dataset_name].N_LFCC,
                            speckwargs={
                                "n_fft": cfg[dataset_name].N_FFT,
                                "win_length": cfg[dataset_name].WIN_LENGTH,
                                "hop_length": cfg[dataset_name].HOP_LENGTH
                                }
                            )
    else:
        raise NotImplementedError("Transformation Type: {transform_type} is not implemented")
    
    if dataset_name == "UrbanSound":
        X = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\signals.pt")
        y = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\labels.pt")
        
    if dataset_name == "SpeechCommands":
        X = torch.load(r"D:\Thesis\Data Sets\Audio\SpeechCommands\preprocessed_data\signals.pt")
        y = torch.load(r"D:\Thesis\Data Sets\Audio\SpeechCommands\preprocessed_data\labels.pt")
        
    if dataset_name == "AudioMNIST":
        X = torch.load(r"D:\Thesis\Data Sets\Audio\AudioMNIST\preprocessed_data\signals.pt")
        y = torch.load(r"D:\Thesis\Data Sets\Audio\AudioMNIST\preprocessed_data\labels.pt")
        
    X_trans = []

    for i in tqdm(range(len(X))):
        transformed_signal = transformation(X[i])
        X_trans.append(transformed_signal)
        
    X_trans = torch.stack(X_trans, dim=0)
    # X_trans = torch.squeeze(X_trans)
    
    X_scaled, X_min, X_max = ts_scaler(X_trans)
                
    return X_scaled, X_min, X_max, y


# TODO: scaler

class CustomDataset(Dataset):
    """Custom dataset loader"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]
    

if __name__ == "__main__":
    
    # preprocess_and_save_speechcommands_dataset(target_sample_rate=cfg.SpeechCommands.TARGET_SAMPLE_RATE,
    #                                             sample_size=cfg.SpeechCommands.SAMPLE_SIZE
    #                                             )
    
    # preprocess_and_save_audiomnist_dataset(target_sample_rate=cfg.AudioMNIST.TARGET_SAMPLE_RATE,
    #                                        sample_size=cfg.AudioMNIST.SAMPLE_SIZE
    #                                        )
    
    # preprocess_and_save_urbansound_dataset(target_sample_rate=cfg.UrbanSound.TARGET_SAMPLE_RATE,
    #                                        sample_size=cfg.UrbanSound.SAMPLE_SIZE
    #                                        )
    
    
    DATASET = "SpeechCommands" # 'AudioMNIST' # "SpeechCommands" # "UrbanSound"
    
    X, y = prepare_inputdata(dataset_name=DATASET, transform_type=TransfType.MFCC)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)
    
    train_set = CustomDataset(X_train, y_train)
    val_set = CustomDataset(X_val, y_val)
    
    BATCH_SIZE = 64
    
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    
    input_sound, label = next(iter(train_dataloader))
    
    d = torch.squeeze(input_sound[1])
