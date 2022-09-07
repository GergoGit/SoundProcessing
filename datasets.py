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
                'SpeechCommands': {'n_observations': 8_730,
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
                'AudioMNIST':       {'n_observations': 8_730,
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
                                    }
                }


TARGET_SAMPLE_RATE = 22050
SAMPLE_SIZE = 22050
SAMPLE_RATE = 22050
N_FFT = 1024
WIN_LENGTH = None
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 128
N_LFCC = 128

######################################
# Preprocess and Save Audio Data
######################################

def preprocess_and_save_speechcommands_dataset(target_sample_rate: int=16000,
                                               sample_size: int=16000
                                               ) -> None:

    import glob
    item_list = ['on','off','up','down','right','left','stop','go','backward','forward','no','yes']
    
    labels = []
    signals = []
        
    for i in range(len(item_list)):
        
        file_path_list = glob.glob("D:\Thesis\Data Sets\Audio\SpeechCommands\audio"+"\\"+item_list[i]+"\*.wav")
        
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
    
    

def preprocess_and_save_audiomnist_dataset(target_sample_rate: int=TARGET_SAMPLE_RATE,
                                           sample_size: int=SAMPLE_SIZE
                                           ) -> None:

    import glob
    file_path_list = glob.glob(r"D:\Thesis\Data Sets\Audio\AudioMNIST\audio\*\*.wav")
    
    labels = []
    signals = []
    
    for file_path in file_path_list:
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




def preprocess_and_save_urbansound_dataset(target_sample_rate: int=TARGET_SAMPLE_RATE,
                                           sample_size: int=SAMPLE_SIZE
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
    
    for i in range(len(metadata_good)):
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


######################################
# Prepare Audio Input Data
######################################

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
    
def prepare_inputdata(dataset_name: str,
                      transform_type: str):
    
    if transform_type == 'MEL':
        transformation = T.MelSpectrogram(
                            sample_rate=cfg[dataset_name].SAMPLE_RATE,
                            n_fft=N_FFT,
                            hop_length=HOP_LENGTH,
                            n_mels=N_MELS
                            )
    
    if dataset_name == "UrbanSound":
        X = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\signals.pt")
        y = torch.load(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\preprocessed_data\labels.pt")
        
        X_trans = []

        for i in range(len(X)):
            transformed_signal = transformation(X[i])
            X_trans.append(transformed_signal)
            
        X_trans = torch.stack(X_trans, dim=0)
                
    return X_trans, y


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
    
    X, y = prepare_inputdata(dataset_name="UrbanSound", transformation=mfcc_transform)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)
    
    train_set = CustomDataset(X_train, y_train)
    val_set = CustomDataset(X_val, y_val)
    
    BATCH_SIZE = 64
    
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)