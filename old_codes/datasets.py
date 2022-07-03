# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 08:47:52 2022

@author: bonnyaigergo
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


"""

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

import os

import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import torchaudio

# str(torchaudio.get_audio_backend())
# pip install PySoundFile

ANNOTATIONS_FILE = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\metadata\UrbanSound8K.csv"
AUDIO_DIR = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio"
SAMPLE_RATE = 22050
n_samples = 22050

class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 n_samples):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        # label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal#, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.n_samples:
            signal = signal[:, :self.n_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.n_samples:
            num_missing_samples = self.n_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        """
        If the number of channels is more than 1 then we need to create a 1 channel signal

        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

def create_train_val_dataset(dataset):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                            n_fft=1024,
                                                            hop_length=512,
                                                            n_mels=64
                                                            )
    data = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            n_samples)
         
    from math import ceil
    TRAIN_PROP = 0.7
    train_size = ceil(len(data)*TRAIN_PROP)
    val_size = len(data)-train_size
    
    train_set, val_set = random_split(data, [train_size, val_size])
    
    return train_set, val_set

if __name__ == "__main__":
    ANNOTATIONS_FILE = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\metadata\UrbanSound8K.csv"
    AUDIO_DIR = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio"
    SAMPLE_RATE = 22050
    n_samples = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
        )
    
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc = 40 
        )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            n_samples)
    print(f"There are {len(usd)} samples in the dataset.")
    signal = usd[1]
    signal, label = usd[1]
    
    signal[0]
    
    
    
    labels = []
    signals = []
    
    import glob
    file_list = glob.glob(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio\*\*.wav")
    
    

    for file in file_list:
        signal, sr = torchaudio.load(file)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
    


