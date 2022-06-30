# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:56:10 2022

@author: bonnyaigergo
https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
https://docs.microsoft.com/en-us/learn/modules/intro-audio-classification-pytorch/
"""

import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\SoundProcessing')

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
# import librosa.display as ld
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import UrbanSoundDataset


METADATA_FILE = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\metadata\UrbanSound8K.csv"
AUDIO_DIR = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio"

CLASS_ID = 5
RANDOM_FILE_NUM = 5

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
N_FFT = 1024 #2048
WIN_LENGTH = None
HOP_LENGTH = 512
N_MELS = 256 # 64, 128
N_MFCC = 128
N_LFCC = 128
    

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
    n_lfcc=N_LFFC,
    speckwargs={
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
    },
)

usd = UrbanSoundDataset(METADATA_FILE,
                        AUDIO_DIR,
                        mfcc_transform,
                        SAMPLE_RATE,
                        NUM_SAMPLES)

signal, label = usd[1]

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    
plot_spectrogram(signal[0])

import glob
# All files and directories ending with .wav and that don't begin with a dot:
file_list = glob.glob(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio\fold"+str(CLASS_ID)+"\*.wav")
# All files and directories ending with .wav with depth of 2 folders, ignoring names beginning with a dot:
# file_list = glob.glob(r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio\*\*.wav")


# play audio file (works in jupyter notebook)
# file_name = r"D:\Thesis\Data Sets\Audio\UrbanSound8K\audio\fold1\7061-6-0-0.wav"
file_name = file_list[RANDOM_FILE_NUM]
ipd.Audio(file_name)

# load single signal data
data, sampling_rate = librosa.load(file_name)
data.shape



# print(librosa.__version__)

# Waveplot
plt.figure(figsize=(12, 4))
plt.title('Waveform')
librosa.display.waveshow(data, sr=sampling_rate)

# Spectrogram with linear and log axis, Mel-Spectrogram
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=sampling_rate, ax=ax[0])
ax[0].set(title='Linear-frequency power spectrogram')
ax[0].label_outer()

# hop length is the non overlapping section of the window length
D = librosa.amplitude_to_db(np.abs(librosa.stft(data, hop_length=HOP_LENGTH)),
                            ref=np.max)
librosa.display.specshow(D, y_axis='log', x_axis='time', 
            sr=sampling_rate, hop_length=HOP_LENGTH, ax=ax[1])
ax[1].set(title='Log-frequency power spectrogram')
ax[1].label_outer()

MS = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
MS_db = librosa.power_to_db(MS, ref=np.max)
librosa.display.specshow(MS_db, 
                               x_axis='time', 
                               y_axis='mel', 
                               ax=ax[2])
ax[2].set(title='Mel-Spectrogram')
ax[2].label_outer()
fig.colorbar(img, ax=ax, format="%+2.f dB")

# Mel-Spectrogram
fig, ax = plt.subplots()
MS = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
MS_db = librosa.power_to_db(MS, ref=np.max)
img = librosa.display.specshow(MS_db, 
                               x_axis='time', 
                               y_axis='mel', 
                               ax=ax)
ax.set(title='Mel-Spectrogram')
fig.colorbar(img, ax=ax, format="%+2.f dB")