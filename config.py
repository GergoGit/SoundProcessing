"""
Parameter dictionaries for pretraining

"""

from easydict import EasyDict as edict

SEED_NUM = 123

################ NN architecture parameters ################

audio_transf_params_dict = edict({
    
    'UrbanSound':     {'TARGET_SAMPLE_RATE': 22_050,
                       'SAMPLE_SIZE': 22_050,
                       'SAMPLE_RATE': 22_050,
                       'N_FFT': 1_024,
                       'WIN_LENGTH': None,
                       'HOP_LENGTH': 512,
                       'N_MELS': 128,
                       'N_MFCC': 128,
                       'N_LFCC': 128
                       },
    'SpeechCommands': {'TARGET_SAMPLE_RATE': 22_050,
                       'SAMPLE_SIZE': 22_050,
                       'SAMPLE_RATE': 22_050,
                       'N_FFT': 1_024,
                       'WIN_LENGTH': None,
                       'HOP_LENGTH': 512,
                       'N_MELS': 128,
                       'N_MFCC': 128,
                       'N_LFCC': 128
                       },
    'AudioMNIST':     {'TARGET_SAMPLE_RATE': 16_000,
                       'SAMPLE_SIZE': 16_000,
                       'SAMPLE_RATE': 16_000,
                       'N_FFT': 1_024,
                       'WIN_LENGTH': None,
                       'HOP_LENGTH': 512,
                       'N_MELS': 128,
                       'N_MFCC': 128,
                       'N_LFCC': 128
                       }
    })