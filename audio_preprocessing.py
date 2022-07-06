import torch
import torchaudio

def cut_if_necessary(signal: torch.tensor, sample_size: int):
    """
    If the length of the signal is longer then sample_size than cut down 
    the rest of the signal on the right side.
    """
    if signal.shape[1] > sample_size:
        signal = signal[:, :sample_size]
    return signal

def right_pad_if_necessary(signal: torch.tensor, sample_size: int):
    """
    If the length of the signal is shorter then sample_size than pad it
    with 0 values on the right side.
    """
    length_signal = signal.shape[1]
    if length_signal < sample_size:
        n_missing_samples = sample_size - length_signal
        last_dim_padding = (0, n_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def resample_if_necessary(signal: torch.tensor, sr: int, target_sample_rate: int):
    """
    If the frequency of the audio waveform differs from the target frequency
    then it is resampled.
    """
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

def stereo_to_mono_if_necessary(signal: torch.tensor):
    """
    If the number of channels is more than 1 (stereo) 
    then we need to create a 1 channel (mono) signal
    """
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal