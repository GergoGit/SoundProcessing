import torch
import torchaudio

def cut_if_necessary(signal, n_samples):
        if signal.shape[1] > n_samples:
            signal = signal[:, :n_samples]
        return signal

def right_pad_if_necessary(signal, n_samples):
    length_signal = signal.shape[1]
    if length_signal < n_samples:
        num_missing_samples = n_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def resample_if_necessary(signal, sr, target_sample_rate):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

def mix_down_if_necessary(signal):
    """
    If the number of channels is more than 1 then we need to create a 1 channel signal

    """
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal