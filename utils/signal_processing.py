import numpy as np
import torch
import librosa
from scipy import signal


def stft(y, n_fft=1024, hop_length=256, win_length=None):
    if win_length is None:
        win_length = n_fft
    
    window = torch.hann_window(win_length, device=y.device)
    
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    return spec


def istft(spec, n_fft=1024, hop_length=256, win_length=None, length=None):
    if win_length is None:
        win_length = n_fft
    
    window = torch.hann_window(win_length, device=spec.device)
    
    y = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        length=length
    )
    
    return y


def spec_to_mag_phase(spec):
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    return mag, phase


def mag_phase_to_spec(mag, phase):
    spec = mag * torch.exp(1j * phase)
    return spec


def normalize_signal(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal


def add_noise(signal, noise, snr_db):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scaling = np.sqrt(desired_noise_power / (noise_power + 1e-10))
    
    noisy_signal = signal + noise * noise_scaling
    return noisy_signal


def create_echo_signal(far_end, impulse_response):
    echo = signal.convolve(far_end, impulse_response, mode='full')
    echo = echo[:len(far_end)]
    return echo


def generate_synthetic_data(
    far_end,
    near_end_speech=None,
    noise=None,
    impulse_response=None,
    snr_db=30,
    ser_db=10
):
    far_end = normalize_signal(far_end)
    
    if impulse_response is None:
        ir_len = 512
        impulse_response = np.random.randn(ir_len) * np.exp(-np.arange(ir_len) / 100)
        impulse_response = normalize_signal(impulse_response)
    
    echo = create_echo_signal(far_end, impulse_response)
    
    if near_end_speech is None:
        near_end_speech = np.zeros_like(far_end)
    else:
        near_end_speech = normalize_signal(near_end_speech)
    
    if noise is None:
        noise = np.random.randn(len(far_end))
    else:
        noise = normalize_signal(noise)
    
    echo_power = np.mean(echo ** 2)
    near_end_power = np.mean(near_end_speech ** 2)
    
    if near_end_power > 0:
        scaling = np.sqrt(echo_power / (near_end_power * 10 ** (ser_db / 10)))
        near_end_speech = near_end_speech * scaling
    
    mic_signal = echo + near_end_speech
    mic_signal = add_noise(mic_signal, noise, snr_db)
    
    return {
        'far_end': far_end,
        'near_end': near_end_speech,
        'echo': echo,
        'mic': mic_signal,
        'impulse_response': impulse_response
    }