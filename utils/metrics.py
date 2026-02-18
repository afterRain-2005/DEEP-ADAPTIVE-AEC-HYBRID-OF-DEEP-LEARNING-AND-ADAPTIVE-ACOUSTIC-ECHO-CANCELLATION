import numpy as np
from scipy import signal
import librosa


def compute_erle(mic_signal, enhanced_signal, echo_signal=None):
    if echo_signal is None:
        echo_signal = mic_signal - enhanced_signal
    
    mic_power = np.mean(mic_signal ** 2)
    enhanced_power = np.mean(enhanced_signal ** 2)
    
    if enhanced_power <= 0:
        return 0
    
    erle = 10 * np.log10(mic_power / (enhanced_power + 1e-10))
    return erle


def compute_sdr(reference_signal, estimated_signal):
    reference_signal = reference_signal.flatten()
    estimated_signal = estimated_signal.flatten()
    
    min_len = min(len(reference_signal), len(estimated_signal))
    reference_signal = reference_signal[:min_len]
    estimated_signal = estimated_signal[:min_len]
    
    s_target = (np.sum(reference_signal * estimated_signal) / (np.sum(reference_signal ** 2) + 1e-10)) * reference_signal
    
    e_res = estimated_signal - s_target
    
    sdr = 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_res ** 2) + 1e-10))
    
    return sdr


def compute_sisdr(reference_signal, estimated_signal):
    reference_signal = reference_signal.flatten()
    estimated_signal = estimated_signal.flatten()
    
    min_len = min(len(reference_signal), len(estimated_signal))
    reference_signal = reference_signal[:min_len]
    estimated_signal = estimated_signal[:min_len]
    
    scale = np.sum(reference_signal * estimated_signal) / (np.sum(reference_signal ** 2) + 1e-10)
    reference_scaled = scale * reference_signal
    
    e_noise = estimated_signal - reference_scaled
    
    sisdr = 10 * np.log10(np.sum(reference_scaled ** 2) / (np.sum(e_noise ** 2) + 1e-10))
    
    return sisdr


def compute_pesq(reference_signal, estimated_signal, sample_rate=16000):
    try:
        from pesq import pesq
        min_len = min(len(reference_signal), len(estimated_signal))
        ref = reference_signal[:min_len]
        est = estimated_signal[:min_len]
        score = pesq(sample_rate, ref, est, 'wb')
        return score
    except ImportError:
        print("PESQ library not available. Install with: pip install pesq")
        return 0.0


def compute_stoi(reference_signal, estimated_signal, sample_rate=16000):
    try:
        from pystoi import stoi
        min_len = min(len(reference_signal), len(estimated_signal))
        ref = reference_signal[:min_len]
        est = estimated_signal[:min_len]
        score = stoi(ref, est, sample_rate, extended=False)
        return score
    except ImportError:
        print("pystoi library not available. Install with: pip install pystoi")
        return 0.0


def evaluate_aec(mic_signal, enhanced_signal, near_end_signal=None, echo_signal=None, sample_rate=16000):
    results = {}
    
    results['erle'] = compute_erle(mic_signal, enhanced_signal, echo_signal)
    
    if near_end_signal is not None:
        results['sdr'] = compute_sdr(near_end_signal, enhanced_signal)
        results['sisdr'] = compute_sisdr(near_end_signal, enhanced_signal)
        results['pesq'] = compute_pesq(near_end_signal, enhanced_signal, sample_rate)
        results['stoi'] = compute_stoi(near_end_signal, enhanced_signal, sample_rate)
    
    return results