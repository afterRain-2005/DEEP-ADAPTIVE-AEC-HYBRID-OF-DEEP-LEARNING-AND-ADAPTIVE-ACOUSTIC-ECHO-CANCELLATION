import torch
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DeepAdaptiveAEC
from utils.signal_processing import (
    stft,
    istft,
    spec_to_mag_phase,
    mag_phase_to_spec,
    generate_synthetic_data,
    normalize_signal
)
from utils.metrics import evaluate_aec


def load_model(checkpoint_path, device='cpu', n_fft=1024):
    dnn_config = {
        'input_dim': n_fft // 2 + 1,
        'hidden_dim': 300,
        'num_lstm_layers': 2,
        'num_fc_layers': 2,
        'dropout': 0.0
    }
    
    nlms_config = {
        'filter_length': 32,
        'step_size': 0.1,
        'eps': 1e-8
    }
    
    model = DeepAdaptiveAEC(dnn_config=dnn_config, nlms_config=nlms_config, mode='joint')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def process_audio(model, far_end, mic_signal, n_fft=1024, hop_length=256, device='cpu'):
    far_end_tensor = torch.FloatTensor(far_end).to(device)
    mic_tensor = torch.FloatTensor(mic_signal).to(device)
    
    Y_spec = stft(mic_tensor.unsqueeze(0), n_fft, hop_length)
    X_spec = stft(far_end_tensor.unsqueeze(0), n_fft, hop_length)
    
    Y_mag, Y_phase = spec_to_mag_phase(Y_spec)
    X_mag, X_phase = spec_to_mag_phase(X_spec)
    
    Y_mag_batch = Y_mag.squeeze(0).permute(1, 0).unsqueeze(0)
    X_mag_batch = X_mag.squeeze(0).permute(1, 0).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(Y_mag_batch, X_mag_batch)
    
    E_hat_mag = outputs['E_hat_mag'].squeeze(0).permute(1, 0).unsqueeze(0)
    
    enhanced_spec = mag_phase_to_spec(E_hat_mag, Y_phase)
    
    enhanced_signal = istft(
        enhanced_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        length=len(mic_signal)
    )
    
    enhanced_signal = enhanced_signal.squeeze(0).cpu().numpy()
    
    return enhanced_signal, {
        'Y_mag': Y_mag.cpu().numpy(),
        'X_mag': X_mag.cpu().numpy(),
        'E_hat_mag': E_hat_mag.cpu().numpy(),
        'X_hat_mag': outputs['X_hat_mag'].cpu().numpy()
    }


def plot_results(far_end, mic_signal, enhanced_signal, near_end, echo, sample_rate, save_path=None):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    time = np.arange(len(far_end)) / sample_rate
    
    axes[0].plot(time, far_end)
    axes[0].set_title('Far-end Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    axes[1].plot(time, mic_signal)
    axes[1].set_title('Microphone Signal (Echo + Near-end + Noise)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)
    
    axes[2].plot(time, enhanced_signal)
    axes[2].set_title('Enhanced Signal')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True)
    
    axes[3].plot(time, near_end, label='Ground Truth Near-end', alpha=0.7)
    axes[3].plot(time, enhanced_signal, label='Enhanced', alpha=0.7)
    axes[3].set_title('Comparison: Ground Truth vs Enhanced')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Amplitude')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Plot saved to {save_path}')
    
    plt.show()


def plot_spectrograms(spec_data, sample_rate, hop_length, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    def plot_spec(ax, spec, title):
        im = ax.imshow(
            20 * np.log10(np.abs(spec.squeeze()) + 1e-10),
            aspect='auto',
            origin='lower',
            extent=[0, spec.shape[1] * hop_length / sample_rate, 0, sample_rate / 2]
        )
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=ax)
    
    plot_spec(axes[0, 0], spec_data['Y_mag'], 'Microphone Magnitude')
    plot_spec(axes[0, 1], spec_data['X_mag'], 'Far-end Magnitude')
    plot_spec(axes[1, 0], spec_data['X_hat_mag'], 'Estimated Nonlinear Reference (X_hat)')
    plot_spec(axes[1, 1], spec_data['E_hat_mag'], 'Enhanced Magnitude (E_hat)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Spectrogram plot saved to {save_path}')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Inference with Deep Adaptive AEC')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--n_fft', type=int, default=1024, help='STFT n_fft')
    parser.add_argument('--hop_length', type=int, default=256, help='STFT hop length')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--duration', type=float, default=5, help='Duration of test signal (seconds)')
    parser.add_argument('--ser_db', type=float, default=-10, help='Signal-to-Echo Ratio (dB)')
    parser.add_argument('--snr_db', type=float, default=10, help='Signal-to-Noise Ratio (dB)')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print('Loading model...')
    model = load_model(args.checkpoint, device=device, n_fft=args.n_fft)
    
    print('Generating test data...')
    num_frames = int(args.sample_rate * args.duration)
    
    t = np.linspace(0, args.duration, num_frames)
    far_end = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    far_end[num_frames//3:2*num_frames//3] *= 0
    
    near_end = 0.5 * np.sin(2 * np.pi * 660 * t)
    near_end[:num_frames//3] = 0
    near_end[2*num_frames//3:] = 0
    
    noise = np.random.randn(num_frames) * 0.05
    
    far_end = normalize_signal(far_end)
    near_end = normalize_signal(near_end)
    
    data = generate_synthetic_data(
        far_end=far_end,
        near_end_speech=near_end,
        noise=noise,
        snr_db=args.snr_db,
        ser_db=args.ser_db
    )
    
    print('Processing audio...')
    enhanced_signal, spec_data = process_audio(
        model,
        data['far_end'],
        data['mic'],
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        device=device
    )
    
    print('Evaluating...')
    metrics = evaluate_aec(
        data['mic'],
        enhanced_signal,
        data['near_end'],
        data['echo'],
        args.sample_rate
    )
    
    print('\nEvaluation Results:')
    for key, value in metrics.items():
        print(f'  {key.upper()}: {value:.4f}')
    
    print('\nSaving audio files...')
    sf.write(os.path.join(args.output_dir, 'far_end.wav'), data['far_end'], args.sample_rate)
    sf.write(os.path.join(args.output_dir, 'mic.wav'), data['mic'], args.sample_rate)
    sf.write(os.path.join(args.output_dir, 'enhanced.wav'), enhanced_signal, args.sample_rate)
    sf.write(os.path.join(args.output_dir, 'near_end_gt.wav'), data['near_end'], args.sample_rate)
    sf.write(os.path.join(args.output_dir, 'echo.wav'), data['echo'], args.sample_rate)
    
    print(f'Audio files saved to {args.output_dir}')
    
    if args.plot:
        print('Plotting results...')
        plot_results(
            data['far_end'],
            data['mic'],
            enhanced_signal,
            data['near_end'],
            data['echo'],
            args.sample_rate,
            save_path=os.path.join(args.output_dir, 'waveforms.png')
        )
        
        plot_spectrograms(
            spec_data,
            args.sample_rate,
            args.hop_length,
            save_path=os.path.join(args.output_dir, 'spectrograms.png')
        )


if __name__ == '__main__':
    main()