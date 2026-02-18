# Deep Adaptive AEC: Hybrid of Deep Learning and Adaptive Acoustic Echo Cancellation

This is a PyTorch implementation of the paper "DEEP ADAPTIVE AEC: HYBRID OF DEEP LEARNING AND ADAPTIVE ACOUSTIC ECHO CANCELLATION".

## Overview

The proposed method integrates adaptive filtering algorithms with modern deep learning to represent a new approach called deep adaptive AEC. The main idea is to represent the inner layers of the DNN as a differentiable NLMS adaptive linear AEC module. This enables the gradients to flow through the NLMS during DNN training to estimate the proper reference signals and step sizes.

## Architecture

- **DNN Module**: LSTM-based network (2 LSTM layers + 2 fully connected layers) that estimates the magnitude and phase of the nonlinear reference signal
- **NLMS Module**: Differentiable Normalized Least Mean Squares adaptive linear filter that performs echo cancellation
- **Hybrid Approach**: Combines the power of deep learning with the effectiveness of continuously adaptive linear AEC

## Project Structure

```
traeTry/
 models/
    __init__.py
    dnn_module.py          # LSTM-based DNN
    nlms_module.py         # Differentiable NLMS
    deep_adaptive_aec.py   # Full Deep Adaptive AEC model
 data/
    dataset.py             # Synthetic dataset generator
 utils/
    __init__.py
    signal_processing.py   # STFT/ISTFT and signal utilities
    metrics.py             # Evaluation metrics (ERLE, SDR, PESQ, etc.)
 training/
    trainer.py             # Training loop and utilities
 train.py                    # Training script
 infer.py                    # Inference script
 requirements.txt            # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --epochs 50 --batch_size 8 --lr 1e-3
```

Available arguments:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-3)
- `--train_samples`: Number of training samples (default: 1000)
- `--val_samples`: Number of validation samples (default: 100)
- `--save_dir`: Directory to save checkpoints (default: checkpoints)
- `--n_fft`: STFT n_fft size (default: 1024)
- `--hop_length`: STFT hop length (default: 256)
- `--sample_rate`: Audio sample rate (default: 16000)
- `--duration`: Duration of each sample in seconds (default: 4)
- `--mode`: Training mode ('joint' for joint echo and noise reduction, 'echo_only' for echo-only)

### Inference

```bash
python infer.py --checkpoint checkpoints/best_model.pth --plot
```

Available arguments:
- `--checkpoint`: Path to model checkpoint (required)
- `--output_dir`: Directory to save outputs (default: output)
- `--n_fft`: STFT n_fft size (default: 1024)
- `--hop_length`: STFT hop length (default: 256)
- `--sample_rate`: Audio sample rate (default: 16000)
- `--duration`: Duration of test signal in seconds (default: 5)
- `--ser_db`: Signal-to-Echo Ratio in dB (default: -10)
- `--snr_db`: Signal-to-Noise Ratio in dB (default: 10)
- `--plot`: Plot results (waveforms and spectrograms)

## Evaluation Metrics

The implementation includes the following metrics:

- **ERLE (Echo Return Loss Enhancement)**: Measures echo reduction
- **SDR (Source-to-Distortion Ratio)**: Measures overall signal quality
- **SI-SDR (Scale-Invariant Source-to-Distortion Ratio)**: Scale-invariant version of SDR
- **PESQ (Perceptual Evaluation of Speech Quality)**: Perceptual speech quality
- **STOI (Short-Time Objective Intelligibility)**: Speech intelligibility measure

## References

```
@inproceedings{zhang2021deep,
  title={Deep Adaptive AEC: Hybrid of Deep Learning and Adaptive Acoustic Echo Cancellation},
  author={Zhang, Hao and Kandar, Srivatsan and Rao, Harsha and Kim, Minje and Pruthi, Tarun and Kristjansson, Trausti},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={846--850},
  year={2021},
  organization={IEEE}
}
```

## License

This is a research implementation for educational purposes.# DEEP-ADAPTIVE-AEC-HYBRID-OF-DEEP-LEARNING-AND-ADAPTIVE-ACOUSTIC-ECHO-CANCELLATION
# DEEP-ADAPTIVE-AEC-HYBRID-OF-DEEP-LEARNING-AND-ADAPTIVE-ACOUSTIC-ECHO-CANCELLATION
# DEEP-ADAPTIVE-AEC-HYBRID-OF-DEEP-LEARNING-AND-ADAPTIVE-ACOUSTIC-ECHO-CANCELLATION
