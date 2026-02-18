import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DeepAdaptiveAEC
from data.dataset import create_dataloaders
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train Deep Adaptive AEC')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=100, help='Number of validation samples')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--n_fft', type=int, default=1024, help='STFT n_fft')
    parser.add_argument('--hop_length', type=int, default=256, help='STFT hop length')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--duration', type=float, default=4, help='Duration of each sample (seconds)')
    parser.add_argument('--mode', type=str, default='joint', choices=['joint', 'echo_only'], help='Training mode')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    dnn_config = {
        'input_dim': args.n_fft // 2 + 1,
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
    
    model = DeepAdaptiveAEC(dnn_config=dnn_config, nlms_config=nlms_config, mode=args.mode)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    train_loader, val_loader = create_dataloaders(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        duration=args.duration
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir
    )
    
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()