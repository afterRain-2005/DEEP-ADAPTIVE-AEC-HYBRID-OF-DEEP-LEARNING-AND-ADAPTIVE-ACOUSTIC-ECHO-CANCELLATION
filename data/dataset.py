import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.signal_processing import generate_synthetic_data, stft, spec_to_mag_phase


class AECDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        sample_rate=16000,
        duration=4,
        n_fft=1024,
        hop_length=256,
        transform=None
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform
        
    def __len__(self):
        return self.num_samples
    
    def _generate_random_signal(self, length):
        t = np.linspace(0, 1, length)
        num_components = np.random.randint(3, 10)
        signal = np.zeros(length)
        
        for _ in range(num_components):
            freq = np.random.uniform(100, 4000)
            amplitude = np.random.uniform(0.1, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        return signal / np.max(np.abs(signal))
    
    def __getitem__(self, idx):
        num_frames = int(self.sample_rate * self.duration)
        
        far_end = self._generate_random_signal(num_frames)
        near_end = self._generate_random_signal(num_frames) * np.random.uniform(0, 1)
        noise = np.random.randn(num_frames) * 0.1
        
        ser_db = np.random.uniform(-20, 10)
        snr_db = np.random.uniform(-5, 20)
        
        data = generate_synthetic_data(
            far_end=far_end,
            near_end_speech=near_end,
            noise=noise,
            snr_db=snr_db,
            ser_db=ser_db
        )
        
        far_end_tensor = torch.FloatTensor(data['far_end'])
        mic_tensor = torch.FloatTensor(data['mic'])
        near_end_tensor = torch.FloatTensor(data['near_end'])
        echo_tensor = torch.FloatTensor(data['echo'])
        
        Y_spec = stft(mic_tensor.unsqueeze(0), self.n_fft, self.hop_length)
        X_spec = stft(far_end_tensor.unsqueeze(0), self.n_fft, self.hop_length)
        T_spec = stft(near_end_tensor.unsqueeze(0), self.n_fft, self.hop_length)
        
        Y_mag, Y_phase = spec_to_mag_phase(Y_spec)
        X_mag, X_phase = spec_to_mag_phase(X_spec)
        T_mag, T_phase = spec_to_mag_phase(T_spec)
        
        Y_mag = Y_mag.squeeze(0).permute(1, 0)
        X_mag = X_mag.squeeze(0).permute(1, 0)
        T_mag = T_mag.squeeze(0).permute(1, 0)
        
        return {
            'Y_mag': Y_mag,
            'X_mag': X_mag,
            'T_mag': T_mag,
            'far_end': far_end_tensor,
            'mic': mic_tensor,
            'near_end': near_end_tensor,
            'echo': echo_tensor
        }


def create_dataloaders(
    train_samples=1000,
    val_samples=100,
    batch_size=8,
    num_workers=0,
    **dataset_kwargs
):
    train_dataset = AECDataset(num_samples=train_samples, **dataset_kwargs)
    val_dataset = AECDataset(num_samples=val_samples, **dataset_kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader