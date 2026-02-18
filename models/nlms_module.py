import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableNLMS(nn.Module):
    def __init__(self, filter_length=32, step_size=0.1, eps=1e-8):
        super(DifferentiableNLMS, self).__init__()
        self.filter_length = filter_length
        self.step_size = step_size
        self.eps = eps
        
    def forward(self, X_hat_mag, Y_mag, W_prev=None):
        batch_size, seq_len, freq_bins = Y_mag.shape
        
        if W_prev is None:
            W = torch.zeros(batch_size, self.filter_length, freq_bins, device=Y_mag.device, dtype=Y_mag.dtype)
        else:
            W = W_prev
            
        E_hat_mag = torch.zeros_like(Y_mag)
        
        for k in range(seq_len):
            if k < self.filter_length:
                pad_len = self.filter_length - k - 1
                X_pad = F.pad(X_hat_mag[:, :k+1], (0, 0, pad_len, 0))
            else:
                X_pad = X_hat_mag[:, k-self.filter_length+1:k+1]
            
            y_hat_k = torch.sum(W * X_pad, dim=1)
            
            e_k = Y_mag[:, k] - y_hat_k
            
            norm = torch.sum(X_pad ** 2, dim=1) + self.eps
            
            step = self.step_size * e_k.unsqueeze(1) * X_pad / norm.unsqueeze(1)
            W = W + step
            
            E_hat_mag[:, k] = e_k
            
        return E_hat_mag, W