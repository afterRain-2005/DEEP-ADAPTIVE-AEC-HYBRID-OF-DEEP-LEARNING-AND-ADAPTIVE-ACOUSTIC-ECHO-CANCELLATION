import torch
import torch.nn as nn
import torch.nn.functional as F
from .dnn_module import LSTM_DNN
from .nlms_module import DifferentiableNLMS


class DeepAdaptiveAEC(nn.Module):
    def __init__(self, dnn_config=None, nlms_config=None, mode='joint'):
        super(DeepAdaptiveAEC, self).__init__()
        
        if dnn_config is None:
            dnn_config = {
                'input_dim': 513,
                'hidden_dim': 300,
                'num_lstm_layers': 2,
                'num_fc_layers': 2,
                'dropout': 0.0
            }
        
        if nlms_config is None:
            nlms_config = {
                'filter_length': 32,
                'step_size': 0.1,
                'eps': 1e-8
            }
        
        self.mode = mode
        
        self.dnn = LSTM_DNN(**dnn_config)
        self.nlms = DifferentiableNLMS(**nlms_config)
        
    def forward(self, Y_mag, X_mag, W_prev=None):
        X_hat_mag, X_hat_phase = self.dnn(Y_mag, X_mag)
        
        E_hat_mag, W = self.nlms(X_hat_mag, Y_mag, W_prev)
        
        return {
            'E_hat_mag': E_hat_mag,
            'X_hat_mag': X_hat_mag,
            'X_hat_phase': X_hat_phase,
            'W': W
        }
    
    def compute_loss(self, outputs, targets):
        if self.mode == 'joint':
            loss = F.mse_loss(outputs['E_hat_mag'], targets['target_mag'])
        else:
            loss = F.mse_loss(outputs['E_hat_mag'], targets['target_mag'])
        
        return loss