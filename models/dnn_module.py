import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_DNN(nn.Module):
    def __init__(self, input_dim=513, hidden_dim=300, num_lstm_layers=2, num_fc_layers=2, dropout=0.0):
        super(LSTM_DNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        fc_layers = []
        in_dim = hidden_dim
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(in_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        self.output_mag = nn.Linear(hidden_dim, input_dim)
        self.output_phase = nn.Linear(hidden_dim, input_dim)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, Y_mag, X_mag):
        batch_size, seq_len, freq_bins = Y_mag.shape
        
        x = torch.cat([Y_mag, X_mag], dim=-1)
        
        lstm_out, _ = self.lstm(x)
        
        fc_out = self.fc_layers(lstm_out)
        
        X_hat_mag = self.sigmoid(self.output_mag(fc_out))
        X_hat_phase = torch.tanh(self.output_phase(fc_out))
        
        return X_hat_mag, X_hat_phase