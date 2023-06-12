import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Note:
Using multi-layer LSTM models with MPS requires MacOS 13.0 or higher
"""

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, n_features=5, hidden_layer_size=128, output_size=1, n_layers=1, batch_size=64):

        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.n_features = n_features
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            self.n_features,
            self.hidden_layer_size,
            self.n_layers,
            bidirectional=True,
            batch_first=True
        )
        self.hidden = self.init_weights()

        self.fc = nn.Linear(2*hidden_layer_size, self.output_size)
    
    def init_weights(self):
        return (torch.zeros(self.n_layers*2, self.batch_size, self.hidden_layer_size).to(device),
                torch.zeros(self.n_layers*2, self.batch_size, self.hidden_layer_size).to(device))

    def forward(self, x):
        
        out, (ht, ct) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    