import torch

import torch.nn as nn
import torch.nn.functional as F

class LSTMNet(nn.Module):
    def __init__(self, out_channels, n_embd, n_tokens, hidden_size, num_layers, device='cpu', bidirectional=False, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.n_embd = n_embd
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.emb = nn.Embedding(n_tokens, n_embd, device=device)
        self.lstm = nn.LSTM(
            input_size=n_embd, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0, 
            device=device)
        self.task = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 100, device=device),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, out_channels, device=device)
        )

    def forward(self, x):
        x = self.emb(x)
        # x shape: (batch_size, sequence_length, embedding_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        # Take the last hidden state of the LSTM
        last_hidden_state = lstm_out[:, -1, :]
        output = self.task(last_hidden_state)
        return output

    def class_params(self):
        return {
            'out_channels': self.out_channels,
            'n_embd': self.n_embd,
            'n_tokens': self.n_tokens,
            'dropout': self.dropout,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }
