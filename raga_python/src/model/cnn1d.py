import torch

import torch.nn as nn
import torch.nn.functional as F

class ConvNet_1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_embd, n_tokens, device='cpu', dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, n_embd, device=device)
        self.ConvNet = nn.Sequential(
            nn.Conv1d(n_embd, 32, kernel_size=kernel_size, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(32, device=device),
            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(32, 64, kernel_size=kernel_size, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(64, device=device),
            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(64, 128, kernel_size=kernel_size, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(128, device=device),
            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(128, 256, kernel_size=kernel_size, device=device),
            nn.ReLU(),
            nn.BatchNorm1d(256, device=device),
            nn.MaxPool1d(kernel_size=5),
            nn.Dropout(dropout),

            nn.Flatten()
        )

         #Fully connected layers for regression or classification
        self.task = nn.Sequential(
            nn.Linear(768, 100, device=device),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(100, out_channels, device=device)
        )

    def calculate_output_size(self, in_channels, kernel_size, stride, padding):
        # Calculate the output size after multiple convolutional and pooling layers
        output_size = (in_channels - kernel_size + 2 * padding) // stride + 1
        #print(f'output_size {output_size}')
        output_size = (output_size - kernel_size + 2 * padding) // stride + 1
        #print(f'output_size {output_size}')
        output_size = (output_size - kernel_size + 2 * padding) // stride + 1
        #print(f'output_size {output_size}')
        output_size = (output_size - kernel_size + 2 * padding) // stride + 1
        #print(f'output_size {output_size}')
        return output_size

    def forward(self, x):
        x = self.emb(x)
        sequence_length = x.shape[2]
        minibatch_length = x.shape[0]
        x = x.view(minibatch_length, sequence_length, -1)
        x = self.ConvNet(x)
        output = self.task(x)
        return output