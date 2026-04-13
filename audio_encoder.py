import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, embed_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        pooled = output.mean(dim=1)
        return self.fc(pooled)   # [B, 256]