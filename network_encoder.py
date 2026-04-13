import torch
import torch.nn as nn

class NetworkEncoder(nn.Module):
    def __init__(self, input_dim=40, embed_dim=256, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()

        self.proj = nn.Linear(input_dim, embed_dim)

        self.pos = nn.Parameter(torch.zeros(1, 512, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            embed_dim * 4,
            dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x) + self.pos[:, :x.shape[1], :]
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)

        return x   # [B, 256]