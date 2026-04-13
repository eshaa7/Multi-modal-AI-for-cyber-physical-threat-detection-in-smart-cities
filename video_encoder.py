import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )

        self.proj = nn.Linear(128*4*4, embed_dim)

        self.pos = nn.Parameter(torch.zeros(1, 512, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = self.cnn(x.view(B*T, C, H, W))
        x = x.view(B*T, -1)

        x = self.proj(x)
        x = x.view(B, T, -1)

        x = self.transformer(x + self.pos[:, :T, :])
        x = self.norm(x)

        return x.mean(dim=1)   # [B, 256]