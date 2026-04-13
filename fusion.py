import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )

    def forward(self, v, a, n):
        # Stack all 3 modalities
        x = torch.stack([v, a, n], dim=1)   # [B, 3, 256]

        attn_out, _ = self.attn(x, x, x)

        return attn_out.mean(dim=1)        # [B, 256]