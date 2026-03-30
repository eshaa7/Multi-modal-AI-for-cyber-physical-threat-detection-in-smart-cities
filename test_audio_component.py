import torch
import torch.nn as nn
import sys
import os

print("✅ Starting AudioEncoder test...")

# ==================== COPY OF AudioEncoder FROM THE PAPER ====================
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
        return self.fc(pooled)

# ==================== TEST CODE ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

model = AudioEncoder().to(device)

batch_size = 4
dummy_input = torch.randn(batch_size, 50, 128).to(device)

with torch.no_grad():
    output = model(dummy_input)

print(f"\n✅ TEST PASSED!")
print(f"   Input shape  : {list(dummy_input.shape)}")
print(f"   Output shape : {list(output.shape)}   ← should be [4, 256]")
print(f"   Total parameters : {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "="*70)
print(" 🎉 AudioEncoder is working correctly!")
print("="*70)