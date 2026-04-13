import torch
from model import UMTDNet

print("Running test...")

model = UMTDNet()

# Dummy inputs
v = torch.randn(2, 10, 3, 64, 64)
a = torch.randn(2, 50, 128)
n = torch.randn(2, 30, 64)

# Forward pass
output = model(v, a, n)

print("SUCCESS ✅")
print("Output shape:", output.shape)