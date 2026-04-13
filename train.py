import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from model import UMTDNet
from dataset import MultiModalDataset

# ─────────────────────────────────────────────
# AUTO LOAD VIDEO (FOLDERS OF FRAMES)
# ─────────────────────────────────────────────
video_root = "data/video"
video_paths = []

for root, dirs, files in os.walk(video_root):
    # only pick folders that contain image files
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.png'))]

    if len(image_files) > 0:
        video_paths.append(root)

    if len(video_paths) == 10:
        break

# ─────────────────────────────────────────────
# AUTO LOAD AUDIO (.wav only)
# ─────────────────────────────────────────────
audio_root = "data/audio"
audio_paths = []

for root, dirs, files in os.walk(audio_root):
    for f in files:
        if f.endswith(".wav"):
            audio_paths.append(os.path.join(root, f))
        if len(audio_paths) == 10:
            break
    if len(audio_paths) == 10:
        break

# ─────────────────────────────────────────────
# LOAD NETWORK (ONLY 1 CSV)
# ─────────────────────────────────────────────
network_root = "data/network"
network_paths = [
    os.path.join(network_root, f)
    for f in os.listdir(network_root)
    if f.endswith(".csv")
]

# fallback safety
if len(network_paths) == 0:
    raise ValueError("❌ No network CSV found!")

network_paths = [network_paths[0]]  # use only one

# ─────────────────────────────────────────────
# PRINT INFO
# ─────────────────────────────────────────────
print(f"Found {len(video_paths)} videos, {len(audio_paths)} audio files, {len(network_paths)} network files\n")

# ─────────────────────────────────────────────
# LABELS (AUTO)
# ─────────────────────────────────────────────
def auto_label(path):
    name = os.path.basename(path).lower()
    if "normal" in name:
        return 0
    return 1

labels = [auto_label(p) for p in video_paths]

# ─────────────────────────────────────────────
# ALIGN LENGTHS
# ─────────────────────────────────────────────
min_len = min(len(video_paths), len(audio_paths))

video_paths = video_paths[:min_len]
audio_paths = audio_paths[:min_len]
labels = labels[:min_len]

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
dataset = MultiModalDataset(video_paths, audio_paths, network_paths, labels)
loader  = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = UMTDNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

print(f"Using device : {device}")
print(f"Training samples: {len(dataset)}\n")

print(f"{'Epoch':<8} {'Loss':<10} {'Acc':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'AUC-ROC'}")
print("─" * 72)

# ─────────────────────────────────────────────
# TRAIN LOOP
# ─────────────────────────────────────────────
for epoch in range(6):
    model.train()

    running_loss = 0.0
    all_labels   = []
    all_preds    = []
    all_scores   = []

    for v, a, n, label in loader:
        v, a, n, label = (
            v.to(device),
            a.to(device),
            n.to(device),
            label.to(device)
        )

        output = model(v, a, n)
        loss   = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        scores = output.detach().cpu().squeeze(1)
        preds  = (scores > 0.5).float()
        labs   = label.cpu().squeeze(1)

        all_scores.extend(scores.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labs.tolist())
        

    # ── Metrics ──────────────────────────────
    avg_loss = running_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    prec     = precision_score(all_labels, all_preds, zero_division=0)
    rec      = recall_score(all_labels, all_preds, zero_division=0)
    f1       = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_scores)
    except:
        auc = float('nan')

    print(f"{epoch+1:<8} {avg_loss:<10.4f} {acc:<10.4f} {prec:<12.4f} {rec:<10.4f} {f1:<10.4f} {auc:.4f}")

print("\nDONE ✅")
print("\n📊 Final Metrics:")
print(f"   Accuracy  : {acc*100:.2f}%")
print(f"   Precision : {prec*100:.2f}%")
print(f"   Recall    : {rec*100:.2f}%")
print(f"   F1 Score  : {f1*100:.2f}%")
print(f"   AUC-ROC   : {auc:.4f}")