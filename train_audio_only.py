import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import sys

print("="*70)
print("UMTD-Net - Audio Only Training (FINAL CLEAN VERSION)")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# AudioEncoder (exact from the paper)
class AudioEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, embed_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        pooled = output.mean(dim=1)
        return self.fc(pooled)

# Correct classifier - outputs [B] shape
class AudioOnlyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AudioEncoder()
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        emb = self.encoder(x)                    # [B, 256]
        return self.classifier(emb).squeeze(1)   # [B] - correct for BCELoss

# Robust dataset loader (skips missing files)
class SimpleUrbanSoundDataset:
    def __init__(self):
        self.samples = []
        self.audio_dir = "datasets/UrbanSound8K"
        import pandas as pd
        csv_path = os.path.join(self.audio_dir, "metadata", "UrbanSound8K.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        THREAT_CLASSES = {3, 6, 8}
        
        for _, row in df.iterrows():
            fname = row["slice_file_name"]
            fold = int(row["fold"])
            cls_id = int(row["classID"])
            path = os.path.join(self.audio_dir, "audio", f"fold{fold}", fname)
            if os.path.exists(path):
                label = 1 if cls_id in THREAT_CLASSES else 0
                self.samples.append((path, label))
            else:
                print(f"⚠️  Missing file skipped: {fname}")
        
        print(f"Loaded {len(self.samples)} audio clips "
              f"({sum(1 for _,l in self.samples if l==1)} threat)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            import librosa
            import numpy as np
            y, sr = librosa.load(path, sr=22050, mono=True)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
            mel_db = mel_db.T
            if mel_db.shape[0] < 50:
                pad = np.zeros((50 - mel_db.shape[0], 128))
                mel_db = np.vstack([mel_db, pad])
            else:
                mel_db = mel_db[:50]
            return torch.tensor(mel_db, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️  Failed to load {os.path.basename(path)} - skipping")
            return torch.zeros(50, 128), torch.tensor(0, dtype=torch.float32)

# ====================== TRAINING ======================
dataset = SimpleUrbanSoundDataset()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

model = AudioOnlyClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print(f"\nStarting training on {len(train_ds)} samples for 10 epochs...\n")

best_f1 = 0.0
for epoch in range(10):
    model.train()
    for spec, labels in train_loader:
        spec, labels = spec.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(spec)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for spec, labels in val_loader:
            spec = spec.to(device)
            outputs = model(spec)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1:2d} → F1: {f1:.4f}   Acc: {acc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/best_audio_encoder.pth")
        print("   → Best model saved!")

print("\n" + "="*60)
print("✅ TRAINING FINISHED!")
print(f"Best F1-Score: {best_f1:.4f}")
print("Checkpoint saved → checkpoints/best_audio_encoder.pth")
print("="*60)