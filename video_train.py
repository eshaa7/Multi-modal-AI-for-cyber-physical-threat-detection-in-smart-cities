

import os, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report)

# ── CONFIG ────────────────────────────────────────────────
EXTRACTED_DIR  = r'C:\Users\Sukaina\UMTD_Video_Project\extracted_frames'
SAVE_PATH      = r'C:\Users\Sukaina\UMTD_Video_Project\video_model.pth'
NORMAL_NAME    = 'NormalVideos'
SEQUENCE_LEN   = 10
FRAMES_NORMAL  = 3000
FRAMES_ANOMALY = 500
BATCH_SIZE     = 16
EPOCHS         = 20
LR             = 1e-4

random.seed(42); np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── COLLECT PATHS ─────────────────────────────────────────
def collect_paths(base_dir, split):
    # Find the split folder (Train or Test) anywhere inside base_dir
    split_dir = None
    for root, dirs, _ in os.walk(base_dir):
        if split in dirs:
            split_dir = os.path.join(root, split)
            break
    if split_dir is None:
        split_dir = base_dir  # fallback

    all_paths, all_labels = [], []
    categories = sorted([c for c in os.listdir(split_dir)
                         if os.path.isdir(os.path.join(split_dir, c))])
    print(f"\n{split}: {split_dir}")
    for cat in categories:
        cat_path = os.path.join(split_dir, cat)
        pngs = sorted([os.path.join(cat_path, f)
                       for f in os.listdir(cat_path) if f.lower().endswith('.png')])
        if not pngs: continue
        is_normal = (cat == NORMAL_NAME)
        limit = FRAMES_NORMAL if is_normal else FRAMES_ANOMALY
        if len(pngs) > limit:
            pngs = pngs[::len(pngs)//limit][:limit]
        label = 0 if is_normal else 1
        all_paths.extend(pngs)
        all_labels.extend([label]*len(pngs))
        print(f"  {cat:<20} {len(pngs):>5}  {'(normal)' if is_normal else '(anomaly)'}")
    return all_paths, all_labels

# ── DATASET ───────────────────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, paths, labels, seq_len, transform):
        self.transform = transform
        self.sequences, self.seq_labels = [], []
        i = 0
        while i + seq_len <= len(paths):
            w = labels[i:i+seq_len]
            if len(set(w)) == 1:
                self.sequences.append(paths[i:i+seq_len])
                self.seq_labels.append(w[-1])
            i += seq_len
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        frames = [self.transform(Image.open(fp).convert('RGB'))
                  for fp in self.sequences[idx]]
        return torch.stack(frames), torch.tensor(self.seq_labels[idx], dtype=torch.float32)

# ── MODEL ─────────────────────────────────────────────────
class VideoThreatDetector(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.proj = nn.Linear(128*4*4, embed_dim)
        self.pos  = nn.Parameter(torch.zeros(1,512,embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls  = nn.Sequential(nn.Linear(embed_dim,128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128,1))

    def _encode(self, x):
        B,T,C,H,W = x.shape
        x = self.cnn(x.view(B*T,C,H,W)).view(B*T,-1)
        x = self.proj(x).view(B,T,-1)
        x = self.transformer(x + self.pos[:,:T,:])
        return self.norm(x).mean(dim=1)

    def forward(self, x):        return self.cls(self._encode(x)).squeeze(1)
    def predict_prob(self, x):   return torch.sigmoid(self.forward(x))
    def get_embedding(self, x):  return self._encode(x)   # [B,256] for fusion

# ── TRAIN / EVAL ──────────────────────────────────────────
def train_epoch(model, loader, opt, crit):
    model.train(); total = 0.0
    for x,y in tqdm(loader, desc='  train', leave=False):
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss = crit(model(x), y)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); total += loss.item()
    return total/len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); probs,labs = [],[]
    for x,y in loader:
        probs.extend(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
        labs.extend(y.numpy())
    probs = np.array(probs); labs = np.array(labs,dtype=int)
    preds = (probs>=0.5).astype(int)
    return dict(acc=accuracy_score(labs,preds), f1=f1_score(labs,preds,zero_division=0),
                auc=roc_auc_score(labs,probs), probs=probs, labels=labs, preds=preds)

# ── MAIN ──────────────────────────────────────────────────
def main():
    print("="*60)
    print("  UMTD-Net | Video Module")
    print("="*60)
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available(): print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    train_paths, train_labels = collect_paths(EXTRACTED_DIR, 'Train')
    test_paths,  test_labels  = collect_paths(EXTRACTED_DIR, 'Test')

    if not train_paths:
        print(f"\nERROR: No frames found in {EXTRACTED_DIR}")
        print("Make sure the extraction step completed successfully.")
        return

    n0 = train_labels.count(0); n1 = train_labels.count(1)
    print(f"\n  Train frames : {len(train_paths):,}  (Normal:{n0}  Anomaly:{n1})")
    print(f"  Test frames  : {len(test_paths):,}")

    tr_tf = transforms.Compose([transforms.Resize((64,64)),
                                 transforms.ColorJitter(0.2,0.2),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
    te_tf = transforms.Compose([transforms.Resize((64,64)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([.5,.5,.5],[.5,.5,.5])])

    train_ds = VideoDataset(train_paths, train_labels, SEQUENCE_LEN, tr_tf)
    test_ds  = VideoDataset(test_paths,  test_labels,  SEQUENCE_LEN, te_tf)

    s0 = train_ds.seq_labels.count(0); s1 = train_ds.seq_labels.count(1)
    print(f"  Train clips  : {len(train_ds):,}  (Normal:{s0}  Anomaly:{s1})")
    print(f"  Test clips   : {len(test_ds):,}")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = VideoThreatDetector().to(DEVICE)
    print(f"  Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    pw   = torch.tensor([s0/max(s1,1)], dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    print(f"  pos_weight   : {pw.item():.4f}\n")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Acc':>8}  {'F1':>8}  {'AUC':>8}  {'Time':>6}")
    print("  " + "-"*50)

    best_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS+1):
        t0   = time.time()
        loss = train_epoch(model, train_loader, opt, crit)
        sch.step()
        m    = evaluate(model, test_loader)
        print(f"  {epoch:>5}  {loss:>8.4f}  {m['acc']*100:>7.2f}%  "
              f"{m['f1']*100:>7.2f}%  {m['auc']:>8.4f}  {time.time()-t0:>5.1f}s")
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_state = {k:v.clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    m  = evaluate(model, test_loader)
    cm = confusion_matrix(m['labels'], m['preds'])

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  Accuracy : {m['acc']*100:.2f}%")
    print(f"  F1 Score : {m['f1']*100:.2f}%")
    print(f"  AUC-ROC  : {m['auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True  Negatives : {cm[0,0]:>5,}  (normal correct)")
    print(f"    False Positives : {cm[0,1]:>5,}  (normal wrongly flagged)")
    print(f"    False Negatives : {cm[1,0]:>5,}  (threat missed)")
    print(f"    True  Positives : {cm[1,1]:>5,}  (threat caught!)")
    print()
    print(classification_report(m['labels'], m['preds'], target_names=['Normal','Anomaly']))
    print("  Paper target: Accuracy 85.2%  F1 83.5%  AUC 0.882")
    print("="*60)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n  ✅ Saved → {SAVE_PATH}")

if __name__ == '__main__':
    main()
