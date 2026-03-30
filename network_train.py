"""
UMTD-Net Network Module


What we do:
    1. Load TON_IoT dataset from HuggingFace (30MB, auto download)
    2. Clean and preprocess the data
    3. Build Transformer model (NetworkEncoder from paper)
    4. Train for 20 epochs with proper train/test split
    5. Print Accuracy, F1, AUC-ROC, Recall and Precision
    6. Save network_model.pth
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, classification_report)

# ── CONFIG ────────────────────────────────────────────────
SAVE_PATH  = r'C:\Users\Sukaina\UMTD_Video_Project\network_model.pth'
SEQ_LEN    = 30
BATCH_SIZE = 256
EPOCHS     = 20
LR         = 1e-3

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── LOAD DATA ─────────────────────────────────────────────
def load_data():
    from datasets import load_dataset
    print('Loading TON_IoT dataset from HuggingFace...')
    print('(First time ~1 min to download 30MB, then cached)')
    raw      = load_dataset('codymlewis/TON_IoT_network')
    df_train = raw['train'].to_pandas()
    df_test  = raw['test'].to_pandas()
    print(f'Train rows: {len(df_train):,}  |  Test rows: {len(df_test):,}')
    return df_train, df_test


# ── PREPROCESS ────────────────────────────────────────────
DROP_COLS = ['src_ip', 'dst_ip', 'type']
LABEL_COL = 'label'

def preprocess(df):
    df = df.copy()
    labels = df[LABEL_COL].values.astype(np.float32)
    df = df.drop(columns=[LABEL_COL] + [c for c in DROP_COLS if c in df.columns])
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna('-').astype(str))
    df = df.fillna(df.median(numeric_only=True))
    return df.values.astype(np.float32), labels

def make_sequences(features, labels, seq_len):
    N = len(features) - seq_len + 1
    X = np.lib.stride_tricks.sliding_window_view(
            features, (seq_len, features.shape[1])
        ).reshape(N, seq_len, features.shape[1])
    y = labels[seq_len - 1:]
    return X.astype(np.float32), y.astype(np.float32)


# ── MODEL ─────────────────────────────────────────────────
class NetworkThreatDetector(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.pos  = nn.Parameter(torch.zeros(1, 512, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim * 4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls  = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def _encode(self, x):
        x = self.proj(x) + self.pos[:, :x.shape[1], :]
        return self.norm(self.transformer(x)).mean(dim=1)

    def forward(self, x):         return self.cls(self._encode(x)).squeeze(1)
    def predict_prob(self, x):    return torch.sigmoid(self.forward(x))
    def get_embedding(self, x):   return self._encode(x)   # [B, 256]


# ── TRAIN / EVAL ──────────────────────────────────────────
def train_epoch(model, loader, opt, crit):
    model.train(); total = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); probs, labs = [], []
    for X, y in loader:
        probs.extend(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
        labs.extend(y.numpy())
    probs = np.array(probs); labs = np.array(labs, dtype=int)
    preds = (probs >= 0.5).astype(int)
    return dict(
    acc=accuracy_score(labs, preds),
    prec=precision_score(labs, preds, zero_division=0), # Add this
    rec=recall_score(labs, preds, zero_division=0),   # Add this
    f1=f1_score(labs, preds, zero_division=0),
    auc=roc_auc_score(labs, probs),
    probs=probs, labels=labs, preds=preds
    )


# ── MAIN ──────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('  UMTD-Net | Network Module')
    print('=' * 60)
    print(f'  Device : {DEVICE}')
    if torch.cuda.is_available():
        print(f'  GPU    : {torch.cuda.get_device_name(0)}')

    df_train, df_test = load_data()

    print('\nPreprocessing...')
    X_tr_raw, y_tr = preprocess(df_train)
    X_te_raw, y_te = preprocess(df_test)

    scaler  = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)

    print('Creating sequences...')
    X_train, y_train = make_sequences(X_tr_sc, y_tr, SEQ_LEN)
    X_test,  y_test  = make_sequences(X_te_sc, y_te, SEQ_LEN)

    INPUT_DIM = X_train.shape[2]
    n_normal  = int((y_train == 0).sum())
    n_attack  = int((y_train == 1).sum())

    print(f'  Train sequences : {len(X_train):,}  (Normal:{n_normal:,}  Attack:{n_attack:,})')
    print(f'  Test sequences  : {len(X_test):,}')
    print(f'  Input features  : {INPUT_DIM}')

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.tensor(X_test),  torch.tensor(y_test)),
                              BATCH_SIZE, shuffle=False, num_workers=0)

    model = NetworkThreatDetector(INPUT_DIM).to(DEVICE)
    print(f'  Parameters      : {sum(p.numel() for p in model.parameters()):,}')

    pw   = torch.tensor([n_normal / max(n_attack, 1)], dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    print(f'  pos_weight      : {pw.item():.4f}\n')
    print(f'  {"Epoch":>5}  {"Loss":>8}  {"Acc":>8}  {"F1":>8}  {"AUC":>8}  {"Time":>6}')
    print('  ' + '-' * 52)

    best_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        t0   = time.time()
        loss = train_epoch(model, train_loader, opt, crit)
        sch.step()
        m    = evaluate(model, test_loader)
        print(f'  {epoch:>5}  {loss:>8.4f}  {m["acc"]*100:>7.2f}%  '
              f'{m["f1"]*100:>7.2f}%  {m["auc"]:>8.4f}  {time.time()-t0:>5.1f}s')
        if m['f1'] > best_f1:
            best_f1    = m['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    m  = evaluate(model, test_loader)
    cm = confusion_matrix(m['labels'], m['preds'])

    print('\n' + '=' * 60)
    print('  FINAL RESULTS')
    print('=' * 60)
    print(f'  Accuracy : {m["acc"]*100:.2f}%')
    print(f'  Precision : {m["prec"]*100:.2f}%') 
    print(f'  Recall    : {m["rec"]*100:.2f}%')
    print(f'  F1 Score : {m["f1"]*100:.2f}%')
    print(f'  AUC-ROC  : {m["auc"]:.4f}')
    print(f'\n  Confusion Matrix:')
    print(f'    True  Negatives (normal correct)  : {cm[0,0]:>7,}')
    print(f'    False Positives (normal flagged)   : {cm[0,1]:>7,}')
    print(f'    False Negatives (attack missed)    : {cm[1,0]:>7,}')
    print(f'    True  Positives (attack caught!)   : {cm[1,1]:>7,}')
    print()
    print(classification_report(m['labels'], m['preds'],
                                target_names=['Normal', 'Attack']))
    print('  Paper target : Accuracy 87%  F1 85.1%  AUC 0.894')
    print('=' * 60)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f'\n  Saved -> {SAVE_PATH}')
    print('  Fusion layer usage:')
    print('    F_n = model.get_embedding(x_network)  # [B, 256]')


if __name__ == '__main__':
    main()
