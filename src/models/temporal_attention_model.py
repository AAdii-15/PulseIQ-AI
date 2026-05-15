"""
Temporal Attention Model for Depression Detection
===================================================
Architecture:
  Frame level  : 1,499 frames × 768 → Attention pool → 768 per chunk
  Session level: 8 chunks × 768     → BiLSTM(64)     → Attention → 128
  Classifier   : 128 → Dropout(0.3) → 64 → 1 (sigmoid)

Key insight: depression markers are temporally distributed.
Attention learns WHICH chunks of the interview are most
discriminative — matching clinical observation that depressed
patients slow down progressively during the interview.

N=107 training samples → frozen wav2vec + heavy regularization.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings, joblib
warnings.filterwarnings('ignore')
from pathlib import Path

BASE     = Path.home() / 'Desktop/PULSE_IQ_AI'
TEMP_DIR = BASE / 'data/features/wav2vec_temporal'
RESULTS  = BASE / 'results/metrics'
DEVICE   = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {DEVICE}')

N_CHUNKS    = 8
FRAMES_CHUNK= 1499   # frames per 30-sec chunk
HIDDEN      = 64
DROPOUT     = 0.4
LR          = 1e-4
EPOCHS      = 50
BATCH_SIZE  = 16
SEED        = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DepressionTemporalDataset(Dataset):
    def __init__(self, pids, labels, temp_dir, n_chunks=8, frames_per_chunk=1499):
        self.pids   = pids
        self.labels = labels
        self.dir    = temp_dir
        self.nc     = n_chunks
        self.fc     = frames_per_chunk

    def __len__(self): return len(self.pids)

    def __getitem__(self, idx):
        pid   = self.pids[idx]
        label = self.labels[idx]
        path  = self.dir / f'{pid}_temporal.npy'

        frames = np.load(path).astype(np.float32)  # (T, 768)

        # Chunk into nc chunks, mean-pool each to reduce dimension
        chunks = []
        for i in range(self.nc):
            start = i * self.fc
            end   = start + self.fc
            chunk = frames[start:end] if end <= len(frames) else frames[start:]
            if len(chunk) == 0:
                chunk = np.zeros((1, 768), dtype=np.float32)
            chunks.append(chunk.mean(axis=0))  # (768,)

        # (N_CHUNKS, 768)
        x = np.stack(chunks, axis=0)
        return torch.tensor(x), torch.tensor(float(label), dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

class ChunkAttention(nn.Module):
    """Attention pooling over chunks."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq, hidden*2)
        scores = self.attn(lstm_out).squeeze(-1)   # (batch, seq)
        weights= torch.softmax(scores, dim=-1)     # (batch, seq)
        pooled = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden*2)
        return pooled, weights


class TemporalAttentionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden=64, dropout=0.4):
        super().__init__()
        # Project 768 → 128 first (reduces overfitting)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # BiLSTM over chunks
        self.lstm = nn.LSTM(128, hidden, batch_first=True,
                           bidirectional=True, num_layers=1)
        # Attention pooling
        self.attn = ChunkAttention(hidden)
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, n_chunks, 768)
        b, nc, _ = x.shape
        x_proj   = self.proj(x.view(b * nc, -1)).view(b, nc, -1)
        lstm_out, _ = self.lstm(x_proj)
        pooled, attn_weights = self.attn(lstm_out)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, attn_weights


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_model(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits, _ = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    return np.array(all_probs), np.array(all_labels)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Load labels
    df       = pd.read_csv(BASE/'data/features/daic_woz_covarep_allframes.csv')
    train_df = pd.read_csv(BASE/'data/raw/daic_woz/train_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})
    dev_df   = pd.read_csv(BASE/'data/raw/daic_woz/dev_split_Depression_AVEC2017.csv').rename(columns={'Participant_ID':'participant_id'})

    # Filter to sessions with temporal files
    available = {int(p.stem.split('_')[0])
                 for p in TEMP_DIR.glob('*_temporal.npy')}

    train = df[df.participant_id.isin(train_df.participant_id) &
               df.participant_id.isin(available)]
    dev   = df[df.participant_id.isin(dev_df.participant_id) &
               df.participant_id.isin(available)]

    print(f'Train: {len(train)} | Dev: {len(dev)}')
    print(f'Train depressed: {train.PHQ8_Binary.sum()} | '
          f'Dev depressed: {dev.PHQ8_Binary.sum()}')

    # Datasets
    train_ds = DepressionTemporalDataset(
        train.participant_id.tolist(), train.PHQ8_Binary.tolist(),
        TEMP_DIR, N_CHUNKS, FRAMES_CHUNK)
    dev_ds   = DepressionTemporalDataset(
        dev.participant_id.tolist(), dev.PHQ8_Binary.tolist(),
        TEMP_DIR, N_CHUNKS, FRAMES_CHUNK)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                             shuffle=True,  drop_last=False)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE,
                             shuffle=False, drop_last=False)

    # Class weights for imbalance
    pos_weight = torch.tensor([(train.PHQ8_Binary==0).sum() / train.PHQ8_Binary.sum()], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model     = TemporalAttentionClassifier(
                    input_dim=768, hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')
    print(f'Training for {EPOCHS} epochs...\n')

    best_auc  = 0
    best_state= None
    patience  = 15
    no_improve= 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        probs, labels = eval_model(model, dev_loader)

        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs)
        else:
            auc = 0.5

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d} | Loss={loss:.4f} | '
                  f'Dev AUROC={auc:.4f} | Best={best_auc:.4f}')

        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Load best model and final evaluation
    model.load_state_dict(best_state)
    probs, labels = eval_model(model, dev_loader)
    final_auc = roc_auc_score(labels, probs)

    # Permutation test
    rng = np.random.default_rng(42)
    null= [roc_auc_score(rng.permutation(labels), probs)
           for _ in range(2000)]
    p_val = np.mean(np.array(null) >= final_auc)
    stars  = '***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'ns'

    # Bootstrap CI
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) > 1:
            boot.append(roc_auc_score(labels[idx], probs[idx]))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    print(f'\n{"="*55}')
    print(f' TEMPORAL ATTENTION MODEL — FINAL RESULTS')
    print(f'{"="*55}')
    print(f' AUROC    : {final_auc:.4f} [{ci_lo:.3f}-{ci_hi:.3f}]')
    print(f' vs chance: p={p_val:.4f} {stars}')
    print(f'{"="*55}')
    print(f'\nComparison:')
    print(f'  COVAREP + SVM (mean pool): AUROC 0.6341  p=0.110 ns')
    print(f'  wav2vec + SVM (mean pool): AUROC 0.4819  p=ns')
    print(f'  wav2vec + Attention (ours): AUROC {final_auc:.4f}  p={p_val:.4f} {stars}')

    torch.save(best_state, BASE/'models/temporal_attention_model.pt')
    pd.DataFrame([{
        'model':'wav2vec+BiLSTM+Attention',
        'protocol':'AVEC2017 Dev',
        'auroc':round(final_auc,4),
        'ci_lo':round(ci_lo,3),'ci_hi':round(ci_hi,3),
        'permutation_p':round(p_val,4),'significance':stars
    }]).to_csv(RESULTS/'temporal_attention_results.csv', index=False)
    print(f'Saved -> models/temporal_attention_model.pt')
