#!/usr/bin/env python3
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import soundfile as sf
import torchaudio
from torchaudio.compliance.kaldi import fbank
from timm import create_model

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CSV_PATH    = "crowtools_classified_updated.csv"
AUDIO_DIR   = "../separation/american_crow_calls"
CACHE_DIR   = "mel_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-4
SAVE_PATH  = "audiomae_finetuned_upsampled_correct.pth"
HOP_FRAMES = 512
MAX_FILES  = None          # None == use all
OVERSAMPLE_FACTOR = 5      # how many times to upweight positive clips

SR            = 16000
N_MELS        = 128
FRAME_LEN_MS  = 10.0
FRAME_SHIFT_MS= 5.0
N_FRAMES      = 1024
MEAN          = -4.2677393
STD           = 4.5689974

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
print(f"[INIT] Using device: {DEVICE}")

# -------------------------------------------------------------------------
# Attention pooling
# -------------------------------------------------------------------------
class AttnPool1d(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    def forward(self, x):
        B, W, D = x.shape
        h = self.heads
        q = self.q(x).view(B, W, h, D//h).permute(0,2,1,3)
        k = self.k(x).view(B, W, h, D//h).permute(0,2,1,3)
        v = self.v(x).view(B, W, h, D//h).permute(0,2,1,3)
        attn = torch.softmax((q @ k.transpose(-2,-1)) / np.sqrt(D//h), dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, W, D)
        return self.out(out).mean(dim=1)

# -------------------------------------------------------------------------
# SupConLoss
# -------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__(); self.temp = temp
    def forward(self, z, labels):
        B = z.size(0)
        mask = (labels @ labels.T > 0).float().to(z.device)
        mask.fill_diagonal_(0)
        logits = (z @ z.T) / self.temp
        logits = logits - logits.max(1, keepdim=True).values
        exp_log = torch.exp(logits) * (1 - torch.eye(B, device=z.device))
        log_prob = logits - torch.log(exp_log.sum(1, keepdim=True) + 1e-12)
        mean_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return -mean_pos.mean()

# -------------------------------------------------------------------------
# Dataset with Mel cache & sliding windows
# -------------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, csv_path, audio_dir, hop_frames, max_files=None):
        df = pd.read_csv(csv_path)
        if max_files:
            df = df.head(max_files)
        label_cols = ["crowCount","crowAge","quality","alert",
                      "begging","softSong","rattle","mob"]
        self.entries = []    # list of (mel_window, label_vec, clip_idx)
        self.clip_has_positive = {}  # clip_idx→True/False
        for idx, row in df.iterrows():
            fname = row["filename"]
            label = row[label_cols].astype(float).values.astype(np.float32)
            positive = label.sum() > 0
            self.clip_has_positive[idx] = positive

            path = os.path.join(audio_dir, fname)
            cache = os.path.join(CACHE_DIR, fname.replace(".wav",".npy"))
            if os.path.exists(cache):
                mel = np.load(cache)
            else:
                y, sr = sf.read(path)
                if y.ndim>1: y=y.mean(1)
                if sr!=SR:
                    y=torchaudio.functional.resample(torch.tensor(y),sr,SR).numpy()
                mel = fbank(torch.tensor(y).unsqueeze(0),
                            htk_compat=True, sample_frequency=SR,
                            use_energy=False, window_type="hanning",
                            num_mel_bins=N_MELS,
                            frame_length=FRAME_LEN_MS,
                            frame_shift=FRAME_SHIFT_MS).squeeze(0).numpy()
                np.save(cache, mel)

            T = mel.shape[0]
            if T < N_FRAMES:
                win = np.pad(mel, ((0,N_FRAMES-T),(0,0)))
                self.entries.append((win, label, idx))
            else:
                s=0
                while s+N_FRAMES<=T:
                    self.entries.append((mel[s:s+N_FRAMES],label,idx))
                    s+=hop_frames
                if s<T:
                    self.entries.append((mel[-N_FRAMES:],label,idx))
        print(f"[DATA] {len(self.entries)} windows from {len(df)} clips loaded")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self,i):
        w,label,clip_idx = self.entries[i]
        w = (torch.tensor(w)-MEAN)/STD
        w = w.unsqueeze(0).float()       # (1, T, M)
        return w, torch.tensor(label), clip_idx

# -------------------------------------------------------------------------
# Build model, optimizer
# -------------------------------------------------------------------------
base = create_model('hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
                    pretrained=True, num_classes=0).to(DEVICE)
pool = AttnPool1d(dim=base.embed_dim).to(DEVICE)
opt = torch.optim.AdamW(list(base.parameters())+list(pool.parameters()), lr=LR)
loss_fn = SupConLoss()

# -------------------------------------------------------------------------
# Prepare dataset and sampler for oversampling positives
# -------------------------------------------------------------------------
ds = WindowDataset(CSV_PATH, AUDIO_DIR, HOP_FRAMES, max_files=MAX_FILES)

# Build weight per window: high for positive clips, low for negatives
weights = []
for (_, _, clip_idx) in ds.entries:
    if ds.clip_has_positive[clip_idx]:
        weights.append(OVERSAMPLE_FACTOR)
    else:
        weights.append(1.0)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

loader = DataLoader(
    ds, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=32, pin_memory=True, persistent_workers=True
)
print(f"[DATA] DataLoader w/ oversampling ready ({len(loader)} batches)")

# -------------------------------------------------------------------------
# Training loop (same as before)
# -------------------------------------------------------------------------
for epoch in range(EPOCHS):
    base.train(); pool.train()
    epoch_loss, t0 = 0.0, time.time()
    for xb, yb, cid in loader:
        try:
            xb, yb = xb.to(DEVICE,non_blocking=True), yb.to(DEVICE)
            opt.zero_grad()
            feats = base(xb)  # (B, D)
            # group by clip_id
            grp = {}
            for f,y,c in zip(feats,yb,cid):
                grp.setdefault(int(c), []).append((f,y))
            # attention-pool per clip
            zs, ys = [], []
            for items in grp.values():
                tensor = torch.stack([t[0] for t in items]).unsqueeze(0)
                z = pool(tensor)  # (1, D)
                zs.append(z); ys.append(items[0][1].unsqueeze(0))
            z_all = F.normalize(torch.cat(zs), dim=1)
            y_all = torch.cat(ys)
            loss = loss_fn(z_all, y_all)
            loss.backward(); opt.step()
            epoch_loss += loss.item()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                # ... CPU fallback as before ...
            else:
                raise
    print(f"[EPOCH {epoch+1}] loss={epoch_loss/len(loader):.4f} time={time.time()-t0:.1f}s")

# -------------------------------------------------------------------------
# Save
# -------------------------------------------------------------------------
torch.save({'base':base.state_dict(),'pool':pool.state_dict()}, SAVE_PATH)
print(f"[SAVE] Model saved to {SAVE_PATH}")