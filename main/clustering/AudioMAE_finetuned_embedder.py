#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torchaudio
from torchaudio.compliance.kaldi import fbank
from timm import create_model

# -----------------------------------------------------------------------------
# 1) Configuration
# -----------------------------------------------------------------------------
AUDIO_DIR      = "../separation/american_crow_calls"
CHECKPOINT     = "../advanced_clustering/audiomae_finetuned_upsampled.pth"
OUTPUT_CSV     = "AudioMAE_finetuned_embeddings.csv"
NUM_WORKERS    = 32
BATCH_SIZE     = 1   # one file at a time to capture duration

# must match your training settings:
SR             = 16000
N_MELS         = 128
FRAME_LEN_MS   = 10.0
FRAME_SHIFT_MS = 5.0
N_FRAMES       = 1024
HOP_FRAMES     = 512
MEAN           = -4.2677393
STD            = 4.5689974

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# 2) Attention‑pooling head (same as training)
# -----------------------------------------------------------------------------
class AttnPool1d(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, windows, D)
        B, W, D = x.shape
        h = self.heads
        q = self.q(x).view(B, W, h, D//h).permute(0,2,1,3)
        k = self.k(x).view(B, W, h, D//h).permute(0,2,1,3)
        v = self.v(x).view(B, W, h, D//h).permute(0,2,1,3)
        attn = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(D//h), dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, W, D)
        return self.out(out).mean(dim=1)

# -----------------------------------------------------------------------------
# 3) Dataset: scan directory for .wav and produce sliding‑window mel + duration
# -----------------------------------------------------------------------------
class DirectoryDataset(Dataset):
    def __init__(self, audio_dir):
        # collect all .wav files (non-recursive)
        self.audio_dir = audio_dir
        self.filenames = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(".wav"))
        if not self.filenames:
            raise RuntimeError(f"No .wav files found in {audio_dir!r}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn   = self.filenames[idx]
        path = os.path.join(self.audio_dir, fn)

        # 1) load audio
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = torchaudio.functional.resample(
                torch.from_numpy(y).float(), sr, SR
            ).numpy()

        # record duration
        duration = y.shape[0] / SR

        # 2) compute mel spectrogram
        mel = fbank(
            torch.from_numpy(y).unsqueeze(0),
            htk_compat=True,
            sample_frequency=SR,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=N_MELS,
            frame_length=FRAME_LEN_MS,
            frame_shift=FRAME_SHIFT_MS
        ).squeeze(0).numpy()  # shape (T, N_MELS)

        # 3) sliding windows
        T = mel.shape[0]
        windows = []
        if T < N_FRAMES:
            pad = np.pad(mel, ((0, N_FRAMES - T), (0,0)))
            windows.append(pad)
        else:
            s = 0
            while s + N_FRAMES <= T:
                windows.append(mel[s:s+N_FRAMES])
                s += HOP_FRAMES
            if s < T:
                windows.append(mel[-N_FRAMES:])

        # 4) normalize & to tensor
        windows = np.stack(windows, axis=0)                    # (W, N_FRAMES, N_MELS)
        windows = (windows - MEAN) / STD
        windows = torch.from_numpy(windows).unsqueeze(1).float()# (W,1,N_FRAMES,N_MELS)

        return fn, windows, duration

# -----------------------------------------------------------------------------
# 4) Load fine‑tuned model
# -----------------------------------------------------------------------------
base = create_model(
    'hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
    pretrained=False, num_classes=0
).to(DEVICE)

embed_dim = base.embed_dim if hasattr(base, 'embed_dim') else base.embed_dim
pool = AttnPool1d(dim=embed_dim).to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
base.load_state_dict(ckpt['base'])
pool.load_state_dict(ckpt['pool'])
base.eval(); pool.eval()
print("[MODEL] Loaded checkpoint:", CHECKPOINT)

# -----------------------------------------------------------------------------
# 5) DataLoader + extraction
# -----------------------------------------------------------------------------
ds     = DirectoryDataset(AUDIO_DIR)
loader = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS)

rows = []
with torch.no_grad():
    for fn, windows, duration in loader:
        fn   = fn[0]
        w    = windows[0].to(DEVICE)        # (W,1,F,M)
        feats= base(w)                     # (W, D)
        feats= feats.unsqueeze(0)          # (1, W, D)
        emb  = pool(feats)                 # (1, D)
        emb  = emb.squeeze(0).cpu().numpy()# (D,)

        log_dur = math.log(duration[0] + 1e-6)
        row = [fn, log_dur] + emb.tolist()
        rows.append(row)

# -----------------------------------------------------------------------------
# 6) Save to CSV
# -----------------------------------------------------------------------------
D = len(rows[0]) - 2
columns = ['filename','log_duration'] + [f'audiomae_feat_{i+1}' for i in range(D)]
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[SAVE] {len(df)} rows → {OUTPUT_CSV}")