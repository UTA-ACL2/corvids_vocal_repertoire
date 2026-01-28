#!/usr/bin/env python3
import os, math, glob
import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.compliance.kaldi import fbank
from timm import create_model

# === Configuration ===
INPUT_DIR   = "../separation/american_crow_0.5_silence"
OUTPUT_CSV  = "crow_features_audiomae_contrastive.csv"
CHECKPOINT  = "../advanced_clustering/audiomae_contrastive_projhead.pth"

SR          = 16000
N_MELS      = 128
WIN_FRAMES  = 1024
HOP_FRAMES  = WIN_FRAMES // 2
BATCH_SIZE  = 16

MEAN, STD   = -4.2677393, 4.5689974

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- Attention‐pooling head ---
class AttnPool1d(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    def forward(self, x):             # x: (B, W, D)
        B,W,D = x.shape; h=self.heads
        q = self.q(x).view(B,W,h,D//h).permute(0,2,1,3)
        k = self.k(x).view(B,W,h,D//h).permute(0,2,1,3)
        v = self.v(x).view(B,W,h,D//h).permute(0,2,1,3)
        attn = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(D//h), dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B,W,D)
        return self.out(out).mean(dim=1)  # (B,D)

# --- Contrastive 128‐D encoder + proj head ---
class AudioMAEEncoder128(nn.Module):
    def __init__(self,
                 model_name='hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
                 proj_dim=128):
        super().__init__()
        self.encoder = create_model(model_name, pretrained=False, num_classes=0)
        d = self.encoder.embed_dim
        self.proj    = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )
    def forward(self, x):
        h = self.encoder(x)   # (B,d)
        z = self.proj(h)      # (B,128)
        return F.normalize(z, dim=1)

# --- Load model & pooling head ---
model     = AudioMAEEncoder128().to(device)
attn_pool = AttnPool1d(128).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
# ckpt contains the state_dict of the projection model itself
model.load_state_dict(ckpt)
# If you also trained attention pooling jointly, load it here:
# attn_pool.load_state_dict(ckpt['attn_pool'])

model.eval()
attn_pool.eval()
print("Loaded 128‑d proj head ✓")

# --- Extraction routine ---
def extract(path):
    # load & resample
    y, sr = sf.read(path)
    if y.ndim>1: y = y.mean(1)
    if sr!=SR:
        y = torchaudio.functional.resample(torch.from_numpy(y).float(), sr, SR).numpy(); sr=SR
    duration = len(y)/sr

    # mel‑fbank
    wav = torch.from_numpy(y).unsqueeze(0)
    mel = fbank(wav, htk_compat=True,
                sample_frequency=sr, use_energy=False,
                window_type="hanning", num_mel_bins=N_MELS,
                frame_length=25.0, frame_shift=10.0).squeeze(0)  # (T,128)

    # sliding windows
    T=mel.shape[0]; wins=[]
    if T<WIN_FRAMES:
        wins.append(F.pad(mel, (0,0, 0, WIN_FRAMES-T)))
    else:
        for s in range(0, T-WIN_FRAMES+1, HOP_FRAMES):
            wins.append(mel[s:s+WIN_FRAMES])

    # normalize & reshape
    ts = [(w-MEAN)/STD for w in wins]
    ts = [t.unsqueeze(0).unsqueeze(0) for t in ts]  # list of (1,1,WIN,128)

    # encode windows
    zs=[]
    with torch.no_grad():
        for i in range(0, len(ts), BATCH_SIZE):
            b = torch.cat(ts[i:i+BATCH_SIZE],0).to(device)  # (B,1,WIN,128)
            z = model(b)                                    # (B,128)
            zs.append(z.cpu())
    zs = torch.cat(zs,0)  # (W,128)

    # attention pool
    with torch.no_grad():
        pooled = attn_pool(zs.unsqueeze(0).to(device))  # (1,128)
    vec = pooled.squeeze(0).detach().cpu().numpy()

    # record
    rec = {"filename": os.path.basename(path),
           "log_duration": float(np.log(duration+1e-12))}
    for i,v in enumerate(vec,1):
        rec[f"feat128_{i}"] = float(v)
    return rec

# --- Main ---
if __name__=="__main__":
    fps = sorted(glob.glob(os.path.join(INPUT_DIR,"*.wav")))
    rows=[]
    for idx,fp in enumerate(fps,1):
        print(f"[{idx}/{len(fps)}] {os.path.basename(fp)}", end="")
        try:
            r = extract(fp); rows.append(r); print(" ✓")
        except Exception as e:
            print(" ✘", e)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f", lineterminator="\n")
    print(f"\nSaved {len(rows)} embeddings → {OUTPUT_CSV}")