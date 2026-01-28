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

# ——— Configuration ———
AUDIO_DIR   = "../separation/american_crow_calls"
OUTPUT_CSV  = "crow_features_attnpool.csv"
CHECKPOINT  = "audiomae_posttrained.pth"

SR          = 16000
N_MELS      = 128
WIN_FRAMES  = 1024
HOP_FRAMES  = WIN_FRAMES // 2
BATCH       = 16
MEAN, STD   = -4.2677393, 4.5689974

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— Model definitions (same as training) ———
class AttnPool1d(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    def forward(self, x):           # x: (B, W, D)
        B, W, D = x.shape; h = self.heads
        q = self.q(x).view(B, W, h, D//h).permute(0,2,1,3)
        k = self.k(x).view(B, W, h, D//h).permute(0,2,1,3)
        v = self.v(x).view(B, W, h, D//h).permute(0,2,1,3)
        attn = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(D//h), dim=-1)
        out  = (attn @ v).transpose(1,2).reshape(B, W, D)
        return self.out(out).mean(dim=1)  # (B, D)

class SimpleMAE(nn.Module):
    def __init__(self, name='hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m'):
        super().__init__()
        self.encoder   = create_model(name, pretrained=False, num_classes=0)
        self.embed_dim = self.encoder.embed_dim
        self.decoder   = nn.Sequential(
            nn.Linear(self.embed_dim, WIN_FRAMES*N_MELS),
            nn.Unflatten(1, (WIN_FRAMES, N_MELS))
        )
    def forward(self, x):
        feat = self.encoder(x)
        return feat, self.decoder(feat)

# ——— Load your fine‑tuned weights ———
model     = SimpleMAE().to(device)
attn_pool = AttnPool1d(model.embed_dim).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt['model'])
attn_pool.load_state_dict(ckpt['attn_pool'])
model.eval(); attn_pool.eval()

# ——— Inference helper ———
def extract(fp):
    y, sr = sf.read(fp)
    if y.ndim>1: y = y.mean(1)
    if sr!=SR:
        y = torchaudio.functional.resample(torch.from_numpy(y).float(), sr, SR).numpy()
        sr = SR
    # mel‑fbank
    mel = fbank(torch.from_numpy(y).unsqueeze(0),
                htk_compat=True, sample_frequency=sr,
                use_energy=False, window_type='hanning',
                num_mel_bins=N_MELS,
                frame_length=10.0, frame_shift=5.0
    ).squeeze(0)  # (T,128)
    # sliding windows
    T = mel.shape[0]; wins=[]
    if T<WIN_FRAMES:
        wins=[ F.pad(mel, (0,0, 0,WIN_FRAMES-T)) ]
    else:
        for s in range(0, T-WIN_FRAMES+1, HOP_FRAMES):
            wins.append(mel[s:s+WIN_FRAMES])
    # normalize & reshape
    ts = [(w-MEAN)/STD for w in wins]
    ts = [t.unsqueeze(0).unsqueeze(0) for t in ts]  # list of (1,1,WIN,128)
    # batch‑forward
    feats=[]
    with torch.no_grad():
        for i in range(0, len(ts), BATCH):
            b = torch.cat(ts[i:i+BATCH],0).to(device)
            f, _ = model(b)
            feats.append(f.cpu())
    allf = torch.cat(feats,0)                  # (num_wins, D)
    # attention‑pool + detach → numpy
    with torch.no_grad():
        p = attn_pool(allf.unsqueeze(0).to(device)).squeeze(0)
    vec = p.detach().cpu().numpy()             # <-- detached here
    rec = {'filename':os.path.basename(fp),
           'log_duration': float(np.log(len(y)/sr + 1e-12))}
    rec.update({f'feat_{i+1}':float(v) for i,v in enumerate(vec)})
    return rec

# ——— Run over directory ———
if __name__=='__main__':
    rows=[]
    files=sorted(glob.glob(os.path.join(AUDIO_DIR,'*.wav')))
    for idx,fp in enumerate(files,1):
        print(f'[{idx}/{len(files)}] {os.path.basename(fp)}', end='')
        try:
            r=extract(fp); rows.append(r); print(' ✓')
        except Exception as e:
            print(' ✘', e)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
    print(f'\nSaved {len(rows)} → {OUTPUT_CSV}')
