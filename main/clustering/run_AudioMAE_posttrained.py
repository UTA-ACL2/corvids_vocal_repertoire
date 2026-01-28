#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import soundfile as sf
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.compliance.kaldi import fbank
import timm

# === Configuration ===
input_dir    = "../separation/american_crow_calls"
output_csv   = "crow_features_audiomae_posttrained.csv"
checkpoint   = "../advanced_clustering/audiomae_posttrained.pth"

sr_target    = 16000
n_mels       = 128
win_frames   = 1024
hop_frames   = win_frames // 2
batch_size   = 16

# AudioMAE normalization
MEAN, STD    = -4.2677393, 4.5689974

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Re‑define your model & pooling exactly as in training script ---
class AttnPool1d(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):  # x: (B, W, D)
        B, W, D = x.shape
        h = self.heads
        q = self.q(x).view(B, W, h, D//h).permute(0,2,1,3)
        k = self.k(x).view(B, W, h, D//h).permute(0,2,1,3)
        v = self.v(x).view(B, W, h, D//h).permute(0,2,1,3)
        attn = torch.softmax((q @ k.transpose(-2,-1)) / (D//h)**0.5, dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, W, D)
        # pool over W
        return self.out(out).mean(dim=1)  # (B, D)

class SimpleMAE(nn.Module):
    def __init__(self,
                 vit_model_name='hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m'):
        super().__init__()
        self.encoder   = timm.create_model(vit_model_name, pretrained=False, num_classes=0)
        self.embed_dim = self.encoder.embed_dim
        # decoder isn’t used at inference, but we load it anyway so state_dict matches
        self.decoder   = nn.Sequential(
            nn.Linear(self.embed_dim, win_frames * n_mels),
            nn.Unflatten(1, (win_frames, n_mels))
        )

    def forward(self, x):
        feats = self.encoder(x)          # (B, D)
        recon = self.decoder(feats)      # (B, T, M)
        return feats, recon

# --- Instantiate and load checkpoint ---
model     = SimpleMAE().to(device)
attn_pool = AttnPool1d(model.embed_dim).to(device)

checkpoint = torch.load(checkpoint, map_location=device)
model.load_state_dict(checkpoint['model'])
attn_pool.load_state_dict(checkpoint['attn_pool'])
model.eval()
attn_pool.eval()
print("Loaded fine‑tuned weights ✅")

# --- Embedding extractor uses your attn_pool for clip pooling ---
def extract_audiomae_embedding(path):
    try:
        # 1) Load + mono + resample
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sr_target:
            y = torchaudio.functional.resample(
                torch.from_numpy(y).float(), sr, sr_target
            ).numpy()
            sr = sr_target
        duration = len(y) / sr

        # 2) Mel-filterbank
        wav = torch.from_numpy(y).unsqueeze(0)
        mel = fbank(
            wav, htk_compat=True, sample_frequency=sr,
            use_energy=False, window_type="hanning",
            num_mel_bins=n_mels, frame_length=25.0, frame_shift=10.0
        ).squeeze(0)  # (T, 128)

        T = mel.shape[0]
        windows = []
        if T < win_frames:
            padded = F.pad(mel, (0,0, 0, win_frames-T))
            windows.append(padded)
        else:
            for start in range(0, T-win_frames+1, hop_frames):
                windows.append(mel[start:start+win_frames])

        # 3) Normalize & reshape
        tensors = [(w - MEAN)/STD for w in windows]
        tensors = [t.unsqueeze(0).unsqueeze(0) for t in tensors]  # list of (1,1,win,mel)

        # 4) Batch through encoder
        feats_all = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.cat(tensors[i:i+batch_size], dim=0).to(device)  # (B,1,win,mel)
            with torch.no_grad():
                feats, _ = model(batch)  # feats: (B, D)
            feats_all.append(feats.cpu())
        feats_all = torch.cat(feats_all, dim=0)  # (num_windows, D)

        # 5) Clip‑level pooling via attn_pool
        pooled = attn_pool(feats_all.unsqueeze(0).to(device))  # (1, D)

        # ─── FIX: detach() before .numpy() ───
        emb = pooled.squeeze(0).detach().cpu().numpy()

        # 6) Build record
        rec = {
            "filename":    os.path.basename(path),
            "log_duration": float(np.log(duration + 1e-12))
        }
        rec.update({f"audiomae_feat_{i+1}": float(emb[i]) for i in range(emb.shape[0])})
        return rec

    except Exception as e:
        print(f"  ✘ Skipping {os.path.basename(path)}: {e}")
        return None

# === Main loop ===
if __name__ == "__main__":
    files = sorted(glob(os.path.join(input_dir, "*.wav")))
    rows  = []
    for idx, f in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {os.path.basename(f)}", end="")
        rec = extract_audiomae_embedding(f)
        if rec:
            rows.append(rec)
            print(" ✓")
        else:
            print(" ✘")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False, lineterminator="\n", float_format="%.6f")
        print(f"\nSaved {len(rows)} embeddings → {output_csv}")
    else:
        print("\nNo embeddings extracted.")