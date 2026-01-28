#!/usr/bin/env python3
import os
import time
import glob

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchaudio.compliance.kaldi import fbank
from torch.utils.data import Dataset, DataLoader
from timm import create_model
from torch.cuda.amp import GradScaler, autocast

# ------------- Config -------------
AUDIO_DIR   = "../separation/american_crow_0.5_silence"
CACHE_DIR   = "mel_cache3"
os.makedirs(CACHE_DIR, exist_ok=True)

EPOCHS      = 20
BATCH_SIZE  = 64  # contrastive benefits from larger batches
LR          = 1e-4
SAVE_PATH   = "audiomae_contrastive_projhead.pth"

SR          = 16000
N_MELS      = 128
FRAME_LEN_MS= 10.0
FRAME_SHIFT_MS = 5.0
N_FRAMES    = 1024
HOP_FRAMES  = 512

MEAN        = -4.2677393
STD         = 4.5689974
PROJ_DIM    = 128
# ----------------------------------

# -------- Dataset --------
class SlidingWindowDataset(Dataset):
    def __init__(self, audio_dir, cache_dir, n_frames, hop_frames,
                 sr=SR, n_mels=N_MELS,
                 frame_len_ms=FRAME_LEN_MS, frame_shift_ms=FRAME_SHIFT_MS,
                 mean=MEAN, std=STD):
        self.files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        self.cache_dir = cache_dir
        self.n_frames, self.hop_frames = n_frames, hop_frames
        self.sr, self.n_mels = sr, n_mels
        self.frame_len_ms, self.frame_shift_ms = frame_len_ms, frame_shift_ms
        self.mean, self.std = mean, std
        self.entries = []
        print("[DATA] Preparing windows...")
        for fp in self.files:
            fname = os.path.basename(fp).replace('.wav', '.npy')
            cache = os.path.join(cache_dir, fname)
            if os.path.exists(cache):
                mel = np.load(cache)
            else:
                y, sr0 = sf.read(fp)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr0 != sr:
                    y = torchaudio.functional.resample(
                        torch.tensor(y), sr0, sr
                    ).numpy()
                mel = fbank(
                    torch.tensor(y).unsqueeze(0),
                    htk_compat=True,
                    sample_frequency=sr,
                    use_energy=False,
                    window_type="hanning",
                    num_mel_bins=n_mels,
                    frame_length=frame_len_ms,
                    frame_shift=frame_shift_ms
                ).squeeze(0).numpy()
                np.save(cache, mel)
            T = mel.shape[0]
            if T < n_frames:
                padded = np.pad(mel, ((0, n_frames - T), (0, 0)))
                self.entries.append(padded)
            else:
                s = 0
                while s + n_frames <= T:
                    self.entries.append(mel[s:s+n_frames])
                    s += hop_frames
                if s < T:
                    self.entries.append(mel[-n_frames:])
        print(f"[DATA] Total windows: {len(self.entries)}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        mel = self.entries[idx]
        mel_t = (torch.tensor(mel).float() - self.mean) / self.std
        return mel_t.unsqueeze(0)

# -------- Augmentations --------
freq_mask = FrequencyMasking(freq_mask_param=30)
time_mask = TimeMasking(time_mask_param=40)

def augment(x: torch.Tensor) -> torch.Tensor:
    x_aug = x.clone()
    for i in range(x.size(0)):
        m = x_aug[i, 0]
        m = freq_mask(m)
        m = time_mask(m)
        x_aug[i, 0] = m
    return x_aug

# -------- Contrastive Loss --------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        mask = (~torch.eye(2 * N, device=sim.device).bool()).float()
        exp_sim = torch.exp(sim) * mask
        denom = exp_sim.sum(dim=1)
        pos = torch.exp((z_i * z_j).sum(dim=1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        return -torch.log(pos / denom).mean()

# -------- Model with Projection Head --------
class AudioMAEEncoder(nn.Module):
    def __init__(self, model_name='hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m', proj_dim=PROJ_DIM):
        super().__init__()
        self.encoder = create_model(model_name, pretrained=True, num_classes=0)
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )

    def forward(self, x):
        emb = self.encoder(x)
        emb = self.proj(emb)
        emb = F.normalize(emb, dim=1)
        return emb

# -------- Training Loop with AMP, Multi-GPU, and CPU Fallback --------
def train():
    dataset = SlidingWindowDataset(AUDIO_DIR, CACHE_DIR, N_FRAMES, HOP_FRAMES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=32, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioMAEEncoder()
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = ContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()

    print(f"[TRAIN] epochs={EPOCHS}, batch={BATCH_SIZE}, device={device}")
    cpu_fallback = False

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, mel_win in enumerate(loader, 1):
            mel_win = mel_win.to(device, non_blocking=True)
            v1 = augment(mel_win)
            v2 = augment(mel_win)

            try:
                with autocast():
                    z1 = model(v1)
                    z2 = model(v2)
                    loss = criterion(z1, z2)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            except RuntimeError as e:
                if 'out of memory' in str(e) and not cpu_fallback:
                    print("[WARN] CUDA OOM during encoding: falling back to CPU")
                    torch.cuda.empty_cache()
                    cpu_fallback = True
                    device = torch.device('cpu')
                    model = model.module if isinstance(model, nn.DataParallel) else model
                    model.to(device)
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    mel_win = mel_win.cpu()
                    v1, v2 = v1.cpu(), v2.cpu()
                    z1 = model(v1)
                    z2 = model(v2)
                    loss = criterion(z1, z2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    raise

            total_loss += loss.item() * mel_win.size(0)
            if step % 50 == 0:
                avg = total_loss / (step * BATCH_SIZE)
                print(f"  [Epoch {epoch}] step {step}/{len(loader)} loss={avg:.4f}")

        avg_epoch = total_loss / len(dataset)
        print(f"[Epoch {epoch}] avg_loss={avg_epoch:.4f} time={time.time()-t0:.1f}s")

    model_final = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(model_final.state_dict(), SAVE_PATH)
    print(f"[SAVE] Contrastive encoder with proj head -> {SAVE_PATH}")

if __name__ == '__main__':
    train()