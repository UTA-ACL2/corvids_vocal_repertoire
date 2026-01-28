import os
import numpy as np
import pandas as pd
import soundfile as sf
from glob import glob

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.compliance.kaldi import fbank
import torchvision.transforms as transforms
from PIL import Image
import timm

# === Configuration ===
input_dir   = "../separation/american_crow_0.5_silence"
output_csv  = "features/brachyrynchos/0.5_silence/crow_features_audiomae_brachyrynchos"

sr_target   = 16000   # AudioMAE expects 16 kHz
n_mels      = 128
win_frames  = 1024    # fixed window length in frames (≈10.24 s)
hop_frames  = win_frames // 2  # 50% overlap
batch_size  = 16      # adjust to fit your GPU memory

# AudioMAE normalization
MEAN, STD = -4.2677393, 4.5689974

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load pretrained AudioMAE ===
model = timm.create_model(
    "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
    pretrained=True,
    num_classes=0
).to(device)
model.eval()

def extract_audiomae_embedding(path):
    try:
        # 1) Load, mono, resample
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sr_target:
            y = torchaudio.functional.resample(
                torch.from_numpy(y).float(), sr, sr_target
            ).numpy()
            sr = sr_target
        duration = len(y) / sr

        # 2) Compute Mel filterbanks
        wav = torch.from_numpy(y).unsqueeze(0)  # (1, N)
        mel = fbank(
            wav,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=n_mels,
            frame_length=25.0,
            frame_shift=10.0
        ).squeeze(0)  # (T, 128)

        T = mel.shape[0]
        # 3) Slide windows
        windows = []
        if T < win_frames:
            # pad single window
            padded = F.pad(mel, (0,0, 0, win_frames-T))
            windows.append(padded)
        else:
            for start in range(0, T - win_frames + 1, hop_frames):
                windows.append(mel[start:start+win_frames])

        # 4) Normalize and reshape to (1,1,win_frames,128)
        tensors = []
        for w in windows:
            norm = (w - MEAN) / STD
            tensors.append(norm.unsqueeze(0).unsqueeze(0))  # (1,1,win,mel)

        # 5) Batch through AudioMAE
        embeddings = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.cat(tensors[i:i+batch_size], dim=0).to(device)  # (B,1,win,mel)
            with torch.no_grad():
                out = model(batch)  # (B, 768)
            embeddings.append(out.cpu())
        embeddings = torch.cat(embeddings, dim=0)  # (num_windows, 768)

        # 6) Mean‑pool across windows
        emb = embeddings.mean(dim=0).numpy()  # (768,)

        # 7) Return record
        rec = {
            "filename": os.path.basename(path),
            "log_duration": float(np.log(duration + 1e-12))
        }
        rec.update({f"audiomae_feat_{i+1}": float(emb[i]) for i in range(emb.shape[0])})
        return rec

    except Exception as e:
        print(f"  ✘ Skipping {os.path.basename(path)}: {e}")
        return None

if __name__ == "__main__":
    files = sorted(glob(os.path.join(input_dir, "*.wav")))
    rows = []
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