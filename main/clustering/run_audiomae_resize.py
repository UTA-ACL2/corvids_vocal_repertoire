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
output_csv  = "features/brachyrynchos/0.5_silence/crow_features_audiomae_resize_brachyrynchos"

sr_target   = 16000   # AudioMAE expects 16 kHz
n_mels      = 128
target_frames = 1024  # Resize all to 1024 frames

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

        # 3) Resize to (target_frames, 128) with interpolation
        mel_np = mel.numpy()
        mel_img = Image.fromarray(mel_np.T)  # PIL expects (W, H), so transpose
        mel_img = mel_img.resize((target_frames, n_mels), Image.BICUBIC)
        mel_resized = np.array(mel_img).T  # Back to (T, 128)

        # 4) Normalize and reshape
        mel_tensor = torch.tensor(mel_resized).float()
        norm = (mel_tensor - MEAN) / STD
        norm = norm.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T,mel)

        # 5) AudioMAE forward
        with torch.no_grad():
            emb = model(norm)  # (1, 768)
        emb = emb.squeeze(0).cpu().numpy()  # (768,)

        # 6) Return record
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