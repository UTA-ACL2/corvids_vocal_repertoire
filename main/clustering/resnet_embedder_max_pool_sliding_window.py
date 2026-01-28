import os
import numpy as np
import pandas as pd
import soundfile as sf
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import librosa  # now pinned to 0.8.1

# === Configuration ===
input_dir   = "../separation/american_crow_calls"
output_csv  = "crow_features_resnet50_attention.csv"

sr_target   = 32000
n_mels      = 128
n_fft       = 2048
hop_length  = 512

window_size = 224   # frames
window_hop  = 112   # 50% overlap
batch_size  = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === ResNet-50 ⟶ 2048‑d extractor ===
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device).eval()

# === Attention pooling ===
class AttentionPooling(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim//2)
        self.fc2 = nn.Linear(dim//2, 1)
    def forward(self, x):
        # x: (N_windows, dim)
        w = F.relu(self.fc1(x))       # (N, dim//2)
        w = self.fc2(w)               # (N, 1)
        w = torch.softmax(w, dim=0)   # normalize over windows
        return (w * x).sum(dim=0)     # (dim,)

attention_pool = AttentionPooling(2048).to(device).eval()

# === ImageNet normalization ===
imagenet_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225]
)

def extract_embedding(path):
    try:
        # 1) load & resample
        y, sr = sf.read(path)
        if not isinstance(y, np.ndarray) or y.size == 0:
            raise ValueError("Empty audio")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        duration = len(y) / sr

        # 2) mel + deltas
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        log_S = librosa.power_to_db(S, ref=np.max)
        d1 = librosa.feature.delta(log_S)
        d2 = librosa.feature.delta(log_S, order=2)
        feat = np.stack([log_S, d1, d2], axis=0)  # (3, n_mels, T)

        T = feat.shape[2]
        if T < window_size:
            pad = window_size - T
            feat = np.pad(feat, ((0,0),(0,0),(0,pad)),
                          mode='constant', constant_values=log_S.min())
            T = window_size

        # 3) windows → RGB tensors
        windows = []
        for start in range(0, T - window_size + 1, window_hop):
            wv = feat[:, :, start:start+window_size]
            mn, mx = wv.min(), wv.max()
            img = ((wv - mn) / (mx - mn + 1e-9) * 255).astype(np.uint8)
            pil = Image.fromarray(np.transpose(img, (2,1,0))).resize((224,224))
            t   = transforms.ToTensor()(pil)
            windows.append(imagenet_norm(t))

        if not windows:
            raise ValueError("No windows extracted")

        # 4) batched ResNet
        all_feats = []
        for i in range(0, len(windows), batch_size):
            batch = torch.stack(windows[i:i+batch_size]).to(device)
            with torch.no_grad():
                out = resnet(batch)            # (B, 2048,1,1)
            out = out.view(out.size(0), -1)    # (B, 2048)
            all_feats.append(out.cpu())
        all_feats = torch.cat(all_feats, dim=0).to(device)  # (N, 2048)

        # 5) attention pooling
        with torch.no_grad():
            emb = attention_pool(all_feats).cpu().numpy()  # (2048,)

        # 6) assemble record
        rec = {
            "filename": os.path.basename(path),
            "log_duration": np.log(duration + 1e-12)
        }
        rec.update({f"feat_{i+1}": float(emb[i]) for i in range(emb.shape[0])})
        return rec

    except Exception as e:
        print(f"  ✘ Skipping {os.path.basename(path)}: {e}")
        return None

# === Main loop ===
if __name__ == "__main__":
    files = sorted(glob(os.path.join(input_dir, "*.wav")))
    out   = []
    for idx, f in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {os.path.basename(f)}", end="")
        rec = extract_embedding(f)
        if rec:
            out.append(rec)
            print(" ✓")
        else:
            print()

    if out:
        pd.DataFrame(out).to_csv(output_csv, index=False)
        print(f"\nSaved {len(out)} embeddings → {output_csv}")
    else:
        print("\nNo embeddings extracted.")
