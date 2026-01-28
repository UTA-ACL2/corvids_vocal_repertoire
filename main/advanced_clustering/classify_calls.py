#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd

# === Paths setup ===
BASE_DIR        = os.path.dirname(__file__)
CLASSIFIER_DIR  = os.path.join(BASE_DIR, "crow-tools", "classifier")
sys.path.insert(0, CLASSIFIER_DIR)

from model import CrowClassifier

# === Configuration ===
EMBEDDINGS_CSV   = os.path.join(BASE_DIR, "../clustering/aves_embeddings_base_bio.csv")
CLUSTER_CSV      = os.path.join(BASE_DIR, "../clustering/crow_cluster_probabilities_70_mates.csv")
OUTPUT_CSV       = os.path.join(BASE_DIR, "crowtools_classified.csv")
CHECKPOINT_PATH  = os.path.join(CLASSIFIER_DIR, "models", "best_model.ckpt")

# === Device & model loading ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[+] Using device: {device}")

model = CrowClassifier.load_from_checkpoint(CHECKPOINT_PATH).to(device)
model.eval()
print(f"[+] Loaded CrowClassifier from checkpoint:\n    {CHECKPOINT_PATH}")

# === Load CSVs ===
print(f"[+] Reading embeddings from: {EMBEDDINGS_CSV}")
df_emb    = pd.read_csv(EMBEDDINGS_CSV)
print(f"[+] Reading cluster probs from: {CLUSTER_CSV}")
df_clust  = pd.read_csv(CLUSTER_CSV)

# Ensure filename exists
if "filename" not in df_emb.columns or "filename" not in df_clust.columns:
    raise ValueError("Both CSVs must include a 'filename' column.")

# Determine embedding columns
embed_cols = [c for c in df_emb.columns if c != "filename"]
if len(embed_cols) != 768:
    raise ValueError(f"Expected 768 embedding dims, found {len(embed_cols)}")

# Determine cluster columns and pseudo‑rattle
cluster_cols = [c for c in df_clust.columns if c.startswith("cluster_")]
if "cluster_6" not in cluster_cols:
    raise ValueError("cluster_probabilities.csv must include a 'cluster_6' column.")

# Compute pseudo_rattle: True if cluster_6 is the highest-prob
df_clust["pseudo_rattle"] = (
    df_clust[cluster_cols].idxmax(axis=1) == "cluster_6"
)

# Merge embeddings with pseudo‑rattle labels
df = df_emb.merge(
    df_clust[["filename", "pseudo_rattle"]],
    on="filename",
    how="inner"
)
print(f"[+] Merged data: {len(df)} rows (with pseudo_rattle)")

# === Inference helper ===
def predict(emb: np.ndarray, pseudo_rattle: bool):
    """
    emb: 1D numpy array of length 768
    pseudo_rattle: whether this file is cluster_6 (True/False)
    """
    x = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)

    return {
        "crowCount":  int(torch.argmax(out["crowCount"], dim=1).item()),
        "crowAge":    int(torch.argmax(out["crowAge"],   dim=1).item() + 1),
        "quality":    int(torch.argmax(out["quality"],   dim=1).item() + 1),
        "alert":      bool((out["alert"].squeeze()    > 0).item()),
        "begging":    bool((out["begging"].squeeze()  > 0).item()),
        "softSong":   bool((out["softSong"].squeeze() > 0).item()),
        # overwrite rattle with our cluster-6 proxy:
        "rattle":    bool(pseudo_rattle),
        "mob":        bool((out["mob"].squeeze()      > 0).item()),
    }

# === Run inference ===
results = []
total = len(df)
for idx, row in df.iterrows():
    emb = row[embed_cols].values
    pseudo = row["pseudo_rattle"]
    pred = predict(emb, pseudo)
    pred["filename"] = row["filename"]
    results.append(pred)

    if (idx + 1) % 500 == 0 or idx == total - 1:
        print(f"  • Processed {idx+1}/{total}")

# === Save output ===
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"[+] Saved classifications to: {OUTPUT_CSV}")