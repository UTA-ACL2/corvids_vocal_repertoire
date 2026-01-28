import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import os
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Configuration ===
input_dir   = "../separation/american_crow_calls_half_second_silence"
output_csv  = "crow_features_output_mfcc_0.5.csv"
sr_target   = 16000
n_mfcc      = 13

# --- Extract MFCC-based features per call ---
def extract_call_features(path):
    try:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)  # convert stereo to mono
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        duration = len(y) / sr

        # === MFCCs ===
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_means = mfccs.mean(axis=1)
        mfcc_stds  = mfccs.std(axis=1)

        return {
            "filename": os.path.basename(path),
            "log_duration": np.log(duration + 1e-12),
            **{f"mfcc{i+1}_mean": mfcc_means[i] for i in range(n_mfcc)},
            **{f"mfcc{i+1}_std":  mfcc_stds[i]  for i in range(n_mfcc)},
        }

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

# === Main processing & PCA ===
if __name__ == "__main__":
    wavs    = sorted(glob(os.path.join(input_dir, "*.wav")))
    records = [extract_call_features(w) for w in wavs]
    records = [r for r in records if r is not None]
    df      = pd.DataFrame(records)

    # PCA on MFCC means and stds separately
    mean_cols = [f"mfcc{i+1}_mean" for i in range(n_mfcc)]
    std_cols  = [f"mfcc{i+1}_std"  for i in range(n_mfcc)]

    def compute_top3_pcs(df, cols, pc_name):
        X = df[cols].values
        if len(df) >= 3:
            Xs  = StandardScaler().fit_transform(X)
            pcs = PCA(n_components=3).fit_transform(Xs)
        else:
            pcs = np.stack([X[:, i] for i in range(min(3, X.shape[1]))], axis=1)
        for j in range(pcs.shape[1]):
            df[f"{pc_name}{j+1}"] = pcs[:, j]

    compute_top3_pcs(df, mean_cols, "mfcc_mean_pc")
    compute_top3_pcs(df, std_cols,  "mfcc_std_pc")

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} calls × {len(df.columns)} columns to {output_csv}")
