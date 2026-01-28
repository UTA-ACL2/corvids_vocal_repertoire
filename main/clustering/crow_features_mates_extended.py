#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from scipy.signal import hilbert
from scipy.stats import moment
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Configuration ===
input_dir    = "../separation/american_crow_0.5_silence"
output_csv   = "crow_features_mates_brachyrynchos_raw_1200.csv"

# Pitch extraction
min_pitch    = 100       # Hz
max_pitch    = 1200      # Hz

# Harmonic analysis
n_harmonics  = 12
frame_size   = 2048
hop_size     = 512

# Numerical stability
epsilon      = 1e-12

# === Harmonic extractor ===
def extract_harmonic_energies(y, sr, f0):
    if len(y) < frame_size:
        return np.zeros((1, n_harmonics))
    frames = []
    for i in range(0, len(y) - frame_size + 1, hop_size):
        frm = y[i:i+frame_size] * np.hanning(frame_size)
        frames.append(frm)
    if not frames:
        return np.zeros((1, n_harmonics))
    H = []
    freqs = np.fft.rfftfreq(frame_size, d=1/sr)
    for frm in frames:
        spec = np.abs(np.fft.rfft(frm))
        row = []
        for h in range(1, n_harmonics+1):
            target_f = f0 * h
            if target_f > freqs[-1]:
                row.append(0.0)
            else:
                idx = np.argmin(np.abs(freqs - target_f))
                row.append(spec[idx])
        H.append(row)
    return np.array(H)

# === Feature extraction ===
def extract_call_features(path):
    try:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        duration = len(y) / sr

        snd       = parselmouth.Sound(path)
        pitch_obj = call(snd, "To Pitch", 0.0, min_pitch, max_pitch)
        n_frames  = pitch_obj.get_number_of_frames()
        pitches   = np.array([
            pitch_obj.get_value_in_frame(i+1) or 0.0
            for i in range(n_frames)
        ])
        pitches_pos = pitches[pitches > 0]

        if len(pitches_pos) < 4:
            mean_f0, p95, p05, max_f0, peak_loc = 0.0, 0.0, 0.0, 0.0, 0.0
            coeffs, residual = [0]*4, np.zeros_like(pitches_pos)
            wobble_f = wobble_p1 = wobble_p2 = instability = 0.0
        else:
            mean_f0 = pitches_pos.mean()
            p95     = np.percentile(pitches_pos, 95)
            p05     = np.percentile(pitches_pos, 5)
            max_f0  = pitches_pos.max()
            peak_loc = np.argmax(pitches_pos) / len(pitches_pos)
            tnorm   = np.linspace(0, 1, len(pitches_pos))
            coeffs  = np.polyfit(tnorm, pitches_pos, 3)
            fitted  = np.polyval(coeffs, tnorm)
            residual = pitches_pos - fitted
            instability = np.sqrt(np.mean(residual**2))
            fft_r = np.abs(np.fft.rfft(residual))
            wobble_f  = np.argmax(fft_r[1:]) + 1 if len(fft_r) > 1 else 0
            wobble_p1 = 1 / (np.std(residual) + epsilon)
            wobble_p2 = 1 / (np.var(residual) + epsilon)

        analytic = hilbert(y)
        amp_env  = np.abs(analytic)
        m2 = moment(amp_env, moment=2)
        m3 = moment(amp_env, moment=3)
        m4 = moment(amp_env, moment=4)
        
        fft_a = np.abs(np.fft.rfft(amp_env))
        amp_wf = np.argmax(fft_a[1:]) + 1 if len(fft_a) > 1 else 0
        amp_wm = fft_a.max() if len(fft_a) > 0 else 0

        H_raw   = extract_harmonic_energies(y, sr, mean_f0)
        H_rel   = H_raw / (H_raw.sum(axis=1, keepdims=True) + epsilon)
        avg_rel = H_rel.mean(axis=0)
        trends  = np.array([
            np.polyfit(np.arange(H_rel.shape[0]), H_rel[:,h], 1)[0]
            if H_rel.shape[0] > 1 else 0.0
            for h in range(n_harmonics)
        ])

        features = {
            "filename":                   os.path.basename(path),
            "mean_f0":                    mean_f0,
            "f0_95th_percentile":         p95,
            "f0_5th_percentile":          p05,
            "f0_peak_location":           peak_loc,
            "f0_peak_value":              max_f0,
            "f0_quadratic_term":          coeffs[2],
            "f0_cubic_term":              coeffs[0],
            "pitch_wobble_frequency":     wobble_f,
            "pitch_wobble_periodicity_1": wobble_p1,
            "pitch_wobble_periodicity_2": wobble_p2,
            "pitch_instability":          instability,
            "call_length":                duration,
            "amplitude_moment_2":         m2,
            "amplitude_moment_3":         m3,
            "amplitude_moment_4":         m4,
            "amplitude_wobble_frequency": amp_wf,
            "amplitude_wobble_magnitude": amp_wm,
        }

        for i in range(n_harmonics):
            features[f"avg_rel_h{i+1}"] = avg_rel[i]
            features[f"trend_h{i+1}"]   = trends[i]

        return features

    except Exception as e:
        with open("feature_errors.log", "a") as log:
            log.write(f"{path}\t{str(e)}\n")
        return None

def compute_top3_pcs(df, cols, prefix):
    if len(df) >= 3 and all(c in df.columns for c in cols):
        X  = df[cols].values
        Xs = StandardScaler().fit_transform(X)
        pcs = PCA(n_components=3).fit_transform(Xs)
        for j in range(3):
            df[f"{prefix}{j+1}"] = pcs[:, j]
        df.drop(columns=cols, inplace=True)
    else:
        print(f"⚠️ Skipping PCA for {prefix}: too few rows or missing columns")

# === Main ===
if __name__ == "__main__":
    wavs = sorted(glob(os.path.join(input_dir, "*.wav")))
    print(f"Extracting features from {len(wavs)} calls...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        results = list(tqdm(
            exe.map(extract_call_features, wavs),
            total=len(wavs),
            desc="Processing clips"
        ))

    records = [r for r in results if r is not None]
    print(f"✅ Extracted for {len(records)}/{len(wavs)} calls")

    df = pd.DataFrame(records)

    avg_cols   = [c for c in df.columns if c.startswith("avg_rel_h")]
    trend_cols = [c for c in df.columns if c.startswith("trend_h")]

    if avg_cols:
        compute_top3_pcs(df, avg_cols,   prefix="harmonic_pc")
    if trend_cols:
        compute_top3_pcs(df, trend_cols, prefix="harmonic_trend_pc")

    df.to_csv(output_csv, index=False)
    print(f"✅ Done: saved {df.shape[0]} × {df.shape[1]} → {output_csv}")