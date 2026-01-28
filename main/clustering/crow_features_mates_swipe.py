#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, lfilter, hilbert
from scipy.stats import moment
import pysptk
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Config ===
input_dir  = "../separation/american_crow_0.5_silence"
output_csv = "crow_features_swipe_brachyrynchos_full.csv"

sr = 22050
frame_size = 1024
hop_size = 256
n_harmonics = 12
epsilon = 1e-12


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=100, highcut=5000, fs=sr, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)


def extract_swipe_pitch(y, sr):
    y = bandpass_filter(y, 100, 5000, sr)
    f0 = pysptk.swipe(y.astype(np.float64), fs=sr, hopsize=hop_size,
                      min=100, max=1200, otype="f0")
    return np.nan_to_num(f0)


def extract_harmonic_energies(y, sr, f0):
    if len(y) < frame_size or f0 <= 0:
        return np.zeros((1, n_harmonics))
    freqs = np.fft.rfftfreq(frame_size, 1 / sr)
    H = []
    for start in range(0, len(y) - frame_size + 1, hop_size):
        frame = y[start:start + frame_size] * np.hanning(frame_size)
        spec = np.abs(np.fft.rfft(frame))
        row = [spec[np.argmin(np.abs(freqs - f0 * h))] if (f0 * h) <= freqs[-1] else 0.0 for h in range(1, n_harmonics + 1)]
        H.append(row)
    return np.array(H)


def extract_features(path):
    try:
        y, file_sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            raise ValueError(f"All files must be {sr} Hz.")
        duration = len(y) / sr

        f0 = extract_swipe_pitch(y, sr)
        pitches_pos = f0[f0 > 0]

        if len(pitches_pos) < 4:
            mean_f0 = p95 = p05 = max_f0 = peak_loc = 0.0
            coeffs = [0, 0, 0, 0]
            residual = np.zeros(0)
            wobble_f = wobble_p1 = wobble_p2 = instability = 0.0
        else:
            mean_f0 = pitches_pos.mean()
            p95 = np.percentile(pitches_pos, 95)
            p05 = np.percentile(pitches_pos, 5)
            max_f0 = pitches_pos.max()
            peak_loc = np.argmax(pitches_pos) / len(pitches_pos)
            tnorm = np.linspace(0, 1, len(pitches_pos))
            coeffs = np.polyfit(tnorm, pitches_pos, 3)
            fitted = np.polyval(coeffs, tnorm)
            residual = pitches_pos - fitted
            instability = np.sqrt(np.mean(residual ** 2))
            fft_r = np.abs(np.fft.rfft(residual))
            wobble_f = np.argmax(fft_r[1:]) + 1 if len(fft_r) > 1 else 0
            wobble_p1 = 1 / (np.std(residual) + epsilon)
            wobble_p2 = 1 / (np.var(residual) + epsilon)

        # Amplitude envelope
        amp_env = np.abs(hilbert(y))
        m2 = moment(amp_env, moment=2)
        m3 = moment(amp_env, moment=3)
        m4 = moment(amp_env, moment=4)
        fft_a = np.abs(np.fft.rfft(amp_env))
        amp_wf = np.argmax(fft_a[1:]) + 1 if len(fft_a) > 1 else 0
        amp_wm = fft_a.max() if len(fft_a) > 0 else 0

        # Harmonics
        H_raw = extract_harmonic_energies(y, sr, mean_f0)
        H_rel = H_raw / (H_raw.sum(axis=1, keepdims=True) + epsilon)
        avg_rel = H_rel.mean(axis=0)
        trends = [np.polyfit(np.arange(H_rel.shape[0]), H_rel[:, h], 1)[0]
                  if H_rel.shape[0] > 1 else 0.0 for h in range(n_harmonics)]

        # Collect features
        feats = {
            "filename": os.path.basename(path),
            "duration_s": duration,
            "mean_f0": mean_f0,
            "f0_95th_percentile": p95,
            "f0_5th_percentile": p05,
            "f0_peak_location": peak_loc,
            "f0_peak_value": max_f0,
            "f0_quadratic_term": coeffs[2],
            "f0_cubic_term": coeffs[0],
            "pitch_wobble_frequency": wobble_f,
            "pitch_wobble_periodicity_1": wobble_p1,
            "pitch_wobble_periodicity_2": wobble_p2,
            "pitch_instability": instability,
            "amplitude_moment_2": m2,
            "amplitude_moment_3": m3,
            "amplitude_moment_4": m4,
            "amplitude_wobble_frequency": amp_wf,
            "amplitude_wobble_magnitude": amp_wm,
        }
        for i, (a, tr) in enumerate(zip(avg_rel, trends), 1):
            feats[f"avg_rel_h{i}"] = a
            feats[f"trend_h{i}"] = tr

        return feats

    except Exception as e:
        with open("swipe_feature_errors.log", "a") as log:
            log.write(f"{path}\t{e}\n")
        return None


def compute_top3_pcs(df, cols, prefix):
    if len(df) >= 3 and set(cols).issubset(df.columns):
        X = df[cols].values
        Xs = StandardScaler().fit_transform(X)
        pcs = PCA(n_components=3).fit_transform(Xs)
        for j in range(3):
            df[f"{prefix}{j+1}"] = pcs[:, j]
        df.drop(columns=cols, inplace=True)


# === Main ===
if __name__ == "__main__":
    wavs = sorted(glob(os.path.join(input_dir, "*.wav")))
    print(f"Extracting features from {len(wavs)} calls...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        results = list(tqdm(exe.map(extract_features, wavs),
                            total=len(wavs),
                            desc="Processing clips"))

    records = [r for r in results if r is not None]
    print(f"✅ Extracted {len(records)}/{len(wavs)} calls")

    df = pd.DataFrame(records)
    avg_cols = [c for c in df.columns if c.startswith("avg_rel_h")]
    trend_cols = [c for c in df.columns if c.startswith("trend_h")]

    if avg_cols:
        compute_top3_pcs(df, avg_cols, prefix="harmonic_pc")
    if trend_cols:
        compute_top3_pcs(df, trend_cols, prefix="harmonic_trend_pc")

    df.to_csv(output_csv, index=False)
    print(f"✅ Done: saved {df.shape[0]}×{df.shape[1]} → {output_csv}")