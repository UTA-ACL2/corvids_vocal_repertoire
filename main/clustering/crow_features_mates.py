import os
import pandas as pd
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from scipy.signal import hilbert
from scipy.stats import moment
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Configuration ===
input_dir   = "../separation/american_crow_calls_half_second_silence"
output_csv  = "crow_features_mates_0.5.csv"
min_pitch   = 75
max_pitch   = 600
n_harmonics = 12
frame_size  = 2048
hop_size    = 512
epsilon     = 1e-12

def safe_log(x):
    return np.log(np.abs(x) + epsilon)

def safe_signed_pow(x, p=0.5):
    return np.sign(x) * (np.abs(x) + epsilon) ** p

def extract_harmonic_energies(y, sr, f0):
    if len(y) < frame_size:
        return np.zeros((1, n_harmonics))
    frames = []
    for i in range(0, len(y) - frame_size, hop_size):
        frm = y[i:i+frame_size]
        if len(frm) < frame_size:
            frm = np.pad(frm, (0, frame_size - len(frm)))
        frames.append(frm * np.hanning(frame_size))
    if not frames:
        return np.zeros((1, n_harmonics))
    H = []
    for frm in frames:
        spec  = np.abs(np.fft.rfft(frm))
        freqs = np.fft.rfftfreq(len(frm), d=1/sr)
        row   = []
        for h in range(1, n_harmonics+1):
            tf = f0 * h
            if tf > freqs[-1]:
                row.append(0.0)
            else:
                row.append(spec[np.argmin(np.abs(freqs - tf))])
        H.append(row)
    return np.array(H)

def extract_call_features(path):
    try:
        y, sr = sf.read(path)
        if y.ndim > 1: y = y.mean(axis=1)
        duration = len(y) / sr

        # === Pitch ===
        snd       = parselmouth.Sound(path)
        pitch_obj = call(snd, "To Pitch", 0.0, min_pitch, max_pitch)
        n_fr      = pitch_obj.get_number_of_frames()
        pitches   = np.array([pitch_obj.get_value_in_frame(i+1) or 0 for i in range(n_fr)])
        pitches_pos = pitches[pitches > 0]

        # Instead of skipping, assign safe defaults if too few pitch frames:
        if len(pitches_pos) < 5:
            # Provide dummy fallback values
            mean_f0 = 0.0
            p95 = 0.0
            p05 = 0.0
            max_f0 = 0.0
            peak_loc = 0.0
            coeffs = [0.0, 0.0, 0.0, 0.0]
            residual = np.zeros_like(pitches)
            instability = 0.0
            wobble_f = 0
            wobble_p1 = 0.0
            wobble_p2 = 0.0
        else:
            mean_f0    = pitches_pos.mean()
            p95        = np.percentile(pitches_pos, 95)
            p05        = np.percentile(pitches_pos, 5)
            max_f0     = pitches_pos.max()
            peak_loc   = np.argmax(pitches_pos) / len(pitches_pos)
            tnorm      = np.linspace(0, 1, len(pitches_pos))
            coeffs     = np.polyfit(tnorm, pitches_pos, 3)
            residual   = pitches_pos - np.polyval(coeffs, tnorm)
            instability = np.sqrt(np.mean(residual**2))
            fft_r      = np.abs(np.fft.rfft(residual))
            wobble_f   = np.argmax(fft_r[1:]) + 1 if len(fft_r)>1 else 0
            wobble_p1  = 1/(np.std(residual)+epsilon)
            wobble_p2  = 1/(np.var(residual)+epsilon)

        # === Amplitude ===
        analytic = hilbert(y)
        amp_env  = np.abs(analytic)
        m2       = moment(amp_env, moment=2)
        m3       = moment(amp_env, moment=3)
        m4       = moment(amp_env, moment=4)
        fft_a    = np.abs(np.fft.rfft(amp_env))
        amp_wf   = np.argmax(fft_a[1:]) + 1 if len(fft_a)>1 else 0
        amp_wm   = fft_a.max() if len(fft_a)>0 else 0

        # === Harmonics ===
        H_raw = extract_harmonic_energies(y, sr, mean_f0)        # (T×12)
        H_rel = H_raw / (H_raw.sum(axis=1, keepdims=True) + epsilon)
        avg_rel = H_rel.mean(axis=0)                             # (12,)
        trends  = np.array([
            np.polyfit(np.arange(H_rel.shape[0]), H_rel[:,h], 1)[0]
            if H_rel.shape[0] > 1 else 0.0
            for h in range(n_harmonics)
        ])

        return {
            "filename": os.path.basename(path),
            "mean_f0": mean_f0,
            "f0_95th_percentile": p95,
            "f0_5th_percentile": p05,
            "f0_peak_location": safe_log(peak_loc),
            "f0_peak_value": max_f0,
            "f0_quadratic_term": coeffs[1],
            "f0_cubic_term": safe_signed_pow(coeffs[0], 0.5),
            "pitch_wobble_frequency": wobble_f,
            "pitch_wobble_periodicity_1": safe_log(wobble_p1),
            "pitch_wobble_periodicity_2": wobble_p2,
            "pitch_instability": instability,
            "call_length": safe_log(duration),
            "amplitude_moment_2": safe_log(m2),
            "amplitude_moment_3": m3,
            "amplitude_moment_4": safe_log(m4),
            "amplitude_wobble_frequency": amp_wf,
            "amplitude_wobble_magnitude": amp_wm,
            **{f"avg_rel_h{i+1}": avg_rel[i] for i in range(n_harmonics)},
            **{f"trend_h{i+1}": trends[i] for i in range(n_harmonics)},
        }

    except Exception as e:
        with open("feature_errors.log", "a") as f:
            f.write(f"Failed: {path} | {str(e)}\n")
        return None

def compute_top3_pcs(df, cols, pc_name):
    X = df[cols].values
    if len(df) >= 3:
        Xs  = StandardScaler().fit_transform(X)
        pcs = PCA(n_components=3).fit_transform(Xs)
    else:
        pcs = np.stack([X[:,i] for i in range(3)], axis=1)
    for j in range(3):
        df[f"{pc_name}{j+1}"] = pcs[:, j]

# === Main execution ===
if __name__ == "__main__":
    wavs = sorted(glob(os.path.join(input_dir, "*.wav")))
    print(f"Extracting features from {len(wavs)} calls...")

    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(extract_call_features, wavs),
                            total=len(wavs),
                            desc="Processing clips"))

    records = [r for r in results if r is not None]
    print(f"✅ Extracted features for {len(records)} / {len(wavs)} calls")

    df = pd.DataFrame(records)

    avg_cols   = [f"avg_rel_h{i+1}" for i in range(n_harmonics)]
    trend_cols = [f"trend_h{i+1}" for i in range(n_harmonics)]

    compute_top3_pcs(df, avg_cols, "harmonic_pc")
    compute_top3_pcs(df, trend_cols, "harmonic_trend_pc")

    df.drop(columns=avg_cols + trend_cols, inplace=True)
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Done: saved {len(df)} calls × {len(df.columns)} columns to {output_csv}")