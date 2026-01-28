import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, find_peaks

# ========== CONFIGURATION ==========
INPUT_DIR = "sample"
OUTPUT_DIR = "envelope_plots"
CSV_LOG = "repetition_counts.csv"
AUDIO_EXTS = [".wav", ".mp3", ".flac"]
GAUSSIAN_SIGMA = 2
MOVING_AVG_WINDOW = 10
MEDIAN_KERNEL = 5
PEAK_DISTANCE = 5      # Min frames between peaks
RELATIVE_PEAK_HEIGHT = 0.3  # Relative to max envelope
# ===================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size) / window_size, mode='same')

def normalize_envelope(envelope):
    max_val = np.max(envelope)
    return envelope / max_val if max_val > 0 else envelope

def process_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    envelope = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(envelope)), sr=sr)

    # Normalize envelope for relative peak detection
    env_g = normalize_envelope(gaussian_filter1d(envelope, sigma=GAUSSIAN_SIGMA))
    env_m = normalize_envelope(moving_average(envelope, MOVING_AVG_WINDOW))
    env_med = normalize_envelope(medfilt(envelope, kernel_size=MEDIAN_KERNEL))

    return times, envelope, env_g, env_m, env_med

def detect_peaks(envelope_norm):
    peaks, _ = find_peaks(envelope_norm, height=RELATIVE_PEAK_HEIGHT, distance=PEAK_DISTANCE)
    return peaks

def plot_and_save(file_name, times, raw, gaussian, moving, median,
                  peaks_g, peaks_m, peaks_med):
    plt.figure(figsize=(12, 4))
    plt.plot(times, normalize_envelope(raw), label="Raw RMS", alpha=0.3, color="gray")
    plt.plot(times, gaussian, label="Gaussian", linewidth=2, color="blue")
    plt.plot(times[peaks_g], gaussian[peaks_g], "bo", label="Peaks (Gaussian)")

    plt.plot(times, moving, label="Moving Avg", linewidth=2, color="green")
    plt.plot(times[peaks_m], moving[peaks_m], "go", label="Peaks (Moving Avg)")

    plt.plot(times, median, label="Median", linewidth=2, color="red")
    plt.plot(times[peaks_med], median[peaks_med], "ro", label="Peaks (Median)")

    plt.title(f"Envelope + Repetition Detection: {file_name}")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(file_name))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_envelope_peaks.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

# Get sorted list of files
file_list = sorted([
    f for f in os.listdir(INPUT_DIR)
    if any(f.lower().endswith(ext) for ext in AUDIO_EXTS)
])

# CSV logging setup
with open(CSV_LOG, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File", "Gaussian_Peaks", "MovingAvg_Peaks", "Median_Peaks"])

    for fname in file_list:
        fpath = os.path.join(INPUT_DIR, fname)
        try:
            times, raw, g, m, med = process_audio_file(fpath)
            peaks_g = detect_peaks(g)
            peaks_m = detect_peaks(m)
            peaks_med = detect_peaks(med)

            plot_and_save(fname, times, raw, g, m, med,
                          peaks_g, peaks_m, peaks_med)

            writer.writerow([fname, len(peaks_g), len(peaks_m), len(peaks_med)])
            print(f"✅ Processed: {fname} | Peaks (G/M/Md): {len(peaks_g)}, {len(peaks_m)}, {len(peaks_med)}")

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")