import os
import librosa
import numpy as np
import csv
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========== CONFIGURATION ==========
INPUT_DIR = "../separation/data/corax/calls"
CSV_LOG = "corax_peaks.csv"
AUDIO_EXTS = [".wav", ".mp3", ".flac"]
GAUSSIAN_SIGMA = 2
PEAK_DISTANCE = 5
RELATIVE_PEAK_HEIGHT = 0.3
NUM_WORKERS = os.cpu_count()  # Or os.cpu_count()
# ===================================

def normalize_envelope(envelope):
    max_val = np.max(envelope)
    return envelope / max_val if max_val > 0 else envelope

def process_file(file_tuple):
    fname, fpath = file_tuple
    try:
        y, sr = librosa.load(fpath, sr=None)
        envelope = librosa.feature.rms(y=y)[0]
        smoothed = gaussian_filter1d(envelope, sigma=GAUSSIAN_SIGMA)
        env_norm = normalize_envelope(smoothed)
        peaks, _ = find_peaks(env_norm, height=RELATIVE_PEAK_HEIGHT, distance=PEAK_DISTANCE)
        return (fname, len(peaks))
    except Exception as e:
        return (fname, f"ERROR: {e}")

# Sorted list of files
file_list = sorted([
    f for f in os.listdir(INPUT_DIR)
    if any(f.lower().endswith(ext) for ext in AUDIO_EXTS)
])
full_paths = [(f, os.path.join(INPUT_DIR, f)) for f in file_list]

# Process in parallel
results = {}
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_file, fp): fp[0] for fp in full_paths}
    for future in as_completed(futures):
        fname, peak_count = future.result()
        results[fname] = peak_count
        print(f"✅ {fname}: {peak_count}")

# Write to CSV in sorted order
with open(CSV_LOG, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File", "Gaussian_Peak_Count"])
    for fname in file_list:
        writer.writerow([fname, results[fname]])