import os
import csv
from multiprocessing import Pool, Manager, Lock
from pydub import AudioSegment, silence
import matplotlib.pyplot as plt
import numpy as np

# === USER CONFIGURATION ===
INPUT_DIR = "sample/raw_denoised_extra"   # Input folder with audio files
SPLIT_DIR = "sample_sequences/raw_denoised_extra_cut"              # Output folder for chunks
MIN_SILENCE_LEN_MS = 10_000                          # Minimum silence length (10 sec)
SILENCE_THRESH_DB = -60                              # Silence threshold (–60 dBFS)
KEEP_SILENCE_MS = 500                                # Silence padding (0.5 sec)
START_INDEX = 1                                       # Starting chunk index
FILENAME = "c"                                        # Prefix for chunk files
CSV_PATH = "sequences_log_sample.csv"             # CSV log file path
NUM_WORKERS = 32                                      # Number of multiprocessing workers
ENABLE_VISUALIZATION = True                           # Toggle waveform plots with splits
# ===========================


def load_processed_files(csv_path):
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row['source_file'])
    return processed


def get_next_index(out_dir):
    if not os.path.exists(out_dir):
        return START_INDEX
    existing = [
        int(f.split('_')[-1].split('.')[0])
        for f in os.listdir(out_dir)
        if f.endswith(".wav") and '_' in f
    ]
    return max(existing) + 1 if existing else START_INDEX


def plot_splits(audio, splits_ms, input_path, out_dir):
    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
        samples = samples.mean(axis=1)  # Convert stereo to mono

    times = np.linspace(0, len(samples) / audio.frame_rate, num=len(samples))

    plt.figure(figsize=(12, 3))
    plt.plot(times, samples, color='gray')
    for split_ms in splits_ms:
        plt.axvline(split_ms / 1000, color='red', linestyle='--', linewidth=1)

    plt.title(f"Splits for {os.path.basename(input_path)}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    plot_path = os.path.join(out_dir, f"{base_name}_splits.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Saved split visualization to: {plot_path}")


def split_on_long_silence(args):
    (input_path, out_dir, filename_prefix,
     min_silence_len_ms, silence_thresh_db, keep_silence_ms,
     lock, index_counter, csv_path, enable_viz) = args

    os.makedirs(out_dir, exist_ok=True)
    audio = AudioSegment.from_file(input_path)

    silence_ranges = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db
    )
    splits_ms = [start for start, end in silence_ranges]

    if enable_viz:
        plot_splits(audio, splits_ms, input_path, out_dir)

    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=keep_silence_ms
    )

    saved = []

    if not chunks:
        with lock:
            idx = index_counter.value
            index_counter.value += 1
        chunk_filename = f"{filename_prefix}_{idx:05d}.wav"
        chunk_path = os.path.join(out_dir, chunk_filename)
        audio.export(chunk_path, format="wav")
        print(f"✔ No silence found, saved full file as: {chunk_filename} (from {os.path.basename(input_path)})")

        with lock:
            with open(csv_path, "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([chunk_filename, os.path.basename(input_path)])
        saved.append((chunk_filename, os.path.basename(input_path)))
        return saved

    for chunk in chunks:
        with lock:
            idx = index_counter.value
            index_counter.value += 1
        chunk_filename = f"{filename_prefix}_{idx:05d}.wav"
        chunk_path = os.path.join(out_dir, chunk_filename)
        chunk.export(chunk_path, format="wav")
        print(f"✔ Exported chunk: {chunk_filename} (from {os.path.basename(input_path)})")

        with lock:
            with open(csv_path, "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([chunk_filename, os.path.basename(input_path)])

        saved.append((chunk_filename, os.path.basename(input_path)))

    return saved


def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    processed_files = load_processed_files(CSV_PATH)

    input_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')) and f not in processed_files
    ]
    input_files.sort(key=lambda f: os.path.getctime(os.path.join(INPUT_DIR, f)))

    with Manager() as manager:
        lock = manager.Lock()
        index_counter = manager.Value('i', get_next_index(SPLIT_DIR))

        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["chunk_filename", "source_file"])

        args_list = [
            (
                os.path.join(INPUT_DIR, fname),
                SPLIT_DIR,
                FILENAME,
                MIN_SILENCE_LEN_MS,
                SILENCE_THRESH_DB,
                KEEP_SILENCE_MS,
                lock,
                index_counter,
                CSV_PATH,
                ENABLE_VISUALIZATION
            )
            for fname in input_files
        ]

        with Pool(processes=NUM_WORKERS) as pool:
            pool.map(split_on_long_silence, args_list)

    print(f"\nDone! Processed {len(input_files)} files.")


if __name__ == "__main__":
    main()