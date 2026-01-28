import os
import time
import csv
import soundfile as sf
import noisereduce as nr
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager


# 🔧 Config
INPUT_FOLDER = "data/corax_conv"
OUTPUT_FOLDER = "noisereduced/corax_conv"
LOG_CSV = "processing_log_corax_conv.csv"
USE_CHUNKING = False


# === Utility Functions ===
def load_audio_fast(filepath):
    audio, sr = sf.read(filepath)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr


def process_file(input_path, output_path_base):
    input_path = Path(input_path)
    output_path_base = Path(output_path_base)
    cleaned_filename = output_path_base.with_suffix(".wav")

    start_time = time.time()

    try:
        cleaned_filename.parent.mkdir(parents=True, exist_ok=True)

        if cleaned_filename.exists():
            print(f"⏭️ Skipped: {input_path}")
            return [str(input_path), "skipped", 0.0, "already exists"]

        audio, sr = load_audio_fast(input_path)
        cleaned_audio = nr.reduce_noise(y=audio, sr=sr)
        sf.write(cleaned_filename, cleaned_audio, sr)

        duration = time.time() - start_time
        print(f"✅ Processed: {input_path} in {duration:.2f}s")
        return [str(input_path), "success", duration, ""]
    except Exception as e:
        print(f"❌ Error: {input_path} – {e}")
        return [str(input_path), "error", 0.0, str(e)]


def prepare_file_pairs(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    file_pairs = []

    for input_file in input_dir.rglob("*.wav"):
        relative_path = input_file.relative_to(input_dir)
        output_base = output_dir / relative_path
        file_pairs.append((str(input_file), str(output_base)))

    return file_pairs


def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# === Main Parallel Function ===
def parallel_process_files(input_dir, output_dir, log_csv, use_chunking=True):
    file_pairs = prepare_file_pairs(input_dir, output_dir)
    total_files = len(file_pairs)

    if total_files == 0:
        print("No .wav files found.")
        return

    cpu_cores = cpu_count()

    with open(log_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "status", "duration_sec", "error"])

        if use_chunking:
            chunks = chunkify(file_pairs, cpu_cores)
            with Pool(processes=cpu_cores) as pool:
                for chunk in tqdm(chunks, total=cpu_cores, desc="🔊 Cleaning audio"):
                    results = pool.starmap(process_file, chunk)
                    writer.writerows(results)
        else:
            with Pool(processes=cpu_cores) as pool:
                for result in tqdm(pool.starmap(process_file, file_pairs), total=total_files, desc="🔊 Cleaning audio"):
                    writer.writerow(result)

    print(f"✅ All done. Log saved to: {log_csv}")


# === Entry Point ===
if __name__ == "__main__":
    start = time.time()
    parallel_process_files(INPUT_FOLDER, OUTPUT_FOLDER, LOG_CSV, use_chunking=USE_CHUNKING)
    print(f"⏱️ Done in {time.time() - start:.2f} seconds.")