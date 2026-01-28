import os
import torchaudio

# 🔧 List your directories here
DIRECTORIES = [
    "data/corax/sequences",
    "data/cornix/sequences",
    "data/corone/sequences",
    "data/ossifragus/sequences",
    "american_crow_sequences"
]

def get_audio_duration(filepath):
    try:
        info = torchaudio.info(filepath)
        return info.num_frames / info.sample_rate
    except Exception as e:
        print(f"Warning: Failed to read {filepath} ({e})")
        return 0.0

results = []

for directory in DIRECTORIES:
    wav_files = [f for f in os.listdir(directory) if f.lower().endswith('.wav')]
    total_duration = sum(get_audio_duration(os.path.join(directory, f)) for f in wav_files)
    results.append({
        "directory": directory,
        "num_files": len(wav_files),
        "total_duration_sec": total_duration
    })

# Save to TXT
OUTPUT_TXT = "directory_audio_summary.txt"
with open(OUTPUT_TXT, 'w') as f:
    f.write("=== Directory Audio Summary ===\n\n")
    for res in results:
        f.write(f"{res['directory']}\n")
        f.write(f"  Number of files: {res['num_files']}\n")
        f.write(f"  Total duration: {res['total_duration_sec']:.2f} seconds\n\n")

print(f"✅ Saved summary to {OUTPUT_TXT}")