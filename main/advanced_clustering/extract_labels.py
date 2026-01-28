import os
import pandas as pd
import shutil

# === CONFIGURATION ===
csv_path = "crowtools_classified.csv"                 # Original CSV file
audio_dir = "../separation/american_crow_calls"       # Directory with original .wav files
output_dir = "mob_rattle_audio"                        # Directory to save filtered files
output_csv = "mob_rattle_subset.csv"                  # Filtered CSV output

# === Create output directory if it doesn't exist ===
os.makedirs(output_dir, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(csv_path)

# Ensure rattle/mob are booleans (in case they are strings like "True"/"False")
df['rattle'] = df['rattle'].astype(bool)
df['mob'] = df['mob'].astype(bool)

# === Filter rows where rattle or mob is True ===
filtered_df = df[(df['rattle']) | (df['mob'])].copy()

# === Copy corresponding audio files ===
for filename in filtered_df['filename']:
    src_path = os.path.join(audio_dir, filename)
    dst_path = os.path.join(output_dir, filename)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"⚠️ Missing file: {src_path}")

# === Save new CSV with only mob, rattle, and filename ===
filtered_df[['filename', 'rattle', 'mob']].to_csv(output_csv, index=False)

print(f"✅ Done. {len(filtered_df)} files copied to '{output_dir}' and CSV saved to '{output_csv}'")