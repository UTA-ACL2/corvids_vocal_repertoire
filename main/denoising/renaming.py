import os

# ---------------------
# User Configuration
# ---------------------
DIRECTORY = "../separation/data/cornix/sequences"   # ← Change this to your folder path
BASE_NAME = "cornix"               # ← Set your desired base name (e.g., "crowcall")
PADDING = 5                          # ← Controls zero padding: 0001, 0002, etc.
# ---------------------

def rename_wav_files(directory, base_name, padding=4):
    files = [f for f in os.listdir(directory) if f.lower().endswith('.wav')]
    files.sort()  # Sort to keep order consistent

    for i, file in enumerate(files, start=1):
        ext = os.path.splitext(file)[1]
        new_name = f"{base_name}_{str(i).zfill(padding)}{ext}"
        src = os.path.join(directory, file)
        dst = os.path.join(directory, new_name)
        os.rename(src, dst)
        print(f"Renamed: {file} → {new_name}")

rename_wav_files(DIRECTORY, BASE_NAME, PADDING)