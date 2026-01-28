import os
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment

VALID_AUDIO_EXTENSIONS = (".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".aiff", ".alac", ".amr")

def convert_file(args):
    src_path, dest_path = args
    try:
        print(f"Converting {src_path} -> {dest_path}")
        audio = AudioSegment.from_file(src_path)
        audio.export(dest_path, format="wav")
        print(f"✅ Converted {src_path}")
    except Exception as e:
        print(f"❌ Error {src_path}: {e}")

def convert_non_wav_parallel(src_dir, dest_dir, max_workers=4):
    os.makedirs(dest_dir, exist_ok=True)
    tasks = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            print(f"Checking file: {file} with extension: {ext}")
            if ext in VALID_AUDIO_EXTENSIONS:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, src_dir)
                filename = os.path.splitext(file)[0]
                output_subdir = os.path.join(dest_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                dest_path = os.path.join(output_subdir, filename + ".wav")

                exists = os.path.exists(dest_path)
                print(f"Dest file: {dest_path} exists? {exists}")

                if not exists:
                    tasks.append((src_path, dest_path))
                else:
                    print(f"Skipping {dest_path} (already exists)")

    print(f"Total files to convert: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(convert_file, tasks)

# Usage example (adjust paths accordingly)
convert_non_wav_parallel("data/corax", "data/corax_conv", max_workers=32)