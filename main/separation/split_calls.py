import os
import torch
import torchaudio
import numpy as np
from panns_inference import SoundEventDetection, labels

# === USER CONFIGURATION ===
INPUT_DIR = "sample_sequences/denoised_split"
OUTPUT_DIR = "sample_sequences/denoised_segments2"
MODEL_PATH = "Cnn14_mAP=0.431.pth"
FILENAME = "brachyrynchos"

CROW_LABELS = ["Crow", "Bird", "Caw", "Raven", "Corvid", "Rattle", "Coo", "Chirp", "Croak", "Groan"]
CONF_THRESHOLD = 0.05
MIN_CALL_FRAMES = 3

PANN_SR = 32000
HOP_SIZE = 320
FRAME_DUR = HOP_SIZE / PANN_SR
NUM_PADDING_FRAMES = 3

def run_panns_detection(wav_path, out_dir, file_index, model, crow_indices):
    try:
        waveform, sr = torchaudio.load(wav_path)
        if sr != PANN_SR:
            waveform = torchaudio.functional.resample(waveform, sr, PANN_SR)

        audio_np = waveform.numpy()
        framewise_output = model.inference(audio_np)[0]
        crow_conf = framewise_output[:, crow_indices]
        crow_conf_max = crow_conf.max(axis=1)

        active = crow_conf_max > CONF_THRESHOLD
        segments = []
        start = None
        for i, val in enumerate(active):
            if val and start is None:
                start = i
            elif not val and start is not None:
                end = i - 1
                if end - start + 1 >= MIN_CALL_FRAMES:
                    segments.append((start, end))
                start = None
        if start is not None:
            end = len(active) - 1
            if end - start + 1 >= MIN_CALL_FRAMES:
                segments.append((start, end))

        if not segments:
            print(f"• No crow calls detected in {os.path.basename(wav_path)}")
            return

        for idx, (sframe, eframe) in enumerate(segments, start=1):
            s_sample = int(sframe * HOP_SIZE)
            e_sample = int((eframe + NUM_PADDING_FRAMES) * HOP_SIZE)
            clip = waveform[:, s_sample:e_sample]
            out_wav = os.path.join(out_dir, f"{FILENAME}_{file_index:05d}_{idx}.wav")
            torchaudio.save(out_wav, clip, PANN_SR)
            print(f"• Saved: {os.path.basename(out_wav)}")
    except Exception as e:
        print(f"• Error processing {wav_path}: {e}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    crow_indices = [labels.index(lbl) for lbl in CROW_LABELS if lbl in labels]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = SoundEventDetection(checkpoint_path=MODEL_PATH, device=device)

    files_to_process = []
    for i, fname in enumerate(sorted(os.listdir(INPUT_DIR))):
        if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            full_path = os.path.join(INPUT_DIR, fname)
            files_to_process.append((full_path, OUTPUT_DIR, i + 1))

    if not files_to_process:
        print("No audio files found in input directory.")
        return

    print(f"Processing {len(files_to_process)} files...\n")

    for wav_path, out_dir, idx in files_to_process:
        run_panns_detection(wav_path, out_dir, idx, model, crow_indices)

if __name__ == "__main__":
    main()