import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
from panns_inference import SoundEventDetection, labels as panns_labels
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt  # Added here for visualization

# === CONFIGURATION ===
INPUT_DIR        = "sample/raw_denoised_extra"
OUTPUT_DIR       = "sample_sequences/raw_denoised_extra_cut_panns"
ANNOT_CSV        = "panns_annotations_sample.csv"
MODEL_TYPE       = "panns"      # or "yamnet"
CROW_LABELS      = ["Crow", "Raven", "Caw", "Corvid", "Rattle", "Chatter", "Coo", "Croak"]
CONF_THRESHOLD   = 0.05
MIN_CALL_FRAMES  = 2
MAX_GAP_FRAMES   = 50
PANN_SR          = 32000
HOP_SIZE         = 320
NUM_PADDING_FRAMES = 3
SAVE_ORIGINAL_NAME = True
PANNS_MODEL_PATH = "Cnn14_mAP=0.431.pth"

VISUALIZE_SPLITTING = True  # <-- Set to True to save visualizations

# ======================

def load_yamnet_labels():
    csv_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    return list(pd.read_csv(csv_path)['display_name'])

def get_crow_indices(label_list):
    return [i for i, lbl in enumerate(label_list) if any(c in lbl for c in CROW_LABELS)]

def create_model_with_fallback(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        print(f"Trying to load model on {device}")
        model = SoundEventDetection(device=device, checkpoint_path=checkpoint_path)
        return model, device
    except RuntimeError as e:
        print(f"Failed to load on {device} due to {e}, retrying on CPU...")
        model = SoundEventDetection(device="cpu", checkpoint_path=checkpoint_path)
        return model, "cpu"

def run_inference_with_fallback(model, device, audio_np, checkpoint_path):
    try:
        return model.inference(audio_np)[0], device, model
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            print("⚠️ CUDA OOM during inference, switching to CPU...")
            model = SoundEventDetection(device="cpu", checkpoint_path=checkpoint_path)
            return model.inference(audio_np)[0], "cpu", model
        else:
            raise e

def run_sed(wav_path, out_dir, file_id, model, device, checkpoint_path, model_type, crow_indices, annotations, save_original_name=False):
    try:
        waveform, sr = torchaudio.load(wav_path)
        if sr != PANN_SR:
            waveform = torchaudio.functional.resample(waveform, sr, PANN_SR)
            sr = PANN_SR
        audio_np = waveform.numpy()

        if model_type == "panns":
            framewise, device, model = run_inference_with_fallback(model, device, audio_np, checkpoint_path)
            crow_scores = framewise[:, crow_indices].max(axis=1)
            hop = HOP_SIZE
        else:
            audio_tf = tf.convert_to_tensor(np.mean(audio_np, axis=0), dtype=tf.float32)
            scores, _, _ = model(audio_tf)
            crow_scores = scores.numpy()[:, crow_indices].max(axis=1)
            hop = int(sr * 0.48)

        active = crow_scores > CONF_THRESHOLD

        # Build segments with gap tolerance
        segments = []
        in_seg = False
        seg_start = None
        silence = 0

        for i, is_active in enumerate(active):
            if is_active:
                if not in_seg:
                    seg_start = i
                    in_seg = True
                silence = 0
            else:
                if in_seg:
                    silence += 1
                    if silence > MAX_GAP_FRAMES:
                        seg_end = i - silence
                        if (seg_end - seg_start + 1) >= MIN_CALL_FRAMES:
                            segments.append((seg_start, seg_end))
                        in_seg = False
                        silence = 0

        if in_seg:
            seg_end = len(active) - 1
            if (seg_end - seg_start + 1) >= MIN_CALL_FRAMES:
                segments.append((seg_start, seg_end))

        if not segments:
            print(f"• No crow call in {os.path.basename(wav_path)}")
            return model, device

        os.makedirs(out_dir, exist_ok=True)
        original_basename = os.path.splitext(os.path.basename(wav_path))[0]

        # === VISUALIZATION ===
        if VISUALIZE_SPLITTING:
            times = np.arange(len(crow_scores)) * (hop / sr)
            plt.figure(figsize=(10, 4))
            plt.plot(times, crow_scores, label="Crow Score")
            plt.axhline(CONF_THRESHOLD, color='red', linestyle='--', label="Threshold")
            for (s_frame, e_frame) in segments:
                s_time = s_frame * (hop / sr)
                e_time = e_frame * (hop / sr)
                plt.axvspan(s_time, e_time, color='green', alpha=0.3)
            plt.title(f"Crow Call Detection: {os.path.basename(wav_path)}")
            plt.xlabel("Time (s)")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            vis_name = original_basename + "_split.png"
            plt.savefig(os.path.join(out_dir, vis_name), dpi=150)
            plt.close()
            print(f"📊 Saved visualization {vis_name}")

        for idx, (s_frame, e_frame) in enumerate(segments, 1):
            s_sample = max(0, (s_frame - NUM_PADDING_FRAMES) * hop)
            e_sample = (e_frame + NUM_PADDING_FRAMES) * hop
            clip = waveform[:, s_sample:e_sample]

            if save_original_name:
                clip_name = f"{original_basename}_{idx}.wav"
            else:
                clip_name = f"{file_id}_{idx}.wav"

            clip_path = os.path.join(out_dir, clip_name)
            if os.path.exists(clip_path):
                print(f"⏩ Clip {clip_name} already exists, skipping saving")
            else:
                torchaudio.save(clip_path, clip, sr)
                print(f"  ✔ {clip_name} | {s_sample/sr:.2f}-{e_sample/sr:.2f}s")

            annotations.append({
                "filename": clip_name,
                "onset":  round(s_sample / sr, 2),
                "offset": round(e_sample / sr, 2),
                "event_label": "crow"
            })

        return model, device

    except Exception as e:
        print(f"❌ Failed on {wav_path}: {e}")
        return model, device

def main():
    if MODEL_TYPE == "panns":
        model, device = create_model_with_fallback(PANNS_MODEL_PATH)
        labels = panns_labels
    else:
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        device = "cpu"
        labels = load_yamnet_labels()

    crow_idxs = get_crow_indices(labels)

    existing_clips = set()
    if os.path.exists(OUTPUT_DIR):
        existing_clips = set(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav"))

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".wav"))
    annotations = []

    for fn in files:
        base = os.path.splitext(fn)[0]
        file_id = base
        clip_prefix = f"{base}_"
        already_processed = any(c.startswith(clip_prefix) for c in existing_clips)
        if already_processed:
            print(f"⏩ Skipping {fn} (clips already exist)")
            continue

        model, device = run_sed(
            wav_path=os.path.join(INPUT_DIR, fn),
            out_dir=OUTPUT_DIR,
            file_id=file_id,
            model=model,
            device=device,
            checkpoint_path=PANNS_MODEL_PATH,
            model_type=MODEL_TYPE,
            crow_indices=crow_idxs,
            annotations=annotations,
            save_original_name=SAVE_ORIGINAL_NAME
        )

    pd.DataFrame(annotations).to_csv(ANNOT_CSV, index=False)
    print(f"\n✅ Saved annotations to {ANNOT_CSV}")

if __name__ == "__main__":
    main()