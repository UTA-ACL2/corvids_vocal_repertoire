import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
INPUT_DIR           = "sample_sequences/validation_samples"
OUTPUT_DIR          = "sample_sequences/validation_calls_finetuned"
ANNOT_CSV           = "panns_annotations_finetuned.csv"
CHECKPOINT_PATH     = "fine_tuned_cnn14_framewise.pth"
NUM_CLASSES_FINETUNED = 4      # your finetuned head size
CONF_THRESHOLD      = 0.10     # crow confidence threshold
MIN_CALL_FRAMES     = 2        # allow even single-frame calls
MAX_GAP_FRAMES      = 10       # small gap tolerance for finer splits
PANN_SR             = 32000
HOP_SIZE            = 320      # 10 ms per frame
NUM_PADDING_FRAMES  = 3        # minimal padding
SAVE_ORIGINAL_NAME  = True
PLOT_DIR            = "framewise_plots"
# ======================

from audioset_tagging_cnn.pytorch.models import Cnn14_DecisionLevelMax

def load_finetuned_model(path, device):
    model = Cnn14_DecisionLevelMax(
        sample_rate=PANN_SR, window_size=1024, hop_size=HOP_SIZE,
        mel_bins=64, fmin=50, fmax=14000, classes_num=527
    )
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model", ckpt)
    # Remove old fc layer weights
    sd = {k: v for k, v in sd.items() if not k.startswith("fc_audioset.")}
    model.load_state_dict(sd, strict=False)
    # Replace final head with fine-tuned head
    model.fc_audioset = torch.nn.Linear(2048, NUM_CLASSES_FINETUNED)
    # Load finetuned head weights if present
    if "fc_audioset.weight" in sd:
        model.fc_audioset.weight.data.copy_(sd["fc_audioset.weight"])
        model.fc_audioset.bias.data.copy_(sd["fc_audioset.bias"])
    model.to(device).eval()

    def inference(self, wav_np):
        if wav_np.ndim == 2:
            wav_np = wav_np[0]  # mono channel
        x = torch.from_numpy(wav_np).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            out = self.forward(x)["framewise_output"]  # (1, T, C)
            print(f"Logits stats - min: {out.min().item():.4f}, max: {out.max().item():.4f}, mean: {out.mean().item():.4f}")
            probs = torch.sigmoid(out)
            print(f"Probs stats - min: {probs.min().item():.4f}, max: {probs.max().item():.4f}, mean: {probs.mean().item():.4f}")
        return probs.cpu().numpy()[0]  # (T, C)
    model.inference = inference.__get__(model)
    return model

def plot_framewise_confidence(confidences, filename):
    plt.figure(figsize=(10, 3))
    plt.plot(confidences, label='Crow confidence')
    plt.xlabel('Frame index')
    plt.ylabel('Confidence')
    plt.title(f'Framewise Crow Confidence: {filename}')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, filename + ".png"))
    plt.close()

def run_sed(wav_path, model, crow_idx, annotations):
    wav, sr = torchaudio.load(wav_path)
    if sr != PANN_SR:
        wav = torchaudio.functional.resample(wav, sr, PANN_SR)
    wav_np = wav.numpy()
    preds = model.inference(wav_np)        # (T, C)
    crow_scores = preds[:, crow_idx]

    # Plot framewise confidence for inspection
    base_filename = os.path.splitext(os.path.basename(wav_path))[0]
    plot_framewise_confidence(crow_scores, base_filename)

    # Mask frames above threshold
    active = crow_scores > CONF_THRESHOLD

    # Find contiguous segments with gap tolerance
    segments, in_seg, start, silence = [], False, None, 0
    for i, a in enumerate(active):
        if a:
            if not in_seg:
                start, in_seg = i, True
            silence = 0
        else:
            if in_seg:
                silence += 1
                if silence > MAX_GAP_FRAMES:
                    end = i - silence
                    if (end - start + 1) >= MIN_CALL_FRAMES:
                        segments.append((start, end))
                    in_seg, silence = False, 0
    if in_seg:
        end = len(active) - 1
        if (end - start + 1) >= MIN_CALL_FRAMES:
            segments.append((start, end))

    if not segments:
        print(f"No crow calls in {os.path.basename(wav_path)}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, (s_frame, e_frame) in enumerate(segments, 1):
        s_samp = s_frame * HOP_SIZE
        e_samp = (e_frame + NUM_PADDING_FRAMES) * HOP_SIZE
        clip = wav[:, s_samp:e_samp]
        name = f"{base_filename}_{idx}.wav" if SAVE_ORIGINAL_NAME else f"call_{idx}.wav"
        torchaudio.save(os.path.join(OUTPUT_DIR, name), clip, PANN_SR)
        annotations.append({
            "filename":   name,
            "onset":      round(s_samp / PANN_SR, 3),
            "offset":     round(e_samp / PANN_SR, 3),
            "event_label":"crow"
        })
        print(f"Saved {name} [{s_samp/PANN_SR:.2f}-{e_samp/PANN_SR:.2f}s]")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_finetuned_model(CHECKPOINT_PATH, device)
    crow_idx = 0  # Adjust if needed to match your crow class index

    annotations = []
    for fn in sorted(os.listdir(INPUT_DIR)):
        if fn.lower().endswith(".wav"):
            print(f"Processing {fn}")
            run_sed(os.path.join(INPUT_DIR, fn), model, crow_idx, annotations)

    pd.DataFrame(annotations).to_csv(ANNOT_CSV, index=False)
    print(f"\n✅ Saved annotations to {ANNOT_CSV}")
    print(f"Framewise confidence plots saved in {PLOT_DIR}/")

if __name__ == "__main__":
    main()
