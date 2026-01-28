import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from panns_inference import SoundEventDetection, labels as panns_labels

AUDIO_PATH = "sample_sequences/denoised_split/brachyrynchos_00014.wav"
CHECKPOINT = "fine_tuned_cnn14.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 50
HOP_SIZE = 320
SR = 32000

def main():
    waveform, sr = torchaudio.load(AUDIO_PATH)
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)

    audio_np = waveform.numpy()
    model = SoundEventDetection(checkpoint_path=CHECKPOINT, device=DEVICE)
    framewise_output = model.inference(audio_np)[0]

    time_axis = np.arange(framewise_output.shape[0]) * (HOP_SIZE / SR)
    max_scores = framewise_output.max(axis=0)
    top_classes_idx = np.argsort(max_scores)[-TOP_K:][::-1]
    top_classes_idx = [int(i) for i in np.ravel(top_classes_idx)]
    labels = list(panns_labels)
    top_labels = [labels[i] if i < len(labels) else f"Unknown({i})" for i in top_classes_idx]

    plt.figure(figsize=(12, 6))
    for i, class_idx in enumerate(top_classes_idx):
        plt.plot(time_axis, framewise_output[:, class_idx], label=top_labels[i])
    plt.title(f"PANNs Framewise Predictions\n{os.path.basename(AUDIO_PATH)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("panns_output_plot.png")
    print("✅ Saved plot to panns_output_plot.png")

if __name__ == "__main__":
    main()
