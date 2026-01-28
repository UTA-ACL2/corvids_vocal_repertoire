import os
import numpy as np
import pandas as pd
import soundfile as sf
from glob import glob
import librosa
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# === Configuration ===
input_dir = "../separation/american_crow_calls_half_second_silence"
output_csv = "crow_features_resnet_0.5_silence.csv"
sr_target = 32000   # you can set 16000 or 32000 as you want
n_mels = 128
n_fft = 2048
hop_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18 model for feature extraction (remove final classification layer)
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # output from last conv layer (512-d)
resnet.to(device)
resnet.eval()

# ImageNet normalization
imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

def extract_resnet_features(filepath):
    try:
        y, sr = sf.read(filepath)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        duration = len(y) / sr

        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.power_to_db(S, ref=np.max)

        # Normalize spectrogram to 0-255 for image
        img = (log_S - log_S.min()) / (log_S.max() - log_S.min())
        img = (img * 255).astype(np.uint8)

        # Convert to PIL Image, resize to 224x224, convert to 3-channel RGB by duplicating
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((224, 224))
        pil_img = pil_img.convert("RGB")

        # Transform to tensor and normalize
        tensor_img = transforms.ToTensor()(pil_img)
        tensor_img = imagenet_norm(tensor_img).unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            features = resnet(tensor_img).squeeze()  # shape (512,)

        features = features.cpu().numpy()

        # Return dict with filename, duration, and 512 features
        return {
            "filename": os.path.basename(filepath),
            "log_duration": np.log(duration + 1e-12),
            **{f"resnet_feat_{i+1}": features[i] for i in range(len(features))}
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

if __name__ == "__main__":
    wav_files = sorted(glob(os.path.join(input_dir, "*.wav")))
    records = []
    for wav in wav_files:
        rec = extract_resnet_features(wav)
        if rec is not None:
            records.append(rec)

    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} calls × {len(df.columns)} columns to {output_csv}")
    else:
        print("No valid audio files processed.")