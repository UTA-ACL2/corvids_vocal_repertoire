import os
import numpy as np
import pandas as pd
import soundfile as sf
from glob import glob
import torch
import librosa
from panns_inference import AudioTagging

input_dir = "../separation/american_crow_calls"
output_csv = "crow_features_panns_embeddings.csv"
sr_target = 32000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioTagging(device=device)

def extract_panns_embedding(filepath):
    try:
        y, sr = sf.read(filepath)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        duration = len(y) / sr
        y = y.astype(np.float32)

        input_tensor = torch.tensor(y).unsqueeze(0).to(device)

        with torch.no_grad():
            _, embedding = model.inference(input_tensor)

        if hasattr(embedding, "cpu"):
            embedding = embedding.cpu().numpy()
        else:
            embedding = np.array(embedding)

        if embedding.ndim == 2:
            embedding = embedding.mean(axis=0)
        elif embedding.ndim == 1:
            pass
        else:
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}")

        return {
            "filename": os.path.basename(filepath),
            "log_duration": np.log(duration + 1e-12),
            **{f"panns_feat_{i+1}": embedding[i] for i in range(len(embedding))}
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

if __name__ == "__main__":
    wav_files = sorted(glob(os.path.join(input_dir, "*.wav")))
    records = []
    for wav in wav_files:
        rec = extract_panns_embedding(wav)
        if rec is not None:
            records.append(rec)

    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} calls × {len(df.columns)} columns to {output_csv}")
    else:
        print("No valid audio files processed.")
