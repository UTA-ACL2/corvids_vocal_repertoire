import os
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import openl3
from glob import glob

# ——— CONFIG ———
INPUT_DIR    = "../separation/american_crow_calls_half_second_silence"
OUTPUT_CSV   = "crow_features.csv"
SR           = 48000
EMBED_SIZE   = 512
CONTENT_TYPE = "env"
HOP_SIZE     = 0.3  # fewer frames ⇒ faster & lower memory

# ——— QUIET MODE ———
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", module="openl3")

# ——— GPU SETUP ———
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_and_warm(device_gpu: bool):
    tf.keras.backend.clear_session()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device_gpu else "-1"
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type=CONTENT_TYPE,
        embedding_size=EMBED_SIZE
    )
    # Warm‑up pass
    dummy = np.zeros(SR * 2, dtype=np.float32)
    _ = openl3.get_audio_embedding(dummy, SR,
                                   model=model,
                                   hop_size=HOP_SIZE,
                                   center=True)
    return model

def embed_file(path: str, model) -> dict:
    """Returns {filename, log_duration, feat_1..feat_N} or raises."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    emb, _ = openl3.get_audio_embedding(
        audio, sr,
        model=model,
        hop_size=HOP_SIZE,
        center=True
    )
    # If emb empty, fall back to zeros:
    if emb.size == 0:
        emb = np.zeros((1, EMBED_SIZE), dtype=np.float32)
    mean = emb.mean(axis=0).astype(np.float32)
    rec = {
        "filename": os.path.basename(path),
        "log_duration": float(np.log(len(audio)/sr + 1e-12))
    }
    rec.update({f"feat_{i+1}": float(mean[i]) for i in range(EMBED_SIZE)})
    return rec

def main():
    files = sorted(glob(os.path.join(INPUT_DIR, "*.wav")))
    if not files:
        print("No .wav files found.")
        return

    # Load models
    use_gpu = bool(gpus)
    model_gpu = load_and_warm(use_gpu)
    model_cpu = None

    results = []
    for idx, fp in enumerate(files, 1):
        name = os.path.basename(fp)
        try:
            rec = embed_file(fp, model_gpu)
            src = "GPU"
        except tf.errors.ResourceExhaustedError:
            # GPU OOM → CPU fallback
            if model_cpu is None:
                model_cpu = load_and_warm(False)
            try:
                rec = embed_file(fp, model_cpu)
                src = "CPU"
            except Exception:
                rec = None
        except Exception:
            rec = None

        if rec:
            results.append(rec)
            print(f"[{idx}/{len(files)}] {name} ✓ ({src})")
        else:
            print(f"[{idx}/{len(files)}] {name} ✗")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done: {len(results)}/{len(files)} succeeded → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
