import os
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# these four modules come from the copied files
import vggish_input
import vggish_params
import vggish_slim
import vggish_postprocess

# ——— CONFIG ———
INPUT_DIR     = "../separation/american_crow_calls_half_second_silence"
OUTPUT_CSV    = "crow_features_vggish.csv"
SR            = 48000
CHECKPOINT    = "vggish_model.ckpt"
PCA_PARAMS    = "vggish_pca_params.npz"

# ——— BUILD MODEL ONCE ———
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()
vggish_slim.define_vggish_slim(training=False)
vggish_slim.load_vggish_slim_checkpoint(sess, CHECKPOINT)

features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
postprocessor = vggish_postprocess.Postprocessor(PCA_PARAMS)

# ——— PROCESS FILES ———
records = []
files = sorted(os.listdir(INPUT_DIR))
for i, fn in enumerate(files, 1):
    if not fn.lower().endswith(".wav"):
        continue
    path = os.path.join(INPUT_DIR, fn)
    # 1) convert waveform to VGGish input examples
    examples = vggish_input.wavfile_to_examples(path)
    # 2) run the model
    [emb_batch] = sess.run([embedding_tensor],
                           feed_dict={features_tensor: examples})
    # 3) PCA & quantize
    emb_batch = postprocessor.postprocess(emb_batch)
    # 4) average over time
    clip_emb = emb_batch.mean(axis=0)
    # 5) record
    rec = {"filename": fn}
    rec.update({f"vggish_{j+1}": float(clip_emb[j]) for j in range(clip_emb.shape[0])})
    records.append(rec)
    print(f"[{i}/{len(files)}] {fn} ✓")

# ——— SAVE CSV ———
pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(records)} embeddings → {OUTPUT_CSV}")