import soundfile as sf
import numpy as np
import pysptk

y, sr = sf.read("../separation/american_crow_0.5_silence/brachyrynchos_00001_1.wav")
if y.ndim > 1:
    y = y.mean(axis=1)

f0 = pysptk.swipe(y.astype(np.float64), fs=sr, hopsize=256, min=100, max=1200, otype="f0")
print(f0[f0 > 0])  # Should see non-zero values. Likely you are seeing an empty array.