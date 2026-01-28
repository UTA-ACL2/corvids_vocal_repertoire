import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# === Load CSV ===
csv_path = "AudioMAE_finetuned_embeddings.csv"  # change this to your actual path
df = pd.read_csv(csv_path)

# === Extract and scale features ===
X = df.iloc[:, 1:].values  # assuming first column is 'filename'
X_scaled = StandardScaler().fit_transform(X)

# === Run t-SNE ===
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# === Plot t-SNE ===
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=20, alpha=0.7)
plt.title("t-SNE projection of AVES Embeddings")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.tight_layout()
plt.savefig("tsne_projection_features_audiomae_finetuned.png", dpi=300)