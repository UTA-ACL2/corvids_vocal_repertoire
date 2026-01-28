import pandas as pd
import umap
import matplotlib.pyplot as plt

# === Load CSV ===
csv_path = "features/cornix/crow_features_audiomae_cornix.csv"  # change this to your CSV path
df = pd.read_csv(csv_path)

# === Extract features (exclude 'filename') ===
X = df.iloc[:, 1:].values   # assuming first column is filename

# === Run UMAP ===
reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42, metric="cosine")
X_umap = reducer.fit_transform(X)

# === Plot (no text) ===
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=40, alpha=0.7)
plt.title("UMAP projection of Crow Call Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()

# === Save to PNG ===
plt.savefig("umap_projection_features_audiomae_cornix.png", dpi=300)