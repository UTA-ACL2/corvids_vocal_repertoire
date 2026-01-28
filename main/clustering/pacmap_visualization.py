import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pacmap  # Make sure to install with: pip install pacmap

# === Load CSV ===
csv_path = "aves_embeddings.csv"  # change this to your file
df = pd.read_csv(csv_path)

# === Extract and scale features ===
X = df.iloc[:, 1:].values  # assuming first column is filename
X_scaled = StandardScaler().fit_transform(X)

# === Run PaCMAP ===
reducer = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
X_pacmap = reducer.fit_transform(X_scaled)

# === Plot ===
plt.figure(figsize=(8, 6))
plt.scatter(X_pacmap[:, 0], X_pacmap[:, 1], s=20, alpha=0.7)
plt.title("PaCMAP projection of AVES Embeddings")
plt.xlabel("PaCMAP-1")
plt.ylabel("PaCMAP-2")
plt.tight_layout()
plt.savefig("pacmap_projection_features_aves.png", dpi=300)