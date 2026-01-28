import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === Load CSV ===
csv_path = "features/corax/crow_features_audiomae_corax.csv"  # change this to your actual file
df = pd.read_csv(csv_path)

# === Extract and scale features ===
X = df.iloc[:, 1:].values  # assuming first column is 'filename'
X_scaled = StandardScaler().fit_transform(X)

# === Run PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === Print explained variance
print("Explained variance ratio (PCA components):", pca.explained_variance_ratio_)

# === Plot PCA ===
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=20, alpha=0.7)
plt.title("PCA projection of Embeddings")
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.tight_layout()
plt.savefig("pca_projection_features_audiomae_corax.png", dpi=300)