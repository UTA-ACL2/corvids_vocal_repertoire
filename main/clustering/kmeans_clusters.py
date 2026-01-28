import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap

# === Load features from CSV ===
df = pd.read_csv("crow_features_audiomae_finetuned_repetition.csv")  # <-- change filename if needed
filenames = df["filename"]
X = df.drop(columns=["filename"], errors='ignore').values

# === Standardize the feature matrix ===
X_scaled = StandardScaler().fit_transform(X)

# === Run K-Means Clustering ===
n_clusters = 20  # <-- change this number to what you want
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# === UMAP for visualization ===
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# === Save cluster labels to CSV ===
output_df = pd.DataFrame({
    "filename": filenames,
    "cluster": labels
})
output_df.to_csv("crow_kmeans_clusters_audiomae_rep.csv", index=False)
print("Saved K-Means cluster labels to csv")

# === Save soft cluster probabilities (one-hot) ===
probs = np.eye(n_clusters)[labels]  # Convert to one-hot encoding
probs_df = pd.DataFrame(probs, columns=[f"cluster_{i}" for i in range(n_clusters)])
probs_df.insert(0, "filename", filenames)
probs_df.to_csv("crow_kmeans_probabilities_audiomae_rep.csv", index=False)
print("Saved K-Means one-hot cluster assignments to csv")

# === Save UMAP visualization ===
plt.figure(figsize=(10, 6))
plt.title(f"K-Means Clustering (k={n_clusters}) + UMAP")
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="tab20", s=10)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("crow_kmeans_umap_audiomae_rep.png", dpi=300)
plt.show()
print("Saved UMAP visualization to png")