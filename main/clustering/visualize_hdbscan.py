import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import hdbscan
import seaborn as sns

# === Configuration ===
input_csv = "crow_features_audiomae_contrastive_brachyrynchos.csv"
output_csv = "crow_cluster_probabilities_audiomae_posttrained_hdbscan.csv"
output_png = "umap_hdbscan_clusters_audiomae_contrastive_brachyrynchos.png"

# === Load CSV ===
df = pd.read_csv(input_csv)
filenames = df["filename"] if "filename" in df.columns else None
X = df.drop(columns=["filename"], errors="ignore").values

# === Normalize ===
X_scaled = StandardScaler().fit_transform(X)

# === Reduce dimensions (PCA → UMAP) ===
X_pca = PCA(n_components=50).fit_transform(X_scaled)
X_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="hamming").fit_transform(X)

# === Fit HDBSCAN ===
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, prediction_data=True)
labels = clusterer.fit_predict(X_umap)

# === Compute soft cluster membership probabilities ===
probs = hdbscan.all_points_membership_vectors(clusterer)  # shape: (n_samples, n_clusters)

# === Format probabilities for CSV ===
probs_df = pd.DataFrame(probs, columns=[f"cluster_{i}" for i in range(probs.shape[1])])
if filenames is not None:
    probs_df.insert(0, "filename", filenames)

# === Save CSV ===
probs_df.to_csv(output_csv, index=False)
print(f"Saved predicted cluster probabilities to {output_csv}")

# === Plot and save UMAP with HDBSCAN cluster labels ===
plt.figure(figsize=(10, 8))
palette = sns.color_palette("tab10", n_colors=len(set(labels)) - (1 if -1 in labels else 0))
sns.scatterplot(
    x=X_umap[:, 0],
    y=X_umap[:, 1],
    hue=labels,
    palette=palette,
    s=10,
    alpha=0.8,
    legend="full"
)
plt.title("HDBSCAN Clusters (UMAP Projection)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()
print(f"Saved UMAP visualization to {output_png}")