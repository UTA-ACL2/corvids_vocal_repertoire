import os
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- 🔧 PARAMETERS TO CONFIGURE --- #
csv_files = [
    "features/corax/crow_features_audiomae_corax.csv",
    "features/cornix/crow_features_audiomae_cornix.csv",
    "features/corone/crow_features_audiomae_corone.csv",
    "features/ossifragus/crow_features_audiomae_ossifragus.csv",
    "features/brachyrynchos/0.5_silence/crow_features_audiomae_brachyrynchos.csv"
]

n_components = 27  # Number of GMM clusters

# --- 📂 Output Files --- #
SOFT_PROB_OUTPUT = "cross_species_0.5_audiomae_soft_cluster_probabilities_27.csv"
PLOT_OUTPUT = "cross_species_0.5_umap_audiomae_gmm_clusters_27.png"

# ---------------------------------- #

# --- 1️⃣ Load and Combine CSVs --- #
dfs = [pd.read_csv(path) for path in csv_files]
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df)} rows from {len(csv_files)} CSV files.")

# --- Extract species from filename prefix and add as a new column --- #
def extract_species(filename):
    return filename.split('_')[0]

df['species'] = df['filename'].apply(extract_species)

# --- 2️⃣ Extract Features & Filenames --- #
non_feature_cols = ['filename', 'species']
feature_cols = [col for col in df.columns if col not in non_feature_cols]

features = df[feature_cols].values
filenames = df['filename'].values
species_labels = df['species'].values

# --- 2.5️⃣ Standardize Features --- #
#scaler = StandardScaler()
#features_scaled = scaler.fit_transform(features)
features_scaled = features

# --- 3️⃣ Dimensionality Reduction with UMAP (2D for visualization only) --- #
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
embedding_2d = umap_reducer.fit_transform(features_scaled)

# --- 4️⃣ Gaussian Mixture Model Clustering --- #
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(features_scaled)
cluster_probs = gmm.predict_proba(features_scaled)  # Soft probabilities
cluster_labels = cluster_probs.argmax(axis=1)

# --- 5️⃣ Evaluate Clustering Quality --- #
if len(set(cluster_labels)) > 1:
    silhouette = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette Score (higher is better): {silhouette:.3f}")
else:
    silhouette = None
    print("GMM found only one cluster; silhouette score not applicable.")

# --- 6️⃣ Save Soft Probabilities to CSV --- #
prob_df = pd.DataFrame(cluster_probs, columns=[f"cluster_{i}" for i in range(n_components)])
prob_df.insert(0, 'filename', filenames)
prob_df.to_csv(SOFT_PROB_OUTPUT, index=False)
print(f"Saved soft probabilities to: {SOFT_PROB_OUTPUT}")

# --- 7️⃣ Save UMAP Plot to File with Transparency and Species Colors --- #
plt.figure(figsize=(10, 5))

# Distinct colors from matplotlib tab10 or tab20
unique_species = sorted(set(species_labels))
cmap = plt.get_cmap('tab10')
colors = [cmap(i % 10) for i in range(len(unique_species))]

# --- UMAP Colored by Species --- #
plt.subplot(1, 2, 1)
for color, species in zip(colors, unique_species):
    idx = species_labels == species
    plt.scatter(
        embedding_2d[idx, 0],
        embedding_2d[idx, 1],
        label=species,
        color=color,
        alpha=0.4,
        edgecolors='none',
        s=20
    )

plt.legend()
plt.title("UMAP Colored by Species (Transparent Overlap)")

# --- UMAP Colored by GMM Clusters --- #
plt.subplot(1, 2, 2)
plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=cluster_labels,
    cmap='Spectral',
    alpha=0.7,
    edgecolors='none',
    s=20
)
plt.title(f"UMAP Colored by GMM Clusters (n={n_components})")
if silhouette is not None:
    plt.suptitle(f"Silhouette Score: {silhouette:.3f}", y=1.05, fontsize=10, color='gray')
plt.colorbar(label='Cluster ID')

plt.tight_layout()
plt.savefig(PLOT_OUTPUT, dpi=300)
plt.close()
print(f"Saved UMAP + GMM plot to: {PLOT_OUTPUT}")