import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns  # For better palettes

# --- 🔧 CONFIGURE FILE PATHS --- #
FEATURE_CSV_FILES = [
    "features/corax/crow_features_audiomae_corax.csv",
    "features/cornix/crow_features_audiomae_cornix.csv",
    "features/corone/crow_features_audiomae_corone.csv",
    "features/ossifragus/crow_features_audiomae_ossifragus.csv",
    "features/brachyrynchos/0.5_silence/crow_features_audiomae_brachyrynchos.csv"
]
SOFT_LABEL_CSV       = "results_diag/std_1_0/soft_cluster_probabilities.csv"
OUTPUT_SPECIES_PNG   = "umap_species.png"
OUTPUT_CLUSTERS_PNG  = "umap_clusters.png"

# --- 1️⃣ Load & Merge --- #
feat_dfs = [pd.read_csv(p) for p in FEATURE_CSV_FILES]
soft_df  = pd.read_csv(SOFT_LABEL_CSV)

# Normalize filename column name in each DF
for df_ in feat_dfs + [soft_df]:
    if 'filename' in df_.columns and 'Filename' not in df_.columns:
        df_.rename(columns={'filename':'Filename'}, inplace=True)

features_df = pd.concat(feat_dfs, ignore_index=True)
df = pd.merge(features_df, soft_df, on='Filename', how='inner')
print(f"Merged {len(df)} rows.")

# Inspect columns
print("All columns:", df.columns.tolist())

# --- 2️⃣ Extract species & cluster columns --- #
# Normalize species
df['species'] = df['Filename'].apply(lambda f: f.split('_')[0])

# Detect soft label columns robustly
soft_cols = [c for c in df.columns if c.lower().startswith('cluster_')]
if not soft_cols:
    raise ValueError(f"No soft‑cluster columns found (tried prefix 'cluster_'). Available columns:\n{df.columns.tolist()}")

feature_cols = [c for c in df.columns if c not in ['Filename','species'] + soft_cols]

X = df[feature_cols].values
cluster_labels = df[soft_cols].values.argmax(axis=1)
species_labels = df['species'].values

# --- Add tiny noise to avoid spectral initialization warning ---
noise_level = 1e-6
X_jittered = X + noise_level * np.random.randn(*X.shape)

# --- 3️⃣ UMAP embedding --- #
umap_reducer = umap.UMAP(n_neighbors=5, min_dist=0.0, spread=1.5, random_state=42)
embedding = umap_reducer.fit_transform(X_jittered)

# --- 4️⃣ Prepare color palettes --- #
# Species (fixed)
species_colors = {
    'corax':        '#1f77b4',
    'cornix':       '#ff7f0e',
    'corone':       '#2ca02c',
    'ossifragus':   '#d62728',
    'brachyrynchos':'#9467bd'
}
species_alpha = 0.2

# Clusters (distinct 30 colors)
n_clusters = cluster_labels.max() + 1
palette = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 12)
cluster_colors = palette[:n_clusters]
cluster_cmap = ListedColormap(cluster_colors)
cluster_alpha = 0.9

# --- 5️⃣ Plot & Save: Species --- #
fig, ax = plt.subplots(figsize=(7,5))
for sp, col in species_colors.items():
    mask = (species_labels == sp)
    ax.scatter(
        embedding[mask, 0], embedding[mask, 1],
        color=col, label=sp,
        alpha=species_alpha, s=40,
        edgecolors='k', linewidth=0.1
    )
ax.set_title("UMAP: Colored by Species", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(title="Species", bbox_to_anchor=(1.05,1), loc='upper left',
          frameon=False, fontsize=9, title_fontsize=10)
fig.tight_layout()
plt.savefig(OUTPUT_SPECIES_PNG, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_SPECIES_PNG}")

# --- 6️⃣ Plot & Save: Clusters --- #
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(
    embedding[:,0], embedding[:,1],
    c=cluster_labels, cmap=cluster_cmap,
    alpha=cluster_alpha, s=20,
    edgecolors='k', linewidth=0.1
)
ax.set_title(f"UMAP: {n_clusters} Clusters (Legend Only)", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])

handles = [
    Line2D([0],[0], marker='o', color='w', label=f'Cluster {i}',
           markerfacecolor=cluster_colors[i], markersize=8, markeredgecolor='k')
    for i in range(n_clusters)
]
ax.legend(handles=handles, title="Clusters",
          bbox_to_anchor=(1.05,1), loc='upper left',
          frameon=False, ncol=2, fontsize=9, title_fontsize=10)

fig.tight_layout()
plt.savefig(OUTPUT_CLUSTERS_PNG, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_CLUSTERS_PNG}")
