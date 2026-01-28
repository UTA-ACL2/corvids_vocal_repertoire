import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.lines import Line2D
import seaborn as sns

# --- CSV FILE PATH ---
SOFT_LABEL_CSV = "results_diag/std_1_0/clustering_assignments.csv"

# --- Load CSV ---
df = pd.read_csv(SOFT_LABEL_CSV)

# --- Extract Data ---
X = df[['UMAP1', 'UMAP2']].values
cluster_labels = df['Cluster'].values
df['species'] = df['Filename'].apply(lambda f: f.split('_')[0])

# --- Species Colors (Fixed) ---
species_colors = {
    'corax':        '#1f77b4',
    'cornix':       '#ff7f0e',
    'corone':       '#2ca02c',
    'ossifragus':   '#d62728',
    'brachyrynchos':'#9467bd'
}

# --- Cluster Colors: Combine tab20 + Set3 + Dark2 for max distinctness ---
palette = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 12) + sns.color_palette("Dark2", 8)
cluster_colors = palette[:30]
cluster_cmap = ListedColormap(cluster_colors)

# --- Plot Species ---
fig, ax = plt.subplots(figsize=(7, 5))

for sp, col in species_colors.items():
    mask = df['species'] == sp
    ax.scatter(
        X[mask, 0], X[mask, 1],
        color=col,
        label=sp,
        alpha=0.4,
        s=40,
        edgecolors='k',
        linewidth=0.3
    )

ax.set_title("UMAP: Colored by Species", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=9, title_fontsize=10)
fig.tight_layout()
plt.savefig("umap_species_27.png", dpi=300, bbox_inches='tight')
plt.close()

# --- Plot Clusters ---
fig, ax = plt.subplots(figsize=(7, 5))

scatter = ax.scatter(
    X[:, 0], X[:, 1],
    c=cluster_labels,
    cmap=cluster_cmap,
    alpha=0.9,
    s=20,
    edgecolors='k',
    linewidth=0.3
)

ax.set_title("UMAP: Clusters", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])

handles = [
    Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
           markerfacecolor=cluster_colors[i], markersize=8, markeredgecolor='k')
    for i in range(30)
]
ax.legend(
    handles=handles,
    title="Clusters",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.,
    frameon=False,
    ncol=2,
    fontsize=9,
    title_fontsize=10
)

fig.tight_layout()
plt.savefig("umap_clusters_27.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 'umap_species_27.png' and 'umap_clusters_27.png' with better distinct colors")