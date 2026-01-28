import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 📄 CSV FILE PATH --- #
SOFT_LABEL_CSV = "results_diag/std_1_0/clustering_assignments.csv"

# --- 📥 Load Data --- #
df = pd.read_csv(SOFT_LABEL_CSV)
df['species'] = df['Filename'].str.split('_').str[0]

# --- Ensure output directory --- #
out_dir = "umap_density_spectrogram_distinct"
os.makedirs(out_dir, exist_ok=True)


# === 📊 1️⃣ Combined Plot (All Species) ===
fig, ax = plt.subplots(figsize=(7, 5), facecolor='black')
ax.set_facecolor('black')

sns.kdeplot(
    data=df,
    x="UMAP1", y="UMAP2",
    fill=True,
    thresh=0.1,
    levels=40,
    cmap="inferno",
    bw_adjust=0.5,
    alpha=1.0,
    ax=ax
)

ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title("All Species Density", color='white', fontsize=14, weight='bold', pad=10)

plt.tight_layout()
out_path_all = os.path.join(out_dir, f"umap_density_all_species.png")
plt.savefig(out_path_all, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Saved: {out_path_all}")


# === 📊 2️⃣ Subplots for each species ===
species_list = sorted(df['species'].unique())
n_cols = 3
n_rows = -(-len(species_list) // n_cols)  # Ceiling divide

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3),
    facecolor='black', constrained_layout=True
)

axes = axes.flatten()

for idx, (ax, sp) in enumerate(zip(axes, species_list)):
    sub = df[df['species'] == sp]
    ax.set_facecolor('black')

    if len(sub) >= 10:
        sns.kdeplot(
            data=sub,
            x="UMAP1", y="UMAP2",
            fill=True,
            thresh=0.1,
            levels=40,
            cmap="inferno",
            bw_adjust=0.5,
            alpha=1.0,
            ax=ax
        )
        ax.set_title(sp.capitalize(), color='white', fontsize=12, weight='bold', pad=5)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Hide any empty plots if species < n_rows * n_cols
for i in range(len(species_list), len(axes)):
    axes[i].axis('off')

fig.suptitle("Species Density", color='white', fontsize=8, weight='bold', y=1.02)
out_path_species = os.path.join(out_dir, f"umap_density_all_species_subplots.png")
plt.savefig(out_path_species, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"✅ Saved: {out_path_species}")