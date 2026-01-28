import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator

# === CONFIGURATION ===
CSV_PATH   = "results_diag/std_1_0/clustering_assignments.csv"
OUTPUT_PNG = "cluster_call_index_probabilities_grid.png"

# === LOAD & PARSE ===
df = pd.read_csv(CSV_PATH)
assert {'Filename','Cluster'}.issubset(df.columns), "CSV must contain 'Filename' and 'Cluster'."
df['call_idx'] = (
    df['Filename']
      .str.replace('.wav','', regex=False)
      .str.split('_')
      .str[-1]
      .astype(int)
)

# === GRID LAYOUT ===
clusters = sorted(df['Cluster'].unique())
n_clusters = len(clusters)
n_cols = 7
n_rows = math.ceil(n_clusters / n_cols)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols*3, n_rows*2.8),
                         constrained_layout=True)
axes = axes.flatten()

# === TICK FORMATTERS ===
def pct_fmt(x, pos):
    return f"{int(x*100)}%" if x>=0 else ""

y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]  # show 0%,25%,50%,75%,100%

# === PLOT EACH CLUSTER ===
for idx, (ax, cl) in enumerate(zip(axes, clusters)):
    sub = df[df['Cluster']==cl]
    sns.histplot(
        data=sub,
        x='call_idx',
        stat='probability',
        binwidth=1,
        shrink=0.8,
        edgecolor='black',
        color='steelblue',
        ax=ax
    )
    # X-axis: integer ticks, but only bottom row labeled
    ax.set_xlim(1, df['call_idx'].max()+1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    if idx < (n_rows-1)*n_cols:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Call Index", fontsize=9)
    # Y-axis: percentage ticks, only first column labeled
    ax.set_ylim(0, 1)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    if idx % n_cols != 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Percent of Sequence", fontsize=9)
    # Title
    ax.set_title(f"Cluster {cl}", fontsize=9, weight='bold')

# TURN OFF EXTRA PANELS
for ax in axes[n_clusters:]:
    ax.axis('off')

fig.suptitle("Per‐Cluster Call‑Index Distributions (Linear scales)", fontsize=14, weight='bold')
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()
print(f"✅ Saved: {OUTPUT_PNG}")
