import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
SOFT_LABEL_CSV    = "results_diag/std_1_0/soft_cluster_probabilities.csv"
OUTPUT_PLOT_ALL   = "sequence_length_boxplot_all_species_log2.png"
OUTPUT_PLOT_GROUP = "sequence_length_boxplot_by_species_log2.png"
OUTPUT_PLOT_DIR   = "sequence_length_by_species_boxplots_log2"

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- LOAD & PARSE ---
df = pd.read_csv(SOFT_LABEL_CSV)
def extract_info(fname):
    sp, sid, ci = fname.replace(".wav","").split("_")
    return sp, sid, int(ci)
df[['species','seq_id','call_idx']] = df['filename'].apply(
    lambda x: pd.Series(extract_info(x))
)

# --- COMPUTE SEQUENCE LENGTHS ---
seq_lengths = df.groupby(['species','seq_id']).size().reset_index(name='sequence_length')

# --- LOG2 TRANSFORM ---
seq_lengths['log2_seq_len'] = np.log2(seq_lengths['sequence_length'])

# --- TICKS SETUP (powers of 2) ---
max_len = seq_lengths['sequence_length'].max()
ticks = [2**i for i in range(0, int(np.ceil(np.log2(max_len)))+1)]
ticks = [t for t in ticks if seq_lengths['sequence_length'].min() <= t <= seq_lengths['sequence_length'].max()]
log2_ticks = np.log2(ticks)

# --- PLOT 1: All species combined ---
plt.figure(figsize=(6,5))
sns.boxplot(y='log2_seq_len', data=seq_lengths, color='lightblue', showfliers=False)
ax = plt.gca()
ax.set_yticks(log2_ticks)
ax.set_yticklabels(ticks)
ax.set_ylabel("Sequence Length (calls) [log2 scale]")
plt.title("Sequence Length Distribution (All Species, log2 scale)")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_ALL, dpi=300)
plt.close()
print(f"✅ Saved {OUTPUT_PLOT_ALL}")

# --- PLOT 2: All species side-by-side ---
plt.figure(figsize=(8,5))
sns.boxplot(x='species', y='log2_seq_len', data=seq_lengths, palette='Set2', showfliers=False)
ax = plt.gca()
ax.set_yticks(log2_ticks)
ax.set_yticklabels(ticks)
ax.set_ylabel("Sequence Length (calls) [log2 scale]")
plt.title("Sequence Length by Species (log2 scale)")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_GROUP, dpi=300)
plt.close()
print(f"✅ Saved {OUTPUT_PLOT_GROUP}")

# --- PLOT 3: Individual per-species ---
for sp, grp in seq_lengths.groupby('species'):
    plt.figure(figsize=(4,4))
    sns.boxplot(y='log2_seq_len', data=grp, color='lightblue', showfliers=False)
    ax = plt.gca()
    ax.set_yticks(log2_ticks)
    ax.set_yticklabels(ticks)
    ax.set_ylabel("Sequence Length (calls) [log2 scale]")
    plt.title(f"{sp}: Sequence Length (log2 scale)")
    plt.tight_layout()
    out = os.path.join(OUTPUT_PLOT_DIR, f"sequence_length_boxplot_{sp}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Saved {out}")
