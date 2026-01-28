import os
import numpy as np
import pandas as pd
from collections import defaultdict

# === CONFIGURATION ===
INPUT_CSV = "results_diag/std_1_0/soft_cluster_probabilities.csv"
OUTPUT_DIR = "transition_matrices"
SEPARATE_BY_SPECIES = True  # 🔧 Set to False for combined only
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD ===
probs_df = pd.read_csv(INPUT_CSV)
cluster_cols = [c for c in probs_df.columns if c.startswith("Cluster") or c.startswith("cluster")]
probs = probs_df[cluster_cols].values
filenames = probs_df['filename'].values

# === UTILITY ===
def get_group_id(filename):
    return "_".join(filename.split("_")[:2])  # species + sequence ID

def get_species(filename):
    return filename.split("_")[0]

# === GROUP SEQUENCES ===
grouped_by_species = defaultdict(lambda: defaultdict(list))
for i, fname in enumerate(filenames):
    species = get_species(fname)
    group_id = get_group_id(fname)
    grouped_by_species[species][group_id].append((i, probs[i]))

n_clusters = probs.shape[1]
n_states = n_clusters + 2  # START + CLUSTERS + END
START, END = 0, n_states - 1
CLUSTER_OFFSET = 1

# === FUNCTION TO BUILD TRANSITION MATRIX ===
def build_transition_matrix(grouped):
    transition_matrix = np.zeros((n_states, n_states))

    for group_id, sequence in grouped.items():
        indices, prob_sequence = zip(*sequence)
        # START to first cluster
        transition_matrix[START, CLUSTER_OFFSET:CLUSTER_OFFSET + n_clusters] += prob_sequence[0]
        # CLUSTER to CLUSTER
        for i in range(len(prob_sequence) - 1):
            p_i = prob_sequence[i]
            p_j = prob_sequence[i + 1]
            transition_matrix[
                CLUSTER_OFFSET : CLUSTER_OFFSET + n_clusters,
                CLUSTER_OFFSET : CLUSTER_OFFSET + n_clusters
            ] += np.outer(p_i, p_j)
        # LAST to END
        transition_matrix[
            CLUSTER_OFFSET : CLUSTER_OFFSET + n_clusters,
            END
        ] += prob_sequence[-1]

    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_matrix,
        row_sums,
        out=np.zeros_like(transition_matrix),
        where=row_sums != 0
    )
    return transition_probs

# === COMBINED ACROSS SPECIES ===
all_grouped = {}
for species_group in grouped_by_species.values():
    all_grouped.update(species_group)

transition_probs = build_transition_matrix(all_grouped)

labels = ['START'] + [f'Cluster_{i}' for i in range(n_clusters)] + ['END']
transition_df = pd.DataFrame(transition_probs, index=labels, columns=labels)
combined_csv = os.path.join(OUTPUT_DIR, "soft_markov_transition_matrix_all_species.csv")
transition_df.to_csv(combined_csv)
print(f"✅ Saved: {combined_csv}")

# === PER-SPECIES ===
if SEPARATE_BY_SPECIES:
    for species, groups in grouped_by_species.items():
        trans = build_transition_matrix(groups)
        trans_df = pd.DataFrame(trans, index=labels, columns=labels)
        out_file = os.path.join(OUTPUT_DIR, f"soft_markov_transition_matrix_{species}.csv")
        trans_df.to_csv(out_file)
        print(f"✅ Saved: {out_file}")