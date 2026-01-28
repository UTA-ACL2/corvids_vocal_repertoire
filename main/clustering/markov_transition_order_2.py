import numpy as np
import pandas as pd
from collections import defaultdict

# 1) Load soft cluster probabilities
probs_df = pd.read_csv("crow_cluster_probabilities_70_mfcc.csv")  # expects 'filename' + cluster probability columns
cluster_cols = probs_df.columns.drop("filename")
probs = probs_df[cluster_cols].values           # shape: (n_samples, n_clusters)
filenames = probs_df["filename"].values

# 2) Group by original audio ID (prefix before subfile index)
def get_group_id(fname):
    return "_".join(fname.split("_")[:2])

grouped = defaultdict(list)
for idx, fname in enumerate(filenames):
    gid = get_group_id(fname)
    grouped[gid].append(idx)

# 3) Define state counts and indices
n_clusters = probs.shape[1]
START, END = 0, n_clusters + 1
n_states = n_clusters + 2    # {START, cluster0…clusterN-1, END}
    
# Helper to pad a soft-vector into full state space
def pad_vec(p):
    v = np.zeros(n_states)
    v[1:1 + n_clusters] = p   # put cluster probs into 1…n_clusters
    return v

# 4) Prepare the collapsed 2nd-order transition counts
#    Rows = ordered pairs of (prev_prev_state, prev_state) => n_states^2
#    Cols = next_state => n_states
T = np.zeros((n_states * n_states, n_states))

# 5) For each sequence, build a padded list [START, p0, p1, …, p_{L-1}, END]
for gid, indices in grouped.items():
    seq = [pad_vec(np.zeros(n_clusters))]  # placeholder for START→START pair, but we'll override
    seq += [pad_vec(probs[i]) for i in sorted(indices)]
    seq += [pad_vec(np.zeros(n_clusters))]  # END vector is one-hot at END
    
    # actually set START and END one-hots
    seq[0] = np.zeros(n_states); seq[0][START] = 1
    seq[-1] = np.zeros(n_states); seq[-1][END] = 1

    # accumulate transitions
    # for each triplet (t, t+1, t+2), add to T at row=(t,t+1) pair, col=t+2
    for t in range(len(seq) - 2):
        p0, p1, p2 = seq[t], seq[t+1], seq[t+2]
        # row-vector of length n_states^2 = outer(p0,p1).ravel()
        row_probs = np.outer(p0, p1).ravel()
        # now broadcast multiply by p2 to get a (n_states^2 × n_states) block
        T += row_probs[:, None] * p2[None, :]

# 6) Normalize each row to sum to 1 (where non-zero)
row_sums = T.sum(axis=1, keepdims=True)
T_probs = np.divide(T, row_sums, out=np.zeros_like(T), where=row_sums != 0)

# === Enforce exact row normalization ===
# Compute row sums of T_probs
row_sums = T_probs.sum(axis=1, keepdims=True)

# Where the sum is > 0, divide each row by its own sum
nonzero = row_sums[:, 0] > 0
T_probs[nonzero] = T_probs[nonzero] / row_sums[nonzero]

# (Optional) Round to, say, 12 decimal places to eliminate leftover noise
T_probs = np.round(T_probs, 12)

# 7) Build human-readable row & column labels
def name(s):
    if s == START: return "START"
    if s == END:   return "END"
    return f"C{s-1}"

# rows: all ordered pairs (i,j)
row_labels = [f"({name(i)},{name(j)})"
              for i in range(n_states) for j in range(n_states)]
col_labels = [name(i) for i in range(n_states)]

# 8) Save to CSV
df_out = pd.DataFrame(T_probs, index=row_labels, columns=col_labels)
df_out.to_csv("soft_markov_2nd_order_collapsed_70_mfcc.csv")

print("Saved 2nd-order collapsed soft Markov matrix to 'soft_markov_2nd_order_collapsed_70_mfcc.csv'")