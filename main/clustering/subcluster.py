import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from string import ascii_lowercase

# ==== CONFIGURATION ====
SOFT_CLUSTER_CSV = "crow_cluster_probabilities_resnet_10.csv"
REPETITION_CSV = "../counting/gaussian_repetition_counts.csv"
OUTPUT_CSV = "subcluster_assignments_resnet_10.csv"
MAX_SUBCLUSTERS = 3
MIN_CLUSTER_PROB = 0.01
# =======================

# Load input data
probs_df = pd.read_csv(SOFT_CLUSTER_CSV)
reps_df = pd.read_csv(REPETITION_CSV)

# Merge on filename
merged = pd.merge(probs_df, reps_df, on="filename", how="inner")
cluster_cols = [col for col in probs_df.columns if col.startswith("cluster_")]

# Prepare subcluster columns (fixed order)
subcluster_columns = []
for cluster_col in cluster_cols:
    for j in range(MAX_SUBCLUSTERS):
        subcluster_columns.append(f"{cluster_col}_{ascii_lowercase[j]}")

# Fit BGMM models per cluster once (outside row loop)
bgmm_models = {}
for cluster_col in cluster_cols:
    cluster_subset = merged[merged[cluster_col] > MIN_CLUSTER_PROB]
    reps_k = cluster_subset["repetition_count"].values.reshape(-1, 1)
    if len(reps_k) < 2:
        continue
    bgmm = BayesianGaussianMixture(
        n_components=MAX_SUBCLUSTERS,
        covariance_type="full",
        random_state=0
    ).fit(reps_k)
    bgmm_models[cluster_col] = bgmm

# Prepare output data list
output_rows = []

for _, row in merged.iterrows():
    repetition = row["repetition_count"]
    row_data = {"filename": row["filename"]}
    rep_vec = np.array([[repetition]])

    # Initialize all subcluster probs to 0
    for sub_col in subcluster_columns:
        row_data[sub_col] = 0.0

    # Fill subcluster probs
    for cluster_col in cluster_cols:
        prob_k = row[cluster_col]
        if prob_k < MIN_CLUSTER_PROB:
            continue
        bgmm = bgmm_models.get(cluster_col, None)
        if bgmm is None:
            continue
        sub_probs = bgmm.predict_proba(rep_vec)[0]
        for j, p_sub in enumerate(sub_probs):
            sub_col = f"{cluster_col}_{ascii_lowercase[j]}"
            row_data[sub_col] = prob_k * p_sub

    output_rows.append(row_data)

# Build DataFrame and normalize rows
final_df = pd.DataFrame(output_rows)
final_df[subcluster_columns] = final_df[subcluster_columns].div(
    final_df[subcluster_columns].sum(axis=1), axis=0
)

# Save to CSV
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved normalized soft subcluster CSV to: {OUTPUT_CSV}")
