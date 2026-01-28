import pandas as pd
import os
import shutil
import numpy as np

# === CONFIGURATION ===
csv_path = "results_diag/std_1_0/soft_cluster_probabilities.csv"
audio_dirs = [
    "../separation/data/ossifragus/calls",
    "../separation/data/cornix/calls",
    "../separation/data/corone/calls",
    "../separation/data/corax/calls",
    "../separation/american_crow_0.5_silence"
]
output_base_dir = "cluster_prototypes_cross_species_27"
threshold_ratio = 0.95  # Keep files with probability >= 95% of the max for this cluster
sample_n = 10  # Number of random files per cluster


# === LOAD CSV ===
df = pd.read_csv(csv_path)
cluster_cols = [col for col in df.columns if col.startswith("Cluster_")]


# === Find file in multiple directories ===
def find_file(filename, dirs):
    for d in dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


# === PROCESS EACH CLUSTER ===
for cluster in cluster_cols:
    max_prob = df[cluster].max()
    threshold = max_prob * threshold_ratio
    filtered_df = df[df[cluster] >= threshold]

    # Sample from those passing the threshold
    sampled_df = filtered_df.sample(n=min(sample_n, len(filtered_df)), random_state=None)

    # Create subdirectory for this cluster
    cluster_output_dir = os.path.join(output_base_dir, cluster)
    os.makedirs(cluster_output_dir, exist_ok=True)

    print(f"\nCopying {len(sampled_df)} random files from {len(filtered_df)} above threshold for {cluster}...")

    for idx, row in sampled_df.iterrows():
        filename = row['Filename']
        src_path = find_file(filename, audio_dirs)

        if src_path:
            dst_path = os.path.join(cluster_output_dir, filename)
            shutil.copy(src_path, dst_path)
            print(f"✔ Copied: {filename} → {cluster}/")
        else:
            print(f"⚠ File not found: {filename}")