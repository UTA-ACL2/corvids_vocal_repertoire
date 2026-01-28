import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

# === CONFIGURATION ===
OUTPUT_DIR = 'results_tied2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_RANGE = range(2, 101)  # K from 2 to 100
RANDOM_SEED = 0
STD_THRESHOLDS = [0.5, 1.0]

# === INPUT: List of CSV file paths for embeddings ===
embedding_files = [
    "features/corax/crow_features_audiomae_corax.csv",
    "features/cornix/crow_features_audiomae_cornix.csv",
    "features/corone/crow_features_audiomae_corone.csv",
    "features/ossifragus/crow_features_audiomae_ossifragus.csv",
    "features/brachyrynchos/0.5_silence/crow_features_audiomae_brachyrynchos.csv"
]

# === LOAD AND CONCATENATE ALL EMBEDDINGS ===
embedding_dfs = [pd.read_csv(f, header=0) for f in embedding_files]
full_df = pd.concat(embedding_dfs, ignore_index=True)

# Extract filenames (first column)
filenames = full_df.iloc[:, 0].tolist()

# Extract features (all columns except the first)
X_raw = full_df.iloc[:, 1:].values.astype(np.float32)

# Standardize features before clustering and UMAP
#X = StandardScaler().fit_transform(X_raw)
X = X_raw

# === 1. UMAP FOR VISUALIZATION ===
umap = UMAP(random_state=RANDOM_SEED)
umap_embeddings = umap.fit_transform(X)
umap_df = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
umap_df['Filename'] = filenames
umap_df.to_csv(os.path.join(OUTPUT_DIR, 'umap_embeddings.csv'), index=False)

# === Helper function to compute BIC for given K ===
def bic_for_k(X, K, covariance_type='tied', random_state=0):
    gm = GaussianMixture(
        n_components=K,
        covariance_type=covariance_type,
        random_state=random_state,
        reg_covar=1e-4,
        max_iter=500
    )
    gm.fit(X)
    return gm.bic(X)

# === 2. TIED-COVARIANCE GMM BIC SCAN (PARALLEL) ===
print("Running tied-covariance BIC tests (parallel)...")
tied_bics = Parallel(n_jobs=-1)(
    delayed(bic_for_k)(X, K, covariance_type='tied', random_state=RANDOM_SEED) for K in tqdm(K_RANGE)
)
tied_bics = np.array(tied_bics)

# Save min BIC and corresponding K
bic_min = tied_bics.min()
bic_min_K = K_RANGE[np.argmin(tied_bics)]
bic_std = tied_bics.std()

with open(os.path.join(OUTPUT_DIR, 'min_bic.txt'), 'w') as f:
    f.write(f'Minimum BIC: {bic_min:.4f}\n')
    f.write(f'K at minimum BIC: {bic_min_K}\n')

# === 3. LOOP OVER STD THRESHOLDS ===
for std_multiplier in STD_THRESHOLDS:
    threshold = bic_min + bic_std * std_multiplier
    candidate_Ks = [K for K, bic in zip(K_RANGE, tied_bics) if bic <= threshold]
    if not candidate_Ks:
        print(f"No Ks within {std_multiplier} std threshold, skipping.")
        continue
    refined_K = min(candidate_Ks)

    # Refine full covariance around refined_K (just K itself here)
    refinement_Ks = [refined_K]
    full_bics = {}
    full_gmms = {}

    print(f"Running full-covariance GMM refinement around K={refined_K} for threshold {std_multiplier}...")
    for K in refinement_Ks:
        gm = GaussianMixture(
            n_components=K,
            covariance_type='full',
            random_state=RANDOM_SEED,
            reg_covar=1e-4,
            max_iter=500
        )
        gm.fit(X)
        full_bics[K] = gm.bic(X)
        full_gmms[K] = gm

    final_K = min(full_bics, key=full_bics.get)
    final_model = full_gmms[final_K]
    final_labels = final_model.predict(X)
    probabilities = final_model.predict_proba(X)  # Soft probabilities here

    # Output directory for this threshold
    suffix = f'std_{str(std_multiplier).replace(".", "_")}'
    threshold_dir = os.path.join(OUTPUT_DIR, suffix)
    os.makedirs(threshold_dir, exist_ok=True)

    # Save hard cluster assignments
    assignments_df = umap_df.copy()
    assignments_df['Cluster'] = final_labels
    assignments_df.to_csv(os.path.join(threshold_dir, 'clustering_assignments.csv'), index=False)

    # === Save soft probabilities CSV ===
    prob_df = pd.DataFrame(probabilities, columns=[f'Cluster_{i}' for i in range(final_K)])
    prob_df.insert(0, 'Filename', filenames)
    prob_df.to_csv(os.path.join(threshold_dir, 'soft_cluster_probabilities.csv'), index=False)

    # === BIC curve plot ===
    plt.figure(figsize=(8, 5))
    plt.plot(K_RANGE, tied_bics, marker='o', label='Tied Covariance BIC')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{std_multiplier} std above min BIC')
    plt.scatter([K for K in K_RANGE if tied_bics[K - K_RANGE.start] <= threshold],
                [b for b in tied_bics if b <= threshold], color='red')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('BIC')
    plt.title(f'BIC vs. K (Tied Covariance) - {std_multiplier} std threshold')
    plt.legend()
    plt.savefig(os.path.join(threshold_dir, 'bic_curve.png'))
    plt.close()

    # === UMAP cluster plot ===
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster', palette='tab10', data=assignments_df, s=40)
    plt.title(f'UMAP Clusters (K={final_K}, {std_multiplier} std threshold)')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(threshold_dir, 'umap_clusters.png'))
    plt.close()

    # === Log file ===
    with open(os.path.join(threshold_dir, 'log.txt'), 'w') as f:
        f.write('==== Clustering Selection Log ====\n')
        f.write(f'Std threshold: {std_multiplier}x std\n')
        f.write(f'Input embedding files: {embedding_files}\n')
        f.write(f'K range tested: {list(K_RANGE)}\n')
        f.write(f'Min BIC (tied): {bic_min:.2f}\n')
        f.write(f'BIC std deviation: {bic_std:.2f}\n')
        f.write(f'BIC threshold: {threshold:.2f}\n')
        f.write(f'Candidate Ks within threshold: {candidate_Ks}\n')
        f.write(f'Chosen refined K (tied): {refined_K}\n')
        f.write(f'Full covariance BICs near K: {full_bics}\n')
        f.write(f'Final chosen K (full): {final_K}\n')

print(f"\nAll results saved to: {OUTPUT_DIR}")