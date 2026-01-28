import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm

# === CONFIGURATION ===
OUTPUT_DIR = 'results_bic_diagonal_with_std'
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_RANGE = range(2, 101)
RANDOM_SEED = 0
STD_THRESHOLDS = [0.5, 1.0]

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

filenames = full_df.iloc[:, 0].tolist()
X_raw = full_df.iloc[:, 1:].values.astype(np.float32)

# === Standardize Features ===
X = StandardScaler().fit_transform(X_raw)

# === UMAP FOR VISUALIZATION ===
umap = UMAP(random_state=RANDOM_SEED)
umap_embeddings = umap.fit_transform(X)
umap_df = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
umap_df['Filename'] = filenames
umap_df.to_csv(os.path.join(OUTPUT_DIR, 'umap_embeddings.csv'), index=False)

# === Helper: BIC for Diagonal Covariance ===
def bic_for_k(X, K, covariance_type='diag', random_state=0):
    gm = GaussianMixture(
        n_components=K,
        covariance_type=covariance_type,
        random_state=random_state,
        reg_covar=1e-4,
        max_iter=500
    )
    gm.fit(X)
    return gm.bic(X)

# === RUN BIC TEST ACROSS K ===
print("Running diagonal-covariance BIC tests (parallel)...")
diagonal_bics = Parallel(n_jobs=-1)(
    delayed(bic_for_k)(X, K, covariance_type='diag', random_state=RANDOM_SEED) for K in tqdm(K_RANGE)
)
diagonal_bics = np.array(diagonal_bics)

# === Save BIC Results ===
bic_results_df = pd.DataFrame({
    'K': list(K_RANGE),
    'BIC': diagonal_bics
})
bic_results_df.to_csv(os.path.join(OUTPUT_DIR, 'bic_results.csv'), index=False)

# === Analyze BIC Minimum and Std Thresholds ===
bic_min = diagonal_bics.min()
bic_min_K = K_RANGE[np.argmin(diagonal_bics)]
bic_std = diagonal_bics.std()

threshold_Ks = {}
for std_mult in STD_THRESHOLDS:
    threshold = bic_min + bic_std * std_mult
    Ks_in_threshold = [K for K, bic in zip(K_RANGE, diagonal_bics) if bic <= threshold]
    threshold_Ks[std_mult] = Ks_in_threshold

# === Plot BIC Curve ===
plt.figure(figsize=(8, 5))
plt.plot(K_RANGE, diagonal_bics, marker='o', label='Diagonal Covariance BIC')

for std_mult, Ks in threshold_Ks.items():
    threshold = bic_min + bic_std * std_mult
    plt.axhline(y=threshold, linestyle='--', label=f'{std_mult} std above min BIC', alpha=0.7)
    bic_subset = [diagonal_bics[K - K_RANGE.start] for K in Ks]
    plt.scatter(Ks, bic_subset, color='red', label=f'Ks within {std_mult} std', zorder=5)

plt.xlabel('Number of Clusters (K)')
plt.ylabel('BIC')
plt.title('BIC vs. K (Diagonal Covariance)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bic_curve.png'), dpi=300)
plt.close()

# === Summary Log ===
with open(os.path.join(OUTPUT_DIR, 'bic_summary.txt'), 'w') as f:
    f.write('==== BIC Summary (Diagonal Covariance) ====\n')
    f.write(f'Minimum BIC: {bic_min:.4f}\n')
    f.write(f'K at minimum BIC: {bic_min_K}\n')
    f.write(f'BIC std deviation: {bic_std:.4f}\n\n')
    for std_mult, Ks in threshold_Ks.items():
        f.write(f'Threshold {std_mult} std above min BIC:\n')
        f.write(f'  BIC threshold value: {bic_min + bic_std * std_mult:.4f}\n')
        f.write(f'  Ks within threshold: {Ks}\n\n')

print(f"\nAll results saved to: {OUTPUT_DIR}")