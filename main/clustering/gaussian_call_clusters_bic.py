import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from multiprocessing import Pool
from tqdm import tqdm

# --- Load CSV ---
df = pd.read_csv("crow_features_audiomae_finetuned_repetition.csv")
filenames = df["filename"] if "filename" in df.columns else None
X = df.drop(columns=["filename"], errors='ignore').values.astype(np.float32)

# --- PCA Dimensionality Reduction ---
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)

# --- Helper function for multiprocessing ---
def fit_gmm(n_components):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='diag',  # Faster than 'full' for high-dimensional data
        max_iter=1000,
        random_state=42
    )
    gmm.fit(X_pca)
    bic = gmm.bic(X_pca)
    return (n_components, gmm, bic)

# --- Multiprocessing with tqdm progress bar (limit to 8 cores) ---
cluster_range = list(range(2, 31))

with Pool(processes=8) as pool:
    results = []
    for result in tqdm(pool.imap_unordered(fit_gmm, cluster_range), total=len(cluster_range), desc="Fitting GMMs"):
        results.append(result)

# --- Sort results by n_components ---
results.sort(key=lambda x: x[0])
bics = [r[2] for r in results]
models = [r[1] for r in results]

# --- Plot BIC scores ---
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, bics, marker='o')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('BIC Score')
plt.title('BIC vs. Number of Clusters (PCA 100 dims, diag covariance)')
plt.grid(True)
plt.tight_layout()
plt.savefig('bic_vs_clusters.png', dpi=300)
plt.close()

# --- Select the best model ---
best_idx = np.argmin(bics)
best_gmm = models[best_idx]
best_n_components = cluster_range[best_idx]
print(f"Best number of clusters (lowest BIC): {best_n_components}")

# --- Predict soft cluster probabilities ---
probs = best_gmm.predict_proba(X_pca)
probs_df = pd.DataFrame(probs, columns=[f'cluster_{i}' for i in range(probs.shape[1])])

if filenames is not None:
    probs_df.insert(0, "filename", filenames)

# --- Save CSV ---
output_csv = f"crow_cluster_probabilities_audiomae_finetune_repetition_{best_n_components}_pca100_diag.csv"
probs_df.to_csv(output_csv, index=False)

print(f"Saved cluster probabilities to {output_csv}")
print("Saved BIC comparison plot as 'bic_vs_clusters.png'")