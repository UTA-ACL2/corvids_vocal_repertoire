import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
import umap

# Load CSV
df = pd.read_csv("features/brachyrynchos/0.5_silence/crow_features_audiomae_resize_brachyrynchos.csv")

# Extract feature matrix and keep filenames
filenames = df["filename"] if "filename" in df.columns else None
X = df.drop(columns=["filename"], errors='ignore').values

# Fit BGMM
bgmm_no_pca = BayesianGaussianMixture(
    n_components=20,
    covariance_type='full',
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=0.1,
    max_iter=1000,
    random_state=42,
)
X_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="hamming", n_components=50).fit_transform(X)
bgmm_no_pca.fit(X_umap)

# Predict soft cluster membership probabilities
probs = bgmm_no_pca.predict_proba(X_umap)  # shape (n_samples, n_components)

# Create DataFrame with cluster probabilities, with columns like 'cluster_0', 'cluster_1', ...
probs_df = pd.DataFrame(probs, columns=[f'cluster_{i}' for i in range(probs.shape[1])])

# Add filenames column back if available
if filenames is not None:
    probs_df.insert(0, "filename", filenames)

# Save to new CSV
probs_df.to_csv("crow_cluster_probabilities_audiomae_repetition_resized_20.csv", index=False)