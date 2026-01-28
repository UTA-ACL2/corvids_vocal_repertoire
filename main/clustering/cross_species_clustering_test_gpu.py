import os
import torch
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import pyro.poutine as poutine
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

pyro.enable_validation(True)
pyro.set_rng_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === CONFIGURATION ===
OUTPUT_DIR = 'results_pyro_gpu'
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_RANGE = range(4, 6)  # smaller range for speed; increase if you want
STD_THRESHOLDS = [0.5, 1.0, 1.5]

embedding_files = [
    "features/corax/crow_features_audiomae_corax.csv",
    "features/cornix/crow_features_audiomae_cornix.csv",
    "features/corone/crow_features_audiomae_corone.csv",
    "features/ossifragus/crow_features_audiomae_ossifragus.csv",
    "features/brachyrynchos/0.5_silence/crow_features_audiomae_brachyrynchos.csv"
]

# === Load and preprocess data ===
embedding_dfs = [pd.read_csv(f) for f in embedding_files]
full_df = pd.concat(embedding_dfs, ignore_index=True)
filenames = full_df.iloc[:, 0].tolist()
X_raw = full_df.iloc[:, 1:].values.astype(np.float32)

# Optional scaling
# scaler = StandardScaler()
# X_np = scaler.fit_transform(X_raw)
X_np = X_raw
X = torch.tensor(X_np).to(device)

N, D = X.shape

# === UMAP for visualization (CPU) ===
umap_model = UMAP(random_state=0)
umap_embeddings = umap_model.fit_transform(X_np)
umap_df = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
umap_df['Filename'] = filenames
umap_df.to_csv(os.path.join(OUTPUT_DIR, 'umap_embeddings.csv'), index=False)

# === Pyro GMM model and guide ===

def model_gmm(data, K):
    # data: NxD tensor
    N, D = data.shape
    with pyro.plate("components", K):
        # Mixture weights
        weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K, device=device)))
        # Means
        locs = pyro.sample("locs", dist.Normal(torch.zeros(K, D, device=device), 10 * torch.ones(K, D, device=device)).to_event(2))
        # Diagonal covariance (stddevs)
        scales = pyro.sample("scales", dist.LogNormal(torch.zeros(K, D, device=device), 0.5 * torch.ones(K, D, device=device)).to_event(2))
    
    with pyro.plate("data", N):
        assignment_probs = []
        for k in range(K):
            assignment_probs.append(dist.Normal(locs[k], scales[k]).log_prob(data).sum(-1).exp())
        probs = torch.stack(assignment_probs, dim=1)
        # Mixture weighting
        probs_weighted = probs * weights
        pyro.sample("obs", dist.Categorical(probs_weighted / probs_weighted.sum(-1, keepdim=True)), obs=None)


# We actually won't use the above model directly but will implement a Variational approach below

# Using mixture model with pyro.infer
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta

def train_gmm(X, K, max_steps=500):
    N, D = X.shape

    # Model definition for GMM with diagonal covariance using normal distributions per cluster
    def model(data):
        weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K, device=device)))
        locs = pyro.sample("locs", dist.Normal(torch.zeros(K, D, device=device), 10 * torch.ones(K, D, device=device)).to_event(2))
        scales = pyro.sample("scales", dist.LogNormal(torch.zeros(K, D, device=device), 0.5 * torch.ones(K, D, device=device)).to_event(2))
        with pyro.plate("data", N):
            assignment_log_probs = []
            for k in range(K):
                assignment_log_probs.append(dist.Normal(locs[k], scales[k]).log_prob(data).sum(-1))
            log_probs = torch.stack(assignment_log_probs, dim=1)  # NxK
            log_weighted = log_probs + torch.log(weights)
            pyro.sample("obs", dist.Categorical(logits=log_weighted), obs=None)

    guide = AutoDelta(model)
    svi = SVI(model, guide, pyro.optim.Adam({"lr": 0.05}), loss=Trace_ELBO())

    pyro.clear_param_store()
    for step in range(max_steps):
        loss = svi.step(X)
        if step % 50 == 0:
            print(f"Step {step} loss = {loss:.2f}")

    # Extract params
    params = {}
    for name, value in pyro.get_param_store().items():
        params[name] = value.detach()

    # Compute soft assignments manually
    weights = params['weights']
    locs = params['locs']
    scales = torch.exp(params['scales'])  # since lognormal
    assignment_log_probs = []
    for k in range(K):
        assignment_log_probs.append(dist.Normal(locs[k], scales[k]).log_prob(X).sum(-1))
    log_probs = torch.stack(assignment_log_probs, dim=1)  # NxK
    log_weighted = log_probs + torch.log(weights)
    log_weights_norm = torch.logsumexp(log_weighted, dim=1, keepdim=True)
    soft_assignments = (log_weighted - log_weights_norm).exp().cpu().numpy()

    # Hard assignments
    hard_assignments = soft_assignments.argmax(axis=1)

    # Approximate BIC: negative ELBO (loss) + penalty
    bic = loss + 0.5 * K * (2 * D + 1) * torch.log(torch.tensor(N, device=device))

    return bic.item(), hard_assignments, soft_assignments


# === Parallel BIC scan ===

def bic_scan(X, k):
    print(f"Fitting K={k}...")
    bic, _, _ = train_gmm(X, k)
    print(f"Done K={k}, BIC={bic}")
    return bic


print("Starting BIC scan...")
bics = Parallel(n_jobs=-1)(
    delayed(bic_scan)(X, k) for k in tqdm(K_RANGE)
)
bics = np.array(bics)

bic_min = bics.min()
bic_min_K = K_RANGE[np.argmin(bics)]
bic_std = bics.std()

with open(os.path.join(OUTPUT_DIR, 'min_bic_pyro.txt'), 'w') as f:
    f.write(f"Minimum BIC: {bic_min}\n")
    f.write(f"K at minimum BIC: {bic_min_K}\n")

print(f"Best K according to BIC: {bic_min_K}")

# === Final fit for best K ===
print(f"Fitting final model with K={bic_min_K}...")
final_bic, final_labels, final_soft_probs = train_gmm(X, bic_min_K)

# Save cluster assignments
assignments_df = umap_df.copy()
assignments_df['Cluster'] = final_labels
assignments_df.to_csv(os.path.join(OUTPUT_DIR, 'final_clustering_assignments.csv'), index=False)

# Save soft probabilities
prob_df = pd.DataFrame(final_soft_probs, columns=[f"Cluster_{i}" for i in range(bic_min_K)])
prob_df.insert(0, 'Filename', filenames)
prob_df.to_csv(os.path.join(OUTPUT_DIR, 'final_soft_cluster_probabilities.csv'), index=False)

# === Plot UMAP colored by clusters ===
plt.figure(figsize=(8, 5))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster', palette='tab10', data=assignments_df, s=40)
plt.title(f'UMAP Clusters (K={bic_min_K})')
plt.legend(title='Cluster')
plt.savefig(os.path.join(OUTPUT_DIR, 'umap_clusters_final.png'))
plt.close()

print(f"All done! Results saved to {OUTPUT_DIR}")