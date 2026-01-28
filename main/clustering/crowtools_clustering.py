import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import silhouette_score
import faiss
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Configuration ===
CSV_PATH = "aves_embeddings_base_bio.csv"       # input CSV: columns filename, embedding_*
OUTPUT_CSV = "crowtools_clusters.csv"
PLOT_PATH = "cluster_umap_scatter.png"
PCA_COMPONENTS = 75
USE_UMAP = True
UMAP_COMPONENTS = 50
MAX_CLUSTER_SIZE = 500
MERGE_THRESHOLD = 0.15
USE_GPU = faiss.get_num_gpus() > 0

# === Helper Functions ===
def reduce_dim_pca(emb, n_components):
    pca = PCA(n_components=min(n_components, emb.shape[1]), random_state=42)
    return pca.fit_transform(emb)

def reduce_dim_umap(emb, n_components):
    reducer = umap.UMAP(n_components=n_components, random_state=42, metric='cosine')
    return reducer.fit_transform(emb)

def normalize_embeddings(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return emb / norms

def hierarchical_split(emb, indices, max_size, level=0):
    if len(indices) <= max_size:
        return {'indices': indices, 'level': level}
    d = emb.shape[1]
    kmeans = faiss.Kmeans(d, 2, niter=20, verbose=False, seed=42, gpu=USE_GPU)
    subset = emb[indices]
    kmeans.train(subset)
    _, assignments = kmeans.index.search(subset, 1)
    clusters = {0: [], 1: []}
    for idx, c in zip(indices, assignments.flatten()):
        clusters[int(c)].append(idx)
    node = {'level': level, 'children': {}}
    for c, inds in clusters.items():
        node['children'][c] = hierarchical_split(emb, inds, max_size, level+1)
    return node

def collect_leaves(tree):
    if 'children' not in tree:
        return [tree]
    leaves = []
    for child in tree['children'].values():
        leaves.extend(collect_leaves(child))
    return leaves

def compute_center(norm_emb, idxs):
    ctr = np.mean(norm_emb[idxs], axis=0)
    return ctr / np.linalg.norm(ctr)

def merge_leaves(leaves, threshold):
    centers = [compute_center(norm_emb, leaf['indices']) for leaf in leaves]
    merged = set()
    new_leaves = []
    for i in tqdm(range(len(centers)), desc="Merging clusters"):
        for j in range(i+1, len(centers)):
            dist = 1 - np.dot(centers[i], centers[j])
            if dist < threshold and i not in merged and j not in merged:
                union = leaves[i]['indices'] + leaves[j]['indices']
                new_leaves.append({'indices': union})
                merged.update([i, j])
    for idx, leaf in enumerate(leaves):
        if idx not in merged:
            new_leaves.append({'indices': leaf['indices']})
    return new_leaves

# === Main Pipeline ===
df = pd.read_csv(CSV_PATH)
emb_cols = [c for c in df.columns if c.startswith('embedding_')]
file_groups = df.groupby('filename')[emb_cols]
file_emb = file_groups.mean().values.astype(np.float32)
filenames = file_groups.mean().index.tolist()

print("Reducing dimensions...")
reduced = reduce_dim_pca(file_emb, PCA_COMPONENTS)
if USE_UMAP:
    print("Applying UMAP...")
    reduced = reduce_dim_umap(reduced, UMAP_COMPONENTS)

print("Normalizing embeddings...")
norm_emb = normalize_embeddings(reduced)

d = norm_emb.shape[1]
print("Building FAISS index...")
if USE_GPU:
    print("Using GPU acceleration for FAISS")
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, d)
else:
    print("Using CPU for FAISS")
    index = faiss.IndexFlatIP(d)
index.add(norm_emb)

print("Running hierarchical clustering...")
all_idx = list(range(len(norm_emb)))

# Progress wrapper
def count_leaves(tree):
    if 'children' not in tree:
        return 1
    return sum(count_leaves(c) for c in tree['children'].values())

tree = hierarchical_split(norm_emb, all_idx, MAX_CLUSTER_SIZE)
leaves = collect_leaves(tree)

print("Merging small clusters...")
leaves = merge_leaves(leaves, MERGE_THRESHOLD)

print("Computing cluster centers...")
ecenters = np.vstack([compute_center(norm_emb, leaf['indices']) for leaf in tqdm(leaves, desc="Computing centers")])
num_clusters = ecenters.shape[0]

# Assign hard labels
labels = np.argmax(np.dot(norm_emb, ecenters.T), axis=1)

# === Diagnostics ===
# Cluster sizes
table = pd.Series(labels).value_counts().sort_index()
print("Cluster sizes (number of files per cluster):")
print(table)

# Silhouette score
eval_score = silhouette_score(norm_emb, labels)
print(f"Silhouette score: {eval_score:.4f}")

# === Visualization ===
print("Generating UMAP scatter plot of clusters...")
vis_2d = umap.UMAP(n_components=2, random_state=42, metric='cosine').fit_transform(norm_emb)
plt.figure(figsize=(8,6))
for cl in range(num_clusters):
    idxs = labels == cl
    plt.scatter(vis_2d[idxs,0], vis_2d[idxs,1], s=5, label=f"{cl}")
plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('UMAP projection colored by cluster')
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Saved cluster scatter plot to {PLOT_PATH}")

print("Computing cosine similarity to cluster centers...")
sim_matrix = np.dot(norm_emb, ecenters.T)

col_names = [f'cluster_{i}' for i in range(num_clusters)]
out_df = pd.DataFrame(sim_matrix, columns=col_names)
out_df.insert(0, 'filename', filenames)

print("Saving output CSV...")
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved per-file cluster similarity to {OUTPUT_CSV} (with {num_clusters} clusters)")