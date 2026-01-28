import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def combine_embedding_csvs(csv_files, metadata_cols=["filename", "log_duration"]):
    """
    Given a list of CSV paths each containing embeddings + metadata_cols,
    returns a DataFrame combining all embeddings normalized and concatenated,
    preserving metadata columns from the first CSV.

    Assumes all CSVs have the same rows in the same order and identical metadata.
    """
    dfs = [pd.read_csv(f) for f in csv_files]

    # Check all filenames match
    for i in range(1, len(dfs)):
        if not all(dfs[0]["filename"] == dfs[i]["filename"]):
            raise ValueError(f"Filename mismatch between {csv_files[0]} and {csv_files[i]}")

    # Extract metadata from first df
    metadata = dfs[0][metadata_cols]

    embedding_arrays = []
    embedding_col_names = []

    for i, df in enumerate(dfs):
        # Extract embedding columns (drop metadata)
        emb_df = df.drop(columns=metadata_cols, errors='ignore')

        # Normalize embeddings for this CSV
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb_df.values)

        embedding_arrays.append(emb_scaled)

        # Generate prefixed column names for embeddings
        prefix = f"emb{i+1}"
        cols = [f"{prefix}_feat_{j+1}" for j in range(emb_scaled.shape[1])]
        embedding_col_names.extend(cols)

    # Concatenate all normalized embeddings horizontally
    combined_embeddings = np.hstack(embedding_arrays)

    # Build final DataFrame with metadata + combined embeddings
    combined_df = pd.DataFrame(combined_embeddings, columns=embedding_col_names)
    final_df = pd.concat([metadata.reset_index(drop=True), combined_df], axis=1)

    return final_df

# === Example usage ===
csv_list = [
    "AudioMAE_finetuned_embeddings.csv",
    "../counting/gaussian_repetition_counts.csv"
]

combined_df = combine_embedding_csvs(csv_list)
combined_df.to_csv("crow_features_audiomae_finetuned_repetition.csv", index=False)
print(f"Saved combined super embeddings CSV with shape {combined_df.shape}")