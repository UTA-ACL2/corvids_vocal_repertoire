import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
csv_path = "transition_matrices/soft_markov_transition_matrix_ossifragus.csv"        # Your input CSV file
output_png = "slides_transition_matrices/ossifragus"   # Output image path
remove_start_end = False               # Remove START/END states
normalize = False                     # Set True to row-normalize (optional)

# === LOAD MARKOV MATRIX ===
df = pd.read_csv(csv_path, index_col=0)

# Remove START/END rows/columns
if remove_start_end:
    df = df.drop(index="START", errors="ignore")
    df = df.drop(columns="END", errors="ignore")

# Optional normalization (row-wise)
if normalize:
    df = df.div(df.sum(axis=1), axis=0)

# === PLOT & SAVE HEATMAP ===
plt.figure(figsize=(14, 10))
sns.heatmap(df, annot=False, cmap="viridis", cbar=True)
plt.xlabel("Next Cluster (Predicted)")
plt.ylabel("Previous Cluster (Actual)")
plt.title("Ossifragus: Markov Matrix as Confusion Matrix")
plt.tight_layout()

# Save as PNG
plt.savefig(output_png, dpi=300)
print(f"✅ Saved confusion matrix as PNG to: {output_png}")