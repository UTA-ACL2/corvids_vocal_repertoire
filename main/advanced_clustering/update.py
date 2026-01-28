import pandas as pd

# Paths to your files
classified_path = "crowtools_classified.csv"
annotated_path = "annotated_labels_crowtools.csv"
output_path = "crowtools_classified_updated.csv"

# Load both CSVs
df_classified = pd.read_csv(classified_path)
df_annotated = pd.read_csv(annotated_path)

# Clean up boolean strings (in case of trailing commas or lowercase strings)
df_annotated['rattle'] = df_annotated['rattle'].astype(str).str.strip().str.lower() == 'true'
df_annotated['mob'] = df_annotated['mob'].astype(str).str.strip().str.lower() == 'true'

# Merge updated rattle/mob values from annotated into classified
df_updated = df_classified.merge(
    df_annotated[['filename', 'rattle', 'mob']],
    on='filename',
    how='left',
    suffixes=('', '_new')
)

# Replace old rattle/mob with annotated values if available
df_updated['rattle'] = df_updated['rattle_new'].combine_first(df_updated['rattle'])
df_updated['mob'] = df_updated['mob_new'].combine_first(df_updated['mob'])

# Drop temporary columns
df_updated = df_updated.drop(columns=['rattle_new', 'mob_new'])

# Save updated CSV
df_updated.to_csv(output_path, index=False)
print(f"[✔] Updated file saved to {output_path}")