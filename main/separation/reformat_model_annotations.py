import pandas as pd
import re

# Load the CSV
df = pd.read_csv("panns_annotations_base.csv")

# Remove the trailing _<number>.wav and replace with just .wav
df["filename"] = df["filename"].apply(
    lambda x: re.sub(r"_\d+(?=\.wav$)", "", x)
)

# Save the modified CSV
df.to_csv("panns_annotations_base_cleaned.csv", index=False)
print("✅ Saved cleaned annotations to panns_annotations_base_cleaned.csv")