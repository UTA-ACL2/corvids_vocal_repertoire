import os
import pandas as pd
from textgrid import TextGrid

# === Directories ===
textgrid_dir = "annotated_sample_sequences"  # Directory with .TextGrid files
output_csv = "call_unit_events.csv"

rows = []

# === Iterate over each TextGrid file ===
for filename in sorted(os.listdir(textgrid_dir)):  # Sort filenames alphabetically
    if filename.endswith(".TextGrid"):
        base_name = os.path.splitext(filename)[0]  # e.g., brachyrynchos_0002
        tg_path = os.path.join(textgrid_dir, filename)
        
        try:
            tg = TextGrid.fromFile(tg_path)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue

        # Find the "Call Unit" tier (case insensitive)
        tier = None
        for t in tg.tiers:
            if t.name.lower() == "call unit":
                tier = t
                break
        
        if tier is None:
            print(f"No 'Call Unit' tier found in {filename}")
            continue

        # Add each interval as a crow event
        for interval in tier.intervals:
            if interval.maxTime - interval.minTime <= 0:
                continue  # skip zero-duration intervals
            rows.append({
                "filename": base_name + ".wav",
                "onset": interval.minTime,
                "offset": interval.maxTime,
                "event_label": "crow"
            })

# === Create DataFrame and sort ===
df = pd.DataFrame(rows)

# Sort by filename and then onset
df = df.sort_values(by=["filename", "onset"]).reset_index(drop=True)

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} events to {output_csv}")