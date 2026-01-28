import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CSV       = "panns_annotations_base.csv"
OUTPUT_CSV      = "model_annotations_merged.csv"
MAX_GAP_SECONDS = 1.0   # merge intervals if the silence between them ≤ 0.5s
# ────────────────────────────────────────────────────────────────────────────────

# 1) Load your annotations
df = pd.read_csv(INPUT_CSV)

# 2) Recover base filename (drop the _N suffix)
df['base_file'] = df['filename'].str.replace(r'_\d+\.wav$', '.wav', regex=True)

# 3) Group by base_file + event_label
merged_rows = []
for (base, label), group in df.groupby(['base_file','event_label']):
    intervals = group[['onset','offset']].sort_values('onset').values.tolist()
    cur_start, cur_end = intervals[0]
    for onset, offset in intervals[1:]:
        gap = onset - cur_end
        if gap <= MAX_GAP_SECONDS:
            # extend current interval
            cur_end = max(cur_end, offset)
        else:
            # push finished interval
            merged_rows.append({
                'filename': base,
                'onset':   round(cur_start,2),
                'offset':  round(cur_end,2),
                'event_label': label
            })
            # start a new one
            cur_start, cur_end = onset, offset
    # push the final interval for this group
    merged_rows.append({
        'filename': base,
        'onset':   round(cur_start,2),
        'offset':  round(cur_end,2),
        'event_label': label
    })

# 4) Save merged annotations
merged_df = pd.DataFrame(merged_rows)
merged_df.to_csv(OUTPUT_CSV, index=False)

print(f"Merged from {len(df)} lines down to {len(merged_df)} lines.")