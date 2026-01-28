#!/usr/bin/env python3
import os
import re
from glob import glob

# === CONFIGURATION ===
AUDIO_DIR = "../separation/american_crow_0.5_silence"
PREFIX    = "brachyrynchos"
EXT       = ".wav"

def find_used_group_ids(prefix, audio_dir):
    """Return a set of integers n for which files brachyrynchos_nnnnn_* exist."""
    used = set()
    pattern = re.compile(rf"{prefix}_(\d{{5}})_\d+{re.escape(EXT)}$")
    for f in os.listdir(audio_dir):
        m = pattern.match(f)
        if m:
            used.add(int(m.group(1)))
    return used

def rename_with_next_group(audio_dir, prefix, ext):
    # 1) find all already-used group IDs
    used_ids = find_used_group_ids(prefix, audio_dir)

    # 2) gather all wavs not matching the target pattern
    all_files = sorted(glob(os.path.join(audio_dir, f"*{ext}")))
    others = []
    target_pattern = re.compile(rf"{prefix}_\d{{5}}_\d+{re.escape(ext)}$")
    for path in all_files:
        fname = os.path.basename(path)
        if target_pattern.match(fname):
            continue  # keep these
        # require a single underscore before segment
        parts = fname.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].endswith(ext):
            print(f"⚠️ Skipping unrecognized: {fname}")
            continue
        base, seg_ext = parts
        segment = seg_ext[:-len(ext)]
        others.append((fname, base, segment))

    # group by original base
    from collections import defaultdict
    groups = defaultdict(list)
    for fname, base, segment in others:
        groups[base].append((fname, segment))

    # 3) for each original base, assign next free group ID and rename
    next_id = 1
    for base in sorted(groups.keys()):
        # find next available ID
        while next_id in used_ids:
            next_id += 1
        used_ids.add(next_id)

        for fname, segment in sorted(groups[base], key=lambda x: x[1]):
            new_base = f"{prefix}_{next_id:05d}"
            new_name = f"{new_base}_{segment}{ext}"
            old_path = os.path.join(audio_dir, fname)
            new_path = os.path.join(audio_dir, new_name)
            if os.path.exists(new_path):
                print(f"⚠️ Target exists, skipping {fname} → {new_name}")
            else:
                print(f"✔ Renaming {fname} → {new_name}")
                os.rename(old_path, new_path)

        next_id += 1

if __name__ == "__main__":
    rename_with_next_group(AUDIO_DIR, PREFIX, EXT)