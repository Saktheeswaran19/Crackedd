"""
Stratified train/validation split of Source-1 entities.

Stratified by:
  (a) country (US / India / etc.)
  (b) singleton vs. has-matches

Run from student_resource/ root:
    python code\\business_entity_resolution\\src\\make_split.py
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

SEED = 42
VAL_RATIO = 0.2

S1_PATH = "dataset/train/train_source1.tsv"
GT_PATH = "dataset/train/train_ground_truth.tsv"
OUT_DIR = os.path.join(os.path.dirname(__file__))
TRAIN_IDS_FILE = os.path.join(OUT_DIR, "split_train_s1_ids.txt")
VAL_IDS_FILE = os.path.join(OUT_DIR, "split_val_s1_ids.txt")


def main():
    print("Loading Source-1 and Ground Truth...")
    s1 = pd.read_csv(S1_PATH, sep='\t', dtype=str, encoding='utf-8',
                     usecols=['entity_id', 'country'])
    gt = pd.read_csv(GT_PATH, sep='\t', dtype=str, encoding='utf-8')

    # Determine singleton vs. matched
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    gt['has_matches'] = gt['matched_entity_ids'].apply(
        lambda x: 1 if x.strip() != '' else 0
    )

    # Merge to get country + has_matches per S1 entity
    merged = s1.merge(
        gt[['source1_entity_id', 'has_matches']],
        left_on='entity_id', right_on='source1_entity_id', how='left'
    )
    merged['has_matches'] = merged['has_matches'].fillna(0).astype(int)

    # Create stratification key: country + singleton/matched
    merged['strat_key'] = merged['country'].fillna('UNKNOWN') + '_' + merged['has_matches'].astype(str)

    print(f"\nTotal S1 entities: {len(merged):,}")
    print(f"\nStratification key distribution:")
    print(merged['strat_key'].value_counts().to_string())

    # Stratified split
    rng = np.random.RandomState(SEED)
    train_ids = []
    val_ids = []

    for key, group in merged.groupby('strat_key'):
        ids = group['entity_id'].values.copy()
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * VAL_RATIO))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    print(f"\nSplit result:")
    print(f"  Train: {len(train_ids):,} ({len(train_ids)/len(merged):.1%})")
    print(f"  Val:   {len(val_ids):,} ({len(val_ids)/len(merged):.1%})")

    # Verify stratification held
    train_set = set(train_ids)
    val_set = set(val_ids)
    assert len(train_set & val_set) == 0, "Overlap between train and val!"
    assert len(train_set) + len(val_set) == len(merged), "IDs lost in split!"

    # Print per-stratum breakdown
    print("\nPer-stratum breakdown:")
    for key, group in merged.groupby('strat_key'):
        ids_in_group = set(group['entity_id'].values)
        n_train = len(ids_in_group & train_set)
        n_val = len(ids_in_group & val_set)
        print(f"  {key}: train={n_train:,} val={n_val:,} "
              f"(val ratio={n_val/(n_train+n_val):.1%})")

    # Save
    with open(TRAIN_IDS_FILE, 'w', encoding='utf-8') as f:
        for sid in sorted(train_ids):
            f.write(sid + '\n')

    with open(VAL_IDS_FILE, 'w', encoding='utf-8') as f:
        for sid in sorted(val_ids):
            f.write(sid + '\n')

    print(f"\nSaved train IDs → {TRAIN_IDS_FILE}")
    print(f"Saved val IDs   → {VAL_IDS_FILE}")


if __name__ == '__main__':
    main()
