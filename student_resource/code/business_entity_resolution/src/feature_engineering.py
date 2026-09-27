"""
Feature Engineering for Business Entity Resolution Matching Model.

Takes candidate pairs from blocking and computes pairwise similarity features.
Supports training, validation, and test modes with streaming disk writes
and memory-optimized data structures.

Run from student_resource/ root:
    python code\\business_entity_resolution\\src\\feature_engineering.py --mode train --sample-s1 100000
    python code\\business_entity_resolution\\src\\feature_engineering.py --mode val --sample-s1 25000
"""

import os
import sys
import re
import math
import argparse
import time
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import jellyfish
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import blocking

# Paths
S1_TRAIN_PATH = "dataset/train/train_source1.tsv"
S2_TRAIN_PATH = "dataset/train/train_source2.tsv"
S3_TRAIN_PATH = "dataset/train/train_source3.tsv"
GT_PATH = "dataset/train/train_ground_truth.tsv"

S1_TEST_PATH = "dataset/test/test_source1.tsv"
S2_TEST_PATH = "dataset/test/test_source2.tsv"
S3_TEST_PATH = "dataset/test/test_source3.tsv"

SRC_DIR = os.path.dirname(__file__)
TRAIN_IDS_FILE = os.path.join(SRC_DIR, "split_train_s1_ids.txt")
VAL_IDS_FILE = os.path.join(SRC_DIR, "split_val_s1_ids.txt")

TRAIN_CANDIDATES_FILE = os.path.join(SRC_DIR, "train_candidate_pairs.tsv")
TEST_CANDIDATES_FILE = os.path.join(SRC_DIR, "test_candidate_pairs.tsv")

OUTPUT_DIR = SRC_DIR

FEATURE_COLS = [
    'name_levenshtein', 'name_partial_ratio', 'name_token_sort_ratio',
    'name_token_set_ratio', 'name_jaro_winkler', 'name_jaccard', 'name_len_ratio',
    'addr_levenshtein', 'addr_token_sort_ratio', 'addr_token_overlap',
    'addr_len_ratio', 'addr_number_match',
    'country_exact', 'combined_jaccard', 'num_shared_blocking_keys'
]


def load_split_ids(path):
    """Load S1 IDs from a split file."""
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def load_entities(s1_path, s2_path, s3_path, needed_s1_ids=None, needed_pool_ids=None):
    """
    Load source entities into compact tuple lookups:
    entity_id -> (business_name, business_address, country)
    """
    print("Loading source entities (memory-optimized)...")
    t0 = time.time()

    s1_lookup = {}
    for chunk in pd.read_csv(s1_path, sep='\t', dtype=str, encoding='utf-8',
                             usecols=['entity_id', 'business_name', 'business_address', 'country'],
                             chunksize=100000):
        chunk = chunk.fillna('')
        for row in chunk.itertuples(index=False):
            eid = row.entity_id
            if needed_s1_ids is None or eid in needed_s1_ids:
                s1_lookup[eid] = (str(row.business_name), str(row.business_address), str(row.country))

    pool_lookup = {}
    for src_path in [s2_path, s3_path]:
        for chunk in pd.read_csv(src_path, sep='\t', dtype=str, encoding='utf-8',
                                 usecols=['entity_id', 'business_name', 'business_address', 'country'],
                                 chunksize=100000):
            chunk = chunk.fillna('')
            for row in chunk.itertuples(index=False):
                eid = row.entity_id
                if needed_pool_ids is None or eid in needed_pool_ids:
                    pool_lookup[eid] = (str(row.business_name), str(row.business_address), str(row.country))

    print(f"Loaded {len(s1_lookup):,} S1 and {len(pool_lookup):,} S2/S3 entities in {time.time()-t0:.1f}s")
    gc.collect()
    return s1_lookup, pool_lookup


def load_ground_truth():
    """Load ground truth as {s1_id: set(matched_ids)}."""
    gt = pd.read_csv(GT_PATH, sep='\t', dtype=str, encoding='utf-8')
    gt_map = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matched = str(row['matched_entity_ids'])
        if matched and matched != 'nan' and matched.strip():
            gt_map[s1_id] = set(m.strip() for m in matched.split(',') if m.strip())
        else:
            gt_map[s1_id] = set()
    return gt_map


def load_candidate_pairs(candidate_file, s1_filter=None, sample_s1=None, seed=42):
    """Load candidate pairs from TSV. Returns {s1_id: [candidate_ids]}."""
    pairs = {}
    with open(candidate_file, 'r', encoding='utf-8') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            s1_id = parts[0]
            if s1_filter is not None and s1_id not in s1_filter:
                continue
            cand_str = parts[1]
            if cand_str.strip():
                pairs[s1_id] = [c.strip() for c in cand_str.split(',') if c.strip()]
            else:
                pairs[s1_id] = []

    # If sampling requested and pairs exceed sample size
    if sample_s1 and sample_s1 > 0 and len(pairs) > sample_s1:
        rng = np.random.RandomState(seed)
        sampled_keys = rng.choice(list(pairs.keys()), size=sample_s1, replace=False)
        pairs = {k: pairs[k] for k in sampled_keys}

    return pairs


# ============= Feature Functions =============

def clean(s):
    """Lowercase, strip punctuation."""
    if not isinstance(s, str) or s.lower() == 'nan':
        return ''
    return re.sub(r'[^\w\s]', ' ', s.lower()).strip()


def extract_numbers(s):
    """Extract all digit sequences from a string."""
    return set(re.findall(r'\b\d+\b', s))


def token_set(s):
    """Return a set of non-empty tokens."""
    return set(s.split()) if s else set()


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def length_ratio(a, b):
    """Ratio of shorter to longer string length."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def compute_pair_features(s1_rec, cand_rec, s1_keys=None):
    """
    Compute pairwise features between an S1 record and a candidate record.
    Accepts either dicts or tuples: (business_name, business_address, country)
    """
    if isinstance(s1_rec, dict):
        name_a, addr_a, country_a = s1_rec['business_name'], s1_rec['business_address'], s1_rec['country']
    else:
        name_a, addr_a, country_a = s1_rec

    if isinstance(cand_rec, dict):
        name_b, addr_b, country_b = cand_rec['business_name'], cand_rec['business_address'], cand_rec['country']
    else:
        name_b, addr_b, country_b = cand_rec

    name_a_clean = clean(name_a)
    name_b_clean = clean(name_b)
    addr_a_clean = clean(addr_a)
    addr_b_clean = clean(addr_b)

    name_tokens_a = token_set(name_a_clean)
    name_tokens_b = token_set(name_b_clean)
    addr_tokens_a = token_set(addr_a_clean)
    addr_tokens_b = token_set(addr_b_clean)

    nums_a = extract_numbers(addr_a_clean)
    nums_b = extract_numbers(addr_b_clean)

    features = {}

    # --- Name features ---
    features['name_levenshtein'] = fuzz.ratio(name_a_clean, name_b_clean) / 100.0
    features['name_partial_ratio'] = fuzz.partial_ratio(name_a_clean, name_b_clean) / 100.0
    features['name_token_sort_ratio'] = fuzz.token_sort_ratio(name_a_clean, name_b_clean) / 100.0
    features['name_token_set_ratio'] = fuzz.token_set_ratio(name_a_clean, name_b_clean) / 100.0

    try:
        features['name_jaro_winkler'] = jellyfish.jaro_winkler_similarity(name_a_clean, name_b_clean)
    except Exception:
        features['name_jaro_winkler'] = 0.0

    features['name_jaccard'] = jaccard(name_tokens_a, name_tokens_b)
    features['name_len_ratio'] = length_ratio(name_a_clean, name_b_clean)

    # --- Address features ---
    features['addr_levenshtein'] = fuzz.ratio(addr_a_clean, addr_b_clean) / 100.0
    features['addr_token_sort_ratio'] = fuzz.token_sort_ratio(addr_a_clean, addr_b_clean) / 100.0
    features['addr_token_overlap'] = jaccard(addr_tokens_a, addr_tokens_b)
    features['addr_len_ratio'] = length_ratio(addr_a_clean, addr_b_clean)

    # Address number match
    if nums_a or nums_b:
        if nums_a and nums_b:
            features['addr_number_match'] = len(nums_a & nums_b) / max(len(nums_a), len(nums_b))
        else:
            features['addr_number_match'] = 0.0
    else:
        features['addr_number_match'] = 0.5

    # --- Cross features ---
    features['country_exact'] = 1.0 if clean(country_a) == clean(country_b) else 0.0

    all_tokens_a = name_tokens_a | addr_tokens_a
    all_tokens_b = name_tokens_b | addr_tokens_b
    features['combined_jaccard'] = jaccard(all_tokens_a, all_tokens_b)

    # Number of shared blocking keys
    if s1_keys is None:
        s1_keys = set(blocking.get_blocking_keys(name_a, addr_a, country_a))
    keys_b = set(blocking.get_blocking_keys(name_b, addr_b, country_b))
    features['num_shared_blocking_keys'] = float(len(s1_keys & keys_b))

    return features


def stream_features_to_tsv(candidate_pairs, s1_lookup, pool_lookup, out_file, gt_map=None):
    """
    Stream features directly to disk in TSV format.
    Zero RAM accumulation, periodic disk flush.
    """
    total_pairs = sum(len(cands) for cands in candidate_pairs.values())
    print(f"Streaming features for {total_pairs:,} candidate pairs directly to {out_file}...")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    has_labels = gt_map is not None

    cols = ['s1_id', 'candidate_id']
    if has_labels:
        cols.append('label')
    cols.extend(FEATURE_COLS)

    n_written = 0
    n_pos = 0
    n_neg = 0
    buffer = []
    BUFFER_SIZE = 10000

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\t'.join(cols) + '\n')

        with tqdm(total=total_pairs, desc="Feature engineering", unit="pairs") as pbar:
            for s1_id, cand_ids in candidate_pairs.items():
                s1_rec = s1_lookup.get(s1_id)
                if s1_rec is None:
                    pbar.update(len(cand_ids))
                    continue

                true_matches = gt_map.get(s1_id, set()) if has_labels else None

                # Precompute S1 blocking keys once for this S1 entity
                if isinstance(s1_rec, dict):
                    s1_keys = set(blocking.get_blocking_keys(s1_rec['business_name'], s1_rec['business_address'], s1_rec['country']))
                else:
                    s1_keys = set(blocking.get_blocking_keys(s1_rec[0], s1_rec[1], s1_rec[2]))

                for cid in cand_ids:
                    cand_rec = pool_lookup.get(cid)
                    if cand_rec is None:
                        pbar.update(1)
                        continue

                    feats = compute_pair_features(s1_rec, cand_rec, s1_keys=s1_keys)

                    row_parts = [s1_id, cid]
                    if has_labels:
                        lbl = 1 if cid in true_matches else 0
                        row_parts.append(str(lbl))
                        if lbl == 1:
                            n_pos += 1
                        else:
                            n_neg += 1

                    for col in FEATURE_COLS:
                        val = feats.get(col, 0.0)
                        row_parts.append(f"{val:.4f}" if isinstance(val, float) else str(val))

                    buffer.append('\t'.join(row_parts) + '\n')
                    n_written += 1
                    pbar.update(1)

                    if len(buffer) >= BUFFER_SIZE:
                        f.writelines(buffer)
                        f.flush()
                        buffer.clear()

            if buffer:
                f.writelines(buffer)
                f.flush()
                buffer.clear()

    print(f"\nWritten {n_written:,} feature rows to {out_file}")
    if has_labels and n_written > 0:
        print(f"Positive pairs: {n_pos:,} ({n_pos/n_written:.1%})")
        print(f"Negative pairs: {n_neg:,} ({n_neg/n_written:.1%})")
        print(f"Pos/Neg ratio:  1:{n_neg/max(1,n_pos):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for entity matching.")
    parser.add_argument("--mode", required=True, choices=["train", "val", "all_train", "test"],
                        help="Which split to process: train, val, all_train, or test")
    parser.add_argument("--sample-s1", type=int, default=100000,
                        help="Sample S1 entities (default: 100000 for fast high-quality training; set 0 for all)")
    parser.add_argument("--candidate-file", default=None,
                        help="Override candidate pairs file path")
    parser.add_argument("--output", default=None,
                        help="Override output file path")
    args = parser.parse_args()

    t_start = time.time()

    if args.mode == "test":
        cand_file = args.candidate_file or TEST_CANDIDATES_FILE
        out_file = args.output or os.path.join(OUTPUT_DIR, "test_features.tsv")
        s1_filter = None
        gt_map = None
        s1_path, s2_path, s3_path = S1_TEST_PATH, S2_TEST_PATH, S3_TEST_PATH
    else:
        cand_file = args.candidate_file or TRAIN_CANDIDATES_FILE
        gt_map = load_ground_truth()
        s1_path, s2_path, s3_path = S1_TRAIN_PATH, S2_TRAIN_PATH, S3_TRAIN_PATH

        if args.mode == "train":
            s1_filter = load_split_ids(TRAIN_IDS_FILE)
            out_file = args.output or os.path.join(OUTPUT_DIR, "train_features.tsv")
        elif args.mode == "val":
            s1_filter = load_split_ids(VAL_IDS_FILE)
            out_file = args.output or os.path.join(OUTPUT_DIR, "val_features.tsv")
            # For validation, if sample-s1 is the default 100000, use 25000 for snappy validation
            if args.sample_s1 == 100000:
                args.sample_s1 = 25000
        else:
            s1_filter = None
            out_file = args.output or os.path.join(OUTPUT_DIR, "all_train_features.tsv")

    sample_val = args.sample_s1 if args.sample_s1 > 0 else None

    print(f"\nMode: {args.mode}")
    print(f"Candidate file: {cand_file}")
    print(f"Output file: {out_file}")
    print(f"Sampling limit: {sample_val or 'ALL'}")

    # Load candidate pairs
    candidate_pairs = load_candidate_pairs(cand_file, s1_filter=s1_filter, sample_s1=sample_val)
    print(f"Loaded candidate pairs for {len(candidate_pairs):,} S1 entities")

    if len(candidate_pairs) == 0:
        print("ERROR: No candidate pairs loaded. Check your candidate file.")
        sys.exit(1)

    # Collect needed entity IDs to minimize memory footprint
    needed_s1 = set(candidate_pairs.keys())
    needed_pool = set()
    for cands in candidate_pairs.values():
        needed_pool.update(cands)

    print(f"Filter active: {len(needed_s1):,} S1 IDs and {len(needed_pool):,} S2/S3 IDs to load")

    # Load only necessary entities
    s1_lookup, pool_lookup = load_entities(s1_path, s2_path, s3_path,
                                           needed_s1_ids=needed_s1,
                                           needed_pool_ids=needed_pool)

    # Stream features directly to disk
    stream_features_to_tsv(candidate_pairs, s1_lookup, pool_lookup, out_file, gt_map=gt_map)

    print(f"Completed in {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
