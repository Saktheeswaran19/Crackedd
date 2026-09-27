"""
EDA on training data for the Amazon Entity Resolution challenge.
Run from student_resource/ root:
    python code\business_entity_resolution\src\eda.py
"""
import pandas as pd
from rapidfuzz import fuzz

pd.set_option("display.max_rows", 50)

STR_COLS = ["entity_id", "business_name", "business_address", "country"]

def load(path):
    # dtype=str is non-negotiable — IDs and postal-code-like strings
    # must never be inferred as int/float.
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8", keep_default_na=False)

print("Loading training files (this may take a minute at ~200-500MB each)...")
s1 = load("dataset/train/train_source1.tsv")
s2 = load("dataset/train/train_source2.tsv")
s3 = load("dataset/train/train_source3.tsv")
gt = load("dataset/train/train_ground_truth.tsv")

print("\n=== Row counts ===")
print(f"Source1: {len(s1):,} | Source2: {len(s2):,} | Source3: {len(s3):,} | GT: {len(gt):,}")

print("\n=== Columns ===")
print("s1:", list(s1.columns))
print("s2:", list(s2.columns))
print("s3:", list(s3.columns))
print("gt:", list(gt.columns))

print("\n=== Country distribution (Source1) ===")
print(s1["country"].value_counts())

print("\n=== Country distribution (Source2) ===")
print(s2["country"].value_counts())

print("\n=== Country distribution (Source3) ===")
print(s3["country"].value_counts())

# ground truth: match-count distribution
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
gt["n_matches"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if x.strip() == "" else len(x.split(","))
)
print("\n=== Match count distribution (ground truth) ===")
print(gt["n_matches"].value_counts().sort_index())
n_singletons = (gt["n_matches"] == 0).sum()
print(f"\nSingletons: {n_singletons:,} / {len(gt):,} ({n_singletons/len(gt):.1%})")

print("\n=== Name length stats (Source1) ===")
print(s1["business_name"].str.len().describe())

print("\n=== Address length stats (Source1) ===")
print(s1["business_address"].str.len().describe())

print("\n=== Sample rows ===")
print(s1.sample(5, random_state=42))
print(gt.sample(5, random_state=42))

# quick peek: does country appear consistently for matched pairs?
print("\n=== Checking country consistency on a few known matches ===")
sample_matches = gt[gt["n_matches"] > 0].sample(min(5, (gt["n_matches"] > 0).sum()), random_state=42)
for _, row in sample_matches.iterrows():
    s1_id = row["source1_entity_id"]
    s1_row = s1[s1["entity_id"] == s1_id]
    print(f"\n{s1_id}: {s1_row['business_name'].values} | {s1_row['country'].values}")
    for mid in row["matched_entity_ids"].split(","):
        src = s2 if mid.startswith("S2-") else s3
        m_row = src[src["entity_id"] == mid]
        print(f"  -> {mid}: {m_row['business_name'].values} | {m_row['country'].values}")



# quantify how many true matches have very low name similarity
gt_matched = gt[gt["n_matches"] > 0].sample(min(2000, (gt["n_matches"] > 0).sum()), random_state=42)
low_sim_count = 0
total_checked = 0

s1_lookup = s1.set_index("entity_id")
s2_lookup = s2.set_index("entity_id")
s3_lookup = s3.set_index("entity_id")

for _, row in gt_matched.iterrows():
    s1_name = s1_lookup.loc[row["source1_entity_id"], "business_name"]
    for mid in row["matched_entity_ids"].split(","):
        lookup = s2_lookup if mid.startswith("S2-") else s3_lookup
        if mid not in lookup.index:
            continue
        m_name = lookup.loc[mid, "business_name"]
        sim = fuzz.token_sort_ratio(s1_name, m_name)
        total_checked += 1
        if sim < 40:  # very low name similarity threshold
            low_sim_count += 1

print(f"\nLow name-similarity true matches: {low_sim_count}/{total_checked} ({low_sim_count/total_checked:.1%})")