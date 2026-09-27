"""
Candidate Generation (Blocking) Module for Business Entity Resolution.
Combines an Inverted Index (lexical/symbolic keys with IDF-weighted ranking)
and Approximate Nearest Neighbors (TF-IDF char n-grams + TruncatedSVD + FAISS).

Follows entity-resolution skill constraints:
- Language and country agnostic: treats `country` as an open string label.
- Low memory footprint: partitions by country, uses sparse TF-IDF + compact SVD embeddings.
- Outputs candidate lists ready for feature engineering and downstream matching.
"""

import math
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss

# Standard legal business suffix tokens across jurisdictions (US, India, France, etc.)
LEGAL_TOKENS = {
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'pvt', 'private', 'limited',
    'co', 'company', 'services', 'partners', 'associates', 'group', 'holdings',
    'enterprises', 'center', 'the', 'and', 'of', '&', 'plc', 'sa', 'sarl', 'gmbh',
    'technologies', 'solutions', 'management', 'international', 'sas', 'eurl'
}

# Common address stop tokens across languages/geographies
COMMON_ADDR = {
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'drive', 'dr', 'lane', 'ln',
    'floor', 'fl', 'suite', 'ste', 'near', 'opp', 'behind', 'beside', 'plot',
    'building', 'bldg', 'block', 'blk', 'house', 'hno', 'door', 'dno', 'sector', 'sec',
    'city', 'state', 'nagar', 'colony', 'rue', 'boulevard', 'bd', 'chemin', 'null', 'nan'
}


def clean_text(s: str) -> str:
    """Lowercase and normalize whitespace/punctuation."""
    if not isinstance(s, str) or s.lower() == 'nan':
        return ""
    return re.sub(r'[^\w\s]', ' ', s.lower()).strip()


def get_blocking_keys(name: str, address: str, country: str):
    """
    Generate language-agnostic blocking keys for inverted indexing.
    Treats country as an open string label (no hardcoded country lists).
    Includes address-only compound keys to bridge cross-script name transliterations.
    """
    keys = []
    c = country.strip().upper() if isinstance(country, str) else "UNKNOWN"

    name_clean = clean_text(name)
    name_tokens = [t for t in name_clean.split() if t]
    sig_name = [t for t in name_tokens if t not in LEGAL_TOKENS and len(t) >= 2]

    addr_clean = clean_text(address)
    addr_tokens = [t for t in addr_clean.split() if t not in COMMON_ADDR and len(t) >= 3]
    nums = [t for t in re.findall(r'\b\d+\b', addr_clean) if len(t) >= 1]

    # 1. Exact sorted significant name tokens & prefixes
    if sig_name:
        keys.append(f"{c}|exact|{' '.join(sorted(sig_name))}")
        keys.append(f"{c}|name_pfx|{sig_name[0][:4]}")
        if len(sig_name) >= 2:
            keys.append(f"{c}|sig2|{sig_name[0]}_{sig_name[1]}")
        for t in sig_name:
            if len(t) >= 4:
                keys.append(f"{c}|token|{t}")
    elif name_clean:
        keys.append(f"{c}|exact|{name_clean}")

    # 2. Address number + first significant name token (high precision across noise)
    if nums and sig_name:
        keys.append(f"{c}|num_name|{nums[0]}_{sig_name[0]}")
        if len(nums) >= 2:
            keys.append(f"{c}|2num_name|{nums[0]}_{nums[1]}_{sig_name[0][:3]}")

    # 3. Address number + significant address token
    if nums and addr_tokens:
        keys.append(f"{c}|num_addr|{nums[0]}_{addr_tokens[0]}")
        if len(addr_tokens) >= 2:
            keys.append(f"{c}|addr2|{addr_tokens[0]}_{addr_tokens[1]}")

    # 4. Address-only compound keys (bridges cross-script name mismatches, e.g. Indic scripts)
    if len(addr_tokens) >= 2:
        for j in range(min(4, len(addr_tokens) - 1)):
            keys.append(f"{c}|addr_bigram|{addr_tokens[j]}_{addr_tokens[j+1]}")
    for t in addr_tokens:
        if len(t) >= 6:
            keys.append(f"{c}|addr_tok|{t}")

    return keys


def build_inverted_index(pool_df: pd.DataFrame, max_posting: int = 120):
    """
    Build inverted index mapping blocking key -> list of candidate entity IDs,
    along with IDF weights per key for damped candidate scoring.
    """
    index = defaultdict(list)
    for _, row in pool_df.iterrows():
        p_id = row['entity_id']
        keys = get_blocking_keys(row['business_name'], row['business_address'], row['country'])
        for k in keys:
            index[k].append(p_id)

    n_pool = len(pool_df)
    idf = {}
    for k, post in list(index.items()):
        if max_posting is not None and len(post) > max_posting:
            index[k] = post[:max_posting]
        idf[k] = math.log(1.0 + n_pool / (len(index[k]) + 1))

    return index, idf


def query_inverted_index(query_df: pd.DataFrame, index: dict, idf: dict, top_k: int = 12):
    """
    Query inverted index for each record in query_df using smooth IDF weighting.
    Returns {query_id: [candidate_ids]} sorted by accumulated IDF score.
    """
    candidates = {}
    for _, row in query_df.iterrows():
        qid = row['entity_id']
        keys = get_blocking_keys(row['business_name'], row['business_address'], row['country'])
        cand_scores = defaultdict(float)
        for k in keys:
            if k in index:
                k_weight = idf.get(k, 1.0)
                for cid in index[k]:
                    cand_scores[cid] += k_weight
        sorted_cands = sorted(cand_scores.items(), key=lambda x: -x[1])
        candidates[qid] = [cid for cid, _ in sorted_cands[:top_k]]
    return candidates


def make_embedding_text(name: str, address: str) -> str:
    """Format combined representation emphasizing name and street/door numbers."""
    n = clean_text(name)
    a = clean_text(address)
    nums = " ".join(re.findall(r'\b\d+\b', a))
    return f"{n} {n} {nums} {a}"


def build_and_query_ann(pool_df: pd.DataFrame, query_df: pd.DataFrame, svd_dim: int = 64, top_k: int = 10):
    """
    ANN blocking using character n-gram TF-IDF, TruncatedSVD, and FAISS IndexFlatIP.
    Partitioned per country to avoid cross-country comparisons and keep memory bounded.
    """
    pool_texts = [make_embedding_text(r['business_name'], r['business_address']) for _, r in pool_df.iterrows()]
    query_texts = [make_embedding_text(r['business_name'], r['business_address']) for _, r in query_df.iterrows()]

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=25000)
    X_pool_sparse = vectorizer.fit_transform(pool_texts)
    X_query_sparse = vectorizer.transform(query_texts)

    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    X_pool_dense = svd.fit_transform(X_pool_sparse).astype(np.float32)
    X_query_dense = svd.transform(X_query_sparse).astype(np.float32)

    faiss.normalize_L2(X_pool_dense)
    faiss.normalize_L2(X_query_dense)

    pool_entity_ids = pool_df['entity_id'].values
    query_entity_ids = query_df['entity_id'].values

    countries = query_df['country'].dropna().unique()
    ann_candidates = defaultdict(list)

    for c in countries:
        c_pool_mask = (pool_df['country'] == c).values
        c_query_mask = (query_df['country'] == c).values

        c_pool_indices = np.where(c_pool_mask)[0]
        c_query_indices = np.where(c_query_mask)[0]

        if len(c_pool_indices) == 0 or len(c_query_indices) == 0:
            continue

        c_pool_vectors = X_pool_dense[c_pool_indices]
        c_query_vectors = X_query_dense[c_query_indices]

        index_faiss = faiss.IndexFlatIP(svd_dim)
        index_faiss.add(c_pool_vectors)
        _, I = index_faiss.search(c_query_vectors, top_k)

        for q_sub_i, p_neighbor_ranks in enumerate(I):
            qid = query_entity_ids[c_query_indices[q_sub_i]]
            ann_candidates[qid] = [
                pool_entity_ids[c_pool_indices[p_idx]]
                for p_idx in p_neighbor_ranks if p_idx >= 0
            ]

    return ann_candidates


def generate_candidates(pool_df: pd.DataFrame, query_df: pd.DataFrame, top_inv: int = 12, top_ann: int = 10):
    """
    Combined candidate generator (Inverted Index + ANN).
    Returns dict mapping query_entity_id -> list of candidate S2/S3 entity_ids.
    """
    # 1. Inverted index with IDF scoring
    inv_index, idf = build_inverted_index(pool_df, max_posting=120)
    inv_cands = query_inverted_index(query_df, inv_index, idf, top_k=top_inv)

    # 2. ANN
    ann_cands = build_and_query_ann(pool_df, query_df, svd_dim=64, top_k=top_ann)

    # 3. Interleaved merge ensuring S2-/S3- valid prefixes and no duplicates
    candidates = {}
    for qid in query_df['entity_id']:
        inv_list = inv_cands.get(qid, [])[:top_inv]
        ann_list = ann_cands.get(qid, [])[:top_ann]

        seen = set()
        merged = []
        for cid in inv_list + ann_list:
            if cid.startswith(('S2-', 'S3-')) and cid not in seen:
                seen.add(cid)
                merged.append(cid)
        candidates[qid] = merged

    return candidates
