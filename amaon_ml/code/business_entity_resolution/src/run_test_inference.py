"""
Production-Scale Test Set Inference & Evaluation Pipeline.

Performs:
1. Country-partitioned streaming blocking (top_inv=12, top_ann=10, SVD_dim=48).
2. On-the-fly pairwise feature engineering (15 compound similarity features).
3. LightGBM match prediction with precision-tuned threshold (0.895).
4. Direct streaming output to matching_results.tsv and candidate_pairs.tsv.
5. Submission format validation via utils/validate_submission.py.
6. Validation report generation with detailed error analysis (--mode val_report).

Run from student_resource/ root:
    # Quick test on France partition first (~15 mins):
    python code\\business_entity_resolution\\src\\run_test_inference.py --mode test --country France

    # Full test set inference (France + US + India):
    python code\\business_entity_resolution\\src\\run_test_inference.py --mode test --country all

    # Validation evaluation & report:
    python code\\business_entity_resolution\\src\\run_test_inference.py --mode val_report
"""

import os
import sys
import io
import time
import math
import gc
import re
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import blocking
from feature_engineering import (
    compute_pair_features, load_entities, load_candidate_pairs,
    load_split_ids, load_ground_truth, FEATURE_COLS
)
from scorer import f_beta_score, macro_f_beta

# Ensure UTF-8 output
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SRC_DIR = os.path.dirname(__file__)
STUDENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SRC_DIR)))
OUTPUT_DIR = os.path.join(STUDENT_DIR, "output")
MODEL_PATH = os.path.join(SRC_DIR, "model.lgb")
CONFIG_PATH = os.path.join(SRC_DIR, "model_config.json")

S1_TEST_PATH = "dataset/test/test_source1.tsv"
S2_TEST_PATH = "dataset/test/test_source2.tsv"
S3_TEST_PATH = "dataset/test/test_source3.tsv"

TOP_INV = 12
TOP_ANN = 10
SVD_DIM = 48
FINAL_MAX_POSTING = 120
STREAM_MAX_POSTING = 300
CHUNK_SIZE = 100000


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def process_test_country_partition(country_label, model, threshold, cand_out_handle, match_out_handle):
    """
    Stream candidate generation, feature engineering, and model inference for one country.
    Writes candidate_pairs and matching_results directly to file handles.
    """
    log("=" * 70)
    log(f"STARTING TEST PARTITION: {country_label}")
    log("=" * 70)
    t_start = time.time()

    # Step 1: Collect up to 100k samples from S2 to fit Vectorizer & SVD
    log(f"[{country_label}] Collecting sample to fit Vectorizer & TruncatedSVD (dim={SVD_DIM})...")
    sample_texts = []
    for chunk in pd.read_csv(S2_TEST_PATH, sep='\t', dtype=str, encoding='utf-8', chunksize=CHUNK_SIZE):
        sub = chunk[chunk['country'] == country_label]
        for row in sub[['business_name', 'business_address']].itertuples(index=False):
            sample_texts.append(blocking.make_embedding_text(str(row[0]), str(row[1])))
            if len(sample_texts) >= 100000:
                break
        if len(sample_texts) >= 100000:
            break

    if not sample_texts:
        log(f"[{country_label}] Warning: No pool records found.")
        return 0, 0, 0

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=25000)
    X_samp = vectorizer.fit_transform(sample_texts)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    svd.fit(X_samp)
    del sample_texts, X_samp
    gc.collect()

    # Step 2: Stream Candidate Pool into Inverted Index & FAISS, and store compact lookup
    log(f"[{country_label}] Streaming pool records (S2 + S3) into Inverted Index & FAISS...")
    t0 = time.time()
    pool_ids = []
    pool_lookup = {}  # entity_id -> (name, addr, country)
    index = defaultdict(list)
    faiss_index = faiss.IndexFlatIP(SVD_DIM)

    p_idx = 0
    for src_name, src_path in [("Source2", S2_TEST_PATH), ("Source3", S3_TEST_PATH)]:
        for chunk in pd.read_csv(src_path, sep='\t', dtype=str, encoding='utf-8',
                                 usecols=['entity_id', 'business_name', 'business_address', 'country'],
                                 chunksize=CHUNK_SIZE):
            c_mask = (chunk['country'] == country_label)
            if not c_mask.any():
                continue
            sub = chunk[c_mask].fillna('')

            batch_texts = []
            for row in sub.itertuples(index=False):
                eid, name, addr, ctry = row.entity_id, str(row.business_name), str(row.business_address), str(row.country)
                pool_ids.append(eid)
                pool_lookup[eid] = (name, addr, ctry)
                batch_texts.append(blocking.make_embedding_text(name, addr))

                keys = blocking.get_blocking_keys(name, addr, country_label)
                for k in keys:
                    post = index[k]
                    if len(post) < STREAM_MAX_POSTING:
                        post.append(p_idx)
                p_idx += 1

            X_b_sparse = vectorizer.transform(batch_texts)
            X_b_dense = svd.transform(X_b_sparse).astype(np.float32)
            faiss.normalize_L2(X_b_dense)
            faiss_index.add(X_b_dense)
            del batch_texts, X_b_sparse, X_b_dense, sub, chunk

    n_pool = len(pool_ids)
    log(f"[{country_label}] Pool complete: {n_pool:,} records in {time.time()-t0:.1f}s ({len(index):,} index keys)")

    # Step 3: Finalize posting lists to FINAL_MAX_POSTING (120) & compute IDF
    log(f"[{country_label}] Finalizing posting lists to {FINAL_MAX_POSTING} and computing IDF...")
    idf = {}
    for k, post in list(index.items()):
        if len(post) > FINAL_MAX_POSTING:
            index[k] = post[:FINAL_MAX_POSTING]
        idf[k] = math.log(1.0 + n_pool / (len(index[k]) + 1))
    gc.collect()

    # Step 4: Stream Source-1 Queries, compute features, score with LightGBM, and stream to disk
    log(f"[{country_label}] Streaming S1 test entities, predicting matches (threshold={threshold:.3f})...")
    t0 = time.time()
    s1_processed = 0
    total_candidates = 0
    total_matches = 0

    cand_lines = []
    match_lines = []
    BATCH_WRITE = 10000

    for chunk in pd.read_csv(S1_TEST_PATH, sep='\t', dtype=str, encoding='utf-8',
                             usecols=['entity_id', 'business_name', 'business_address', 'country'],
                             chunksize=25000):
        c_mask = (chunk['country'] == country_label)
        if not c_mask.any():
            continue
        sub = chunk[c_mask].fillna('')

        q_texts = [blocking.make_embedding_text(str(r[1]), str(r[2]))
                   for r in sub[['entity_id', 'business_name', 'business_address']].itertuples(index=False)]
        X_q_sparse = vectorizer.transform(q_texts)
        X_q_dense = svd.transform(X_q_sparse).astype(np.float32)
        faiss.normalize_L2(X_q_dense)
        _, ann_I = faiss_index.search(X_q_dense, TOP_ANN)
        del q_texts, X_q_sparse, X_q_dense

        for row_i, r in enumerate(sub.itertuples(index=False)):
            s1_id = r.entity_id
            b_name, b_addr, b_ctry = str(r.business_name), str(r.business_address), str(r.country)
            s1_rec = (b_name, b_addr, b_ctry)
            s1_keys = set(blocking.get_blocking_keys(b_name, b_addr, country_label))

            cand_scores = defaultdict(float)
            for k in s1_keys:
                if k in index:
                    k_w = idf[k]
                    for p_i in index[k]:
                        cand_scores[p_i] += k_w

            sorted_inv = sorted(cand_scores.items(), key=lambda x: -x[1])
            inv_cands = [pool_ids[p_i] for p_i, _ in sorted_inv[:TOP_INV]]

            ann_p_indices = ann_I[row_i]
            ann_cands = [pool_ids[p_i] for p_i in ann_p_indices if p_i >= 0]

            seen = set()
            merged_cands = []
            for cid in inv_cands + ann_cands:
                if cid.startswith(('S2-', 'S3-')) and cid not in seen:
                    seen.add(cid)
                    merged_cands.append(cid)

            cand_lines.append(f"{s1_id}\t{','.join(merged_cands)}\n")
            total_candidates += len(merged_cands)

            # Score candidates with LightGBM if any exist
            matched_cids = []
            if merged_cands:
                feat_rows = []
                valid_cids = []
                for cid in merged_cands:
                    cand_rec = pool_lookup.get(cid)
                    if cand_rec is None:
                        continue
                    feats = compute_pair_features(s1_rec, cand_rec, s1_keys=s1_keys)
                    feat_rows.append([feats.get(f, 0.0) for f in FEATURE_COLS])
                    valid_cids.append(cid)

                if feat_rows:
                    X_pair = np.array(feat_rows, dtype=np.float32)
                    probs = model.predict(X_pair)
                    for c_idx, prob in enumerate(probs):
                        if prob >= threshold:
                            matched_cids.append(valid_cids[c_idx])

            match_lines.append(f"{s1_id}\t{','.join(matched_cids)}\n")
            total_matches += len(matched_cids)
            s1_processed += 1

            if len(cand_lines) >= BATCH_WRITE:
                cand_out_handle.writelines(cand_lines)
                match_out_handle.writelines(match_lines)
                cand_out_handle.flush()
                match_out_handle.flush()
                cand_lines.clear()
                match_lines.clear()

        if s1_processed % 50000 == 0:
            log(f"[{country_label}] S1 Processed: {s1_processed:,} entities | Candidates: {total_candidates:,} | Matches: {total_matches:,}")

    if cand_lines:
        cand_out_handle.writelines(cand_lines)
        match_out_handle.writelines(match_lines)
        cand_out_handle.flush()
        match_out_handle.flush()
        cand_lines.clear()
        match_lines.clear()

    total_time = time.time() - t_start
    log(f"[{country_label}] Completed: {s1_processed:,} S1 entities, {total_candidates:,} candidates ({total_candidates/max(1,s1_processed):.2f}/entity), {total_matches:,} matches in {total_time/60:.2f} mins")

    # Cleanup partition memory
    del pool_ids, pool_lookup, index, idf, faiss_index, vectorizer, svd
    gc.collect()

    return s1_processed, total_candidates, total_matches


def generate_val_report(predictions, ground_truth, candidate_pairs, report_path):
    """Generate a comprehensive validation report."""
    print("\n" + "=" * 80)
    print("GENERATING VALIDATION REPORT")
    print("=" * 80)

    all_s1_ids = set(ground_truth.keys())
    f05 = macro_f_beta(predictions, ground_truth, beta=0.5)

    per_entity_details = []
    tp_total = fp_total = fn_total = 0
    singleton_correct = singleton_wrong = 0
    matched_correct = matched_wrong = 0

    for s1_id in all_s1_ids:
        pred_set = predictions.get(s1_id, set())
        true_set = ground_truth.get(s1_id, set())
        score = f_beta_score(pred_set, true_set, beta=0.5)

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        is_singleton = len(true_set) == 0
        if is_singleton:
            if len(pred_set) == 0:
                singleton_correct += 1
            else:
                singleton_wrong += 1
        else:
            if score >= 0.5:
                matched_correct += 1
            else:
                matched_wrong += 1

        per_entity_details.append({
            's1_id': s1_id,
            'true_matches': len(true_set),
            'pred_matches': len(pred_set),
            'tp': tp, 'fp': fp, 'fn': fn,
            'f05_score': score,
            'is_singleton': is_singleton,
        })

    details_df = pd.DataFrame(per_entity_details)

    blocking_recalled = 0
    total_true_pairs = 0
    for s1_id, true_set in ground_truth.items():
        if true_set:
            total_true_pairs += len(true_set)
            cands = set(candidate_pairs.get(s1_id, []))
            blocking_recalled += len(true_set & cands)

    blocking_recall = blocking_recalled / max(1, total_true_pairs)
    micro_precision = tp_total / max(1, tp_total + fp_total)
    micro_recall = tp_total / max(1, tp_total + fn_total)

    n_singletons = (details_df['is_singleton']).sum()
    n_matched = len(details_df) - n_singletons

    cand_sizes = [len(v) for v in candidate_pairs.values()]
    avg_cands = np.mean(cand_sizes) if cand_sizes else 0

    report_lines = [
        "=" * 80,
        "AMAZON ML CHALLENGE 2026 — VALIDATION REPORT",
        "=" * 80,
        "",
        "## 1. Dataset Summary",
        f"  Total S1 entities evaluated:   {len(all_s1_ids):,}",
        f"  Singletons (no true matches):  {n_singletons:,} ({n_singletons/len(all_s1_ids):.1%})",
        f"  Matched entities:              {n_matched:,} ({n_matched/len(all_s1_ids):.1%})",
        f"  Total true match pairs:        {total_true_pairs:,}",
        "",
        "## 2. Blocking / Candidate Generation",
        f"  Blocking recall:               {blocking_recall:.4%} ({blocking_recalled:,}/{total_true_pairs:,})",
        f"  Avg candidates per entity:     {avg_cands:.2f}",
        f"  Total candidate pairs:         {sum(cand_sizes):,}",
        "",
        "## 3. Matching Model Performance",
        f"  *** Macro F_0.5 (official):    {f05:.4f} ***",
        f"  Micro precision:               {micro_precision:.4f}",
        f"  Micro recall:                  {micro_recall:.4f}",
        f"  Total TP: {tp_total:,}  FP: {fp_total:,}  FN: {fn_total:,}",
        "",
        "## 4. Singleton Performance",
        f"  Singletons correctly empty:    {singleton_correct:,} / {n_singletons:,} ({singleton_correct/max(1,n_singletons):.1%})",
        f"  Singletons with false merges:  {singleton_wrong:,} / {n_singletons:,} ({singleton_wrong/max(1,n_singletons):.1%})",
        "",
        "## 5. Matched Entity Performance",
        f"  Matched entities F_0.5 >= 0.5: {matched_correct:,} / {n_matched:,} ({matched_correct/max(1,n_matched):.1%})",
        f"  Matched entities F_0.5 < 0.5:  {matched_wrong:,} / {n_matched:,} ({matched_wrong/max(1,n_matched):.1%})",
        "",
        "## 6. Per-Entity F_0.5 Distribution",
        f"  Mean:   {details_df['f05_score'].mean():.4f}",
        f"  Median: {details_df['f05_score'].median():.4f}",
        f"  Std:    {details_df['f05_score'].std():.4f}",
        f"  Min:    {details_df['f05_score'].min():.4f}",
        f"  Max:    {details_df['f05_score'].max():.4f}",
        "",
        "  Score distribution:",
        f"    F_0.5 = 1.0 (perfect):       {(details_df['f05_score'] == 1.0).sum():,} ({(details_df['f05_score'] == 1.0).mean():.1%})",
        f"    F_0.5 >= 0.8:                 {(details_df['f05_score'] >= 0.8).sum():,} ({(details_df['f05_score'] >= 0.8).mean():.1%})",
        f"    F_0.5 >= 0.5:                 {(details_df['f05_score'] >= 0.5).sum():,} ({(details_df['f05_score'] >= 0.5).mean():.1%})",
        f"    F_0.5 = 0.0 (total miss):    {(details_df['f05_score'] == 0.0).sum():,} ({(details_df['f05_score'] == 0.0).mean():.1%})",
        "",
        "## 7. Error Analysis",
    ]

    worst = details_df.nlargest(20, 'fp')
    report_lines.append("  Top false positive entities (most wrong merges):")
    for _, row in worst.head(10).iterrows():
        report_lines.append(f"    {row['s1_id']}: {row['fp']} FP, {row['tp']} TP, F_0.5={row['f05_score']:.3f}")

    worst_fn = details_df[details_df['fn'] > 0].nlargest(10, 'fn')
    report_lines.append("")
    report_lines.append("  Top false negative entities (most missed matches):")
    for _, row in worst_fn.iterrows():
        report_lines.append(f"    {row['s1_id']}: {row['fn']} FN, {row['tp']} TP, F_0.5={row['f05_score']:.3f}")

    report_lines.extend(["", "=" * 80])
    report_text = '\n'.join(report_lines)
    print(report_text)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved → {report_path}")

    details_path = report_path.replace('.txt', '_details.tsv')
    details_df.to_csv(details_path, sep='\t', index=False)
    print(f"Per-entity details saved → {details_path}")

    return f05


def main():
    parser = argparse.ArgumentParser(description="Run test inference or validation report.")
    parser.add_argument("--mode", required=True, choices=["test", "val_report"],
                        help="test: generate submission files; val_report: evaluate on val split")
    parser.add_argument("--country", default="all", choices=["all", "France", "US", "India"],
                        help="Country partition for test inference (default: all)")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    t_start = time.time()

    # Load model and config
    print("Loading model...")
    model = lgb.Booster(model_file=args.model)

    with open(args.config, 'r') as f:
        config = json.load(f)
    threshold = config.get('best_threshold', 0.895)
    print(f"Using threshold: {threshold:.3f}")

    if args.mode == "test":
        print("\n=== TEST SET INFERENCE ===")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        final_cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
        final_match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")

        countries_to_run = ["France", "US", "India"] if args.country == "all" else [args.country]

        for country in countries_to_run:
            part_cand_file = os.path.join(SRC_DIR, f"test_candidate_pairs_{country}.tsv")
            part_match_file = os.path.join(SRC_DIR, f"test_matching_results_{country}.tsv")

            with open(part_cand_file, 'w', encoding='utf-8') as cand_f, \
                 open(part_match_file, 'w', encoding='utf-8') as match_f:
                cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
                match_f.write("source1_entity_id\tmatched_entity_ids\n")
                process_test_country_partition(country, model, threshold, cand_f, match_f)

        # Merge completed country partitions if all countries run
        if args.country == "all":
            log("Merging all country partitions into final submission files...")
            written_ids = set()

            with open(final_cand_path, 'w', encoding='utf-8') as out_cand, \
                 open(final_match_path, 'w', encoding='utf-8') as out_match:
                out_cand.write("source1_entity_id\tcandidate_entity_ids\n")
                out_match.write("source1_entity_id\tmatched_entity_ids\n")

                for country in ["France", "US", "India"]:
                    part_c = os.path.join(SRC_DIR, f"test_candidate_pairs_{country}.tsv")
                    part_m = os.path.join(SRC_DIR, f"test_matching_results_{country}.tsv")

                    if os.path.exists(part_c) and os.path.exists(part_m):
                        with open(part_c, 'r', encoding='utf-8') as fc, open(part_m, 'r', encoding='utf-8') as fm:
                            next(fc)
                            next(fm)
                            for line_c, line_m in zip(fc, fm):
                                out_cand.write(line_c)
                                out_match.write(line_m)
                                s1_id = line_c.split('\t', 1)[0]
                                written_ids.add(s1_id)

                # Ensure every S1 test entity has a row (even if missing from partitions)
                log("Verifying complete Source-1 coverage against test_source1.tsv...")
                all_s1_test = pd.read_csv(S1_TEST_PATH, sep='\t', usecols=['entity_id'], dtype=str)
                missing_count = 0
                for eid in all_s1_test['entity_id']:
                    if eid not in written_ids:
                        out_cand.write(f"{eid}\t\n")
                        out_match.write(f"{eid}\t\n")
                        missing_count += 1

                log(f"Merged output: {len(all_s1_test):,} S1 entities ({missing_count:,} missing padded with empty)")

            log(f"Candidate output: {final_cand_path}")
            log(f"Matching output:  {final_match_path}")

            # Run submission validator
            validator_script = os.path.join(STUDENT_DIR, "utils", "validate_submission.py")
            if os.path.exists(validator_script):
                log("\nRunning official submission validator...")
                os.system(f'"{sys.executable}" "{validator_script}" '
                          f'--matching "{final_match_path}" '
                          f'--candidate "{final_cand_path}" '
                          f'--test-dir "dataset/test"')

    elif args.mode == "val_report":
        print("\n=== VALIDATION REPORT ===")
        val_feat_path = os.path.join(SRC_DIR, "val_features.tsv")

        if os.path.exists(val_feat_path):
            print(f"Loading precomputed validation features from {val_feat_path}...")
            val_df = pd.read_csv(val_feat_path, sep='\t')
            print(f"Loaded {len(val_df):,} validation pairs.")

            X_val = val_df[FEATURE_COLS].values
            probs = model.predict(X_val)

            predictions = defaultdict(set)
            val_gt = defaultdict(set)
            candidate_pairs = defaultdict(list)

            for i, row in enumerate(val_df[['s1_id', 'candidate_id', 'label']].itertuples(index=False)):
                s1_id, cid, lbl = row[0], row[1], int(row[2])
                candidate_pairs[s1_id].append(cid)
                if lbl == 1:
                    val_gt[s1_id].add(cid)
                if probs[i] >= threshold:
                    predictions[s1_id].add(cid)

            for s1_id in candidate_pairs.keys():
                if s1_id not in val_gt:
                    val_gt[s1_id] = set()
                if s1_id not in predictions:
                    predictions[s1_id] = set()

            report_path = os.path.join(SRC_DIR, "validation_report.txt")
            f05 = generate_val_report(predictions, val_gt, candidate_pairs, report_path)

    total_time = time.time() - t_start
    print(f"\nTotal execution time: {total_time/60:.1f} minutes ({total_time:.1f}s)")


if __name__ == '__main__':
    main()
