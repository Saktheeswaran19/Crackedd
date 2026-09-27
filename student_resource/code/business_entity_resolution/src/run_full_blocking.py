"""
Production-scale candidate generation (blocking) pipeline with streaming memory protection.

Fixes applied:
1. Cap posting list growth DURING streaming: once len(index[k]) reaches STREAM_MAX_POSTING (300),
   further entries for that key are discarded immediately.
2. Fast tuple-based streaming with explicit chunk deallocation and periodic gc.collect().
3. Memory checkpoint logging every 500k records (Process RSS MB and System RAM % via Windows API).
4. Configurable partition execution (--country India / --country US / --country all).
"""

import sys
import io
import os
import time
import math
import gc
import re
import argparse
import ctypes
from ctypes import wintypes
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
import blocking

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "train_candidate_pairs.tsv")
S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"
GT_PATH = "dataset/train/train_ground_truth.tsv"

TOP_INV = 12
TOP_ANN = 10
SVD_DIM = 48
FINAL_MAX_POSTING = 120
STREAM_MAX_POSTING = 300
CHUNK_SIZE = 100000


# --- Memory Tracking via Windows ctypes API ---
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', ctypes.c_ulong),
        ('dwMemoryLoad', ctypes.c_ulong),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]


def get_mem_checkpoint():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    sys_used_gb = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 ** 3)
    sys_total_gb = stat.ullTotalPhys / (1024 ** 3)
    load_pct = stat.dwMemoryLoad

    handle = ctypes.windll.kernel32.GetCurrentProcess()
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    proc_rss_mb = counters.WorkingSetSize / (1024 * 1024)
    proc_peak_mb = counters.PeakWorkingSetSize / (1024 * 1024)

    return f"Proc RSS: {proc_rss_mb:.1f}MB (Peak: {proc_peak_mb:.1f}MB) | Sys RAM: {sys_used_gb:.2f}/{sys_total_gb:.2f}GB ({load_pct}%)", load_pct, proc_peak_mb


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def process_country_partition(country_label, output_handle):
    log("="*70)
    log(f"STARTING COUNTRY PARTITION: {country_label}")
    mem_str, _, _ = get_mem_checkpoint()
    log(f"Initial Memory: {mem_str}")
    log("="*70)
    t_start = time.time()
    peak_sys_load = 0
    peak_proc_mb = 0

    # Step 1: Collect 100k sample to fit Vectorizer & SVD
    log(f"[{country_label}] Collecting 100k sample to fit Vectorizer and TruncatedSVD...")
    t0 = time.time()
    sample_texts = []
    for chunk in pd.read_csv(S2_PATH, sep='\t', dtype=str, encoding='utf-8', chunksize=CHUNK_SIZE):
        sub = chunk[chunk['country'] == country_label]
        for row in sub[['business_name', 'business_address']].itertuples(index=False):
            sample_texts.append(blocking.make_embedding_text(row[0], row[1]))
            if len(sample_texts) >= 100000:
                break
        if len(sample_texts) >= 100000:
            break

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=25000)
    X_samp_sparse = vectorizer.fit_transform(sample_texts)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    svd.fit(X_samp_sparse)
    del sample_texts, X_samp_sparse
    gc.collect()
    log(f"[{country_label}] Vectorizer & SVD (dim={SVD_DIM}) fitted in {time.time()-t0:.2f}s")

    # Step 2: Stream candidate pool from S2 and S3 into Inverted Index & FAISS
    # FIX: Cap posting list length during streaming to STREAM_MAX_POSTING (300)
    log(f"[{country_label}] Streaming candidate records from S2 and S3 (in-stream posting cap: {STREAM_MAX_POSTING})...")
    t0 = time.time()
    pool_ids = []
    index = defaultdict(list)
    faiss_index = faiss.IndexFlatIP(SVD_DIM)

    p_idx = 0
    records_since_checkpoint = 0

    for src_name, src_path in [("Source2", S2_PATH), ("Source3", S3_PATH)]:
        log(f"[{country_label}] Streaming {src_name}...")
        for chunk in pd.read_csv(src_path, sep='\t', dtype=str, encoding='utf-8', chunksize=CHUNK_SIZE):
            c_mask = (chunk['country'] == country_label)
            if not c_mask.any():
                continue
            sub = chunk[c_mask]

            batch_texts = []
            for row in sub[['entity_id', 'business_name', 'business_address']].itertuples(index=False):
                p_id, p_name, p_addr = row[0], row[1], row[2]
                pool_ids.append(p_id)
                batch_texts.append(blocking.make_embedding_text(p_name, p_addr))

                keys = blocking.get_blocking_keys(p_name, p_addr, country_label)
                for k in keys:
                    post = index[k]
                    # IN-STREAM CAP: stop growth at 300 to prevent ballooning on common keys
                    if len(post) < STREAM_MAX_POSTING:
                        post.append(p_idx)
                p_idx += 1

            # Transform batch and add to FAISS
            X_b_sparse = vectorizer.transform(batch_texts)
            X_b_dense = svd.transform(X_b_sparse).astype(np.float32)
            faiss.normalize_L2(X_b_dense)
            faiss_index.add(X_b_dense)

            records_since_checkpoint += len(sub)
            del batch_texts, X_b_sparse, X_b_dense, sub, chunk

            if records_since_checkpoint >= 500000:
                gc.collect()
                mem_str, load_pct, proc_mb = get_mem_checkpoint()
                peak_sys_load = max(peak_sys_load, load_pct)
                peak_proc_mb = max(peak_proc_mb, proc_mb)
                log(f"[{country_label}] Checkpoint: {p_idx:,} pool records streamed ({len(index):,} index keys) | {mem_str}")
                records_since_checkpoint = 0

    gc.collect()
    n_pool = len(pool_ids)
    mem_str, load_pct, proc_mb = get_mem_checkpoint()
    peak_sys_load = max(peak_sys_load, load_pct)
    peak_proc_mb = max(peak_proc_mb, proc_mb)
    log(f"[{country_label}] Pool complete: {n_pool:,} records in {time.time()-t0:.2f}s | {mem_str}")

    # Step 3: Finalize posting lists to FINAL_MAX_POSTING (120) & compute IDF
    log(f"[{country_label}] Finalizing posting lists to {FINAL_MAX_POSTING} and computing IDF weights...")
    t0 = time.time()
    idf = {}
    for k, post in list(index.items()):
        if len(post) > FINAL_MAX_POSTING:
            index[k] = post[:FINAL_MAX_POSTING]
        idf[k] = math.log(1.0 + n_pool / (len(index[k]) + 1))
    gc.collect()
    log(f"[{country_label}] Inverted index finalized in {time.time()-t0:.2f}s with {len(index):,} keys")

    # Step 4: Stream Source-1 Queries and write candidates directly to disk
    log(f"[{country_label}] Streaming Source-1 records and writing candidates incrementally...")
    t0 = time.time()
    s1_processed = 0
    total_candidates_generated = 0
    records_since_checkpoint = 0

    for chunk in pd.read_csv(S1_PATH, sep='\t', dtype=str, encoding='utf-8', chunksize=50000):
        c_mask = (chunk['country'] == country_label)
        if not c_mask.any():
            continue
        sub = chunk[c_mask]

        q_texts = [blocking.make_embedding_text(r[0], r[1]) for r in sub[['business_name', 'business_address']].itertuples(index=False)]
        X_q_sparse = vectorizer.transform(q_texts)
        X_q_dense = svd.transform(X_q_sparse).astype(np.float32)
        faiss.normalize_L2(X_q_dense)
        _, ann_I = faiss_index.search(X_q_dense, TOP_ANN)
        del q_texts, X_q_sparse, X_q_dense

        chunk_lines = []
        for row_i, r in enumerate(sub[['entity_id', 'business_name', 'business_address']].itertuples(index=False)):
            s1_id, b_name, b_addr = r[0], r[1], r[2]
            keys = blocking.get_blocking_keys(b_name, b_addr, country_label)

            cand_scores = defaultdict(float)
            for k in keys:
                if k in index:
                    k_w = idf[k]
                    for p_i in index[k]:
                        cand_scores[p_i] += k_w

            sorted_inv = sorted(cand_scores.items(), key=lambda x: -x[1])
            inv_cands = [pool_ids[p_i] for p_i, _ in sorted_inv[:TOP_INV]]

            ann_p_indices = ann_I[row_i]
            ann_cands = [pool_ids[p_i] for p_i in ann_p_indices if p_i >= 0]

            seen = set()
            merged = []
            for cid in inv_cands + ann_cands:
                if cid.startswith(('S2-', 'S3-')) and cid not in seen:
                    seen.add(cid)
                    merged.append(cid)

            chunk_lines.append(f"{s1_id}\t{','.join(merged)}\n")
            total_candidates_generated += len(merged)

        output_handle.writelines(chunk_lines)
        output_handle.flush()
        s1_processed += len(sub)
        records_since_checkpoint += len(sub)
        del sub, chunk, chunk_lines

        if records_since_checkpoint >= 200000:
            gc.collect()
            mem_str, load_pct, proc_mb = get_mem_checkpoint()
            peak_sys_load = max(peak_sys_load, load_pct)
            peak_proc_mb = max(peak_proc_mb, proc_mb)
            log(f"[{country_label}] S1 Progress: {s1_processed:,} records ({total_candidates_generated:,} cands, avg {total_candidates_generated/s1_processed:.2f}/entity) | {mem_str}")
            records_since_checkpoint = 0

    t_query_total = time.time() - t0
    log(f"[{country_label}] S1 query completed in {t_query_total:.2f}s ({s1_processed/max(1,t_query_total):.1f} entities/sec)")
    log(f"[{country_label}] Total candidates: {total_candidates_generated:,} (Avg {total_candidates_generated/max(1,s1_processed):.2f}/entity)")

    # Complete cleanup of partition structures
    del pool_ids, index, idf, vectorizer, svd, faiss_index
    gc.collect()
    mem_str, _, _ = get_mem_checkpoint()
    log(f"[{country_label}] Partition finished in {time.time()-t_start:.2f}s. Memory after cleanup: {mem_str}")
    log(f"[{country_label}] PEAK MEMORY: Process WorkingSet = {peak_proc_mb:.1f}MB, Peak System Load = {peak_sys_load}%")
    return s1_processed, total_candidates_generated, peak_sys_load, peak_proc_mb


def evaluate_ground_truth_recall(candidate_file, gt_file, filter_s1_ids=None):
    log("="*70)
    log("EVALUATING OVERALL TRUE MATCH RECALL AGAINST GROUND TRUTH")
    log("="*70)
    t0 = time.time()

    log("Loading Ground Truth matching records...")
    gt_map = {}
    total_true_pairs = 0
    total_singletons = 0
    total_s1_gt = 0

    for chunk in pd.read_csv(gt_file, sep='\t', dtype=str, encoding='utf-8', chunksize=200000):
        if filter_s1_ids is not None:
            chunk = chunk[chunk['source1_entity_id'].isin(filter_s1_ids)]
            if chunk.empty:
                continue

        for _, r in chunk.iterrows():
            total_s1_gt += 1
            s1_id = r['source1_entity_id']
            m_str = str(r['matched_entity_ids'])
            if m_str and m_str != 'nan':
                mids = set(m.strip() for m in m_str.split(',') if m.strip())
                if mids:
                    gt_map[s1_id] = mids
                    total_true_pairs += len(mids)
                else:
                    total_singletons += 1
            else:
                total_singletons += 1

    log(f"Ground Truth loaded in {time.time()-t0:.2f}s: {total_s1_gt:,} S1 entities ({total_singletons:,} singletons, {total_true_pairs:,} true match pairs)")

    log("Streaming generated candidate pairs to calculate full recall...")
    t0 = time.time()
    recalled_pairs = 0
    candidates_checked = 0

    with open(candidate_file, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            candidates_checked += 1
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            s1_id = parts[0]
            cand_str = parts[1]
            if not cand_str:
                continue
            true_set = gt_map.get(s1_id)
            if true_set:
                cand_set = set(cand_str.split(','))
                recalled_pairs += len(true_set & cand_set)

    overall_recall = recalled_pairs / max(1, total_true_pairs)
    log(f"Recall evaluation finished in {time.time()-t0:.2f}s:")
    log(f"  Total True Match Pairs: {total_true_pairs:,}")
    log(f"  True Matches Recalled:  {recalled_pairs:,}")
    log(f"  Partition Recall:       {overall_recall:.4%} ({recalled_pairs:,} / {total_true_pairs:,})")
    return overall_recall, total_true_pairs, recalled_pairs, total_singletons


def main():
    parser = argparse.ArgumentParser(description="Full scale blocking pipeline with memory protection.")
    parser.add_argument("--country", default="India", choices=["India", "US", "all"],
                        help="Country partition to run (default: India for validation test).")
    args = parser.parse_args()

    start_time = time.time()
    log("ML CHALLENGE 2026: CANDIDATE GENERATION PIPELINE")
    log(f"Country Target: {args.country}")
    log(f"Output File:    {OUTPUT_FILE}")
    log(f"Configuration:  top_inv={TOP_INV}, top_ann={TOP_ANN}, SVD_DIM={SVD_DIM}, STREAM_MAX_POSTING={STREAM_MAX_POSTING}")

    countries_to_run = ["India"] if args.country == "India" else (["US"] if args.country == "US" else ["US", "India"])

    # If running single country, append country suffix to output file
    if args.country != "all":
        out_path = OUTPUT_FILE.replace(".tsv", f"_{args.country}.tsv")
    else:
        out_path = OUTPUT_FILE

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    total_s1 = 0
    total_candidates = 0
    peak_system_load = 0
    peak_process_rss = 0

    with open(out_path, 'a', encoding='utf-8') as out_f:
        for c in countries_to_run:
            s1_c, cands_c, peak_sys, peak_proc = process_country_partition(c, out_f)
            total_s1 += s1_c
            total_candidates += cands_c
            peak_system_load = max(peak_system_load, peak_sys)
            peak_process_rss = max(peak_process_rss, peak_proc)

    total_time = time.time() - start_time
    avg_cands = total_candidates / max(1, total_s1)

    log("="*70)
    log(f"PARTITION EXECUTION COMPLETE: {args.country}")
    log(f"Total S1 Entities Processed: {total_s1:,}")
    log(f"Total Candidates Generated:   {total_candidates:,}")
    log(f"Average Candidates / Entity:  {avg_cands:.2f}")
    log(f"Total Pipeline Runtime:       {total_time/60:.2f} minutes ({total_time:.1f}s)")
    log(f"Peak Process WorkingSet (RSS):{peak_process_rss:.1f} MB")
    log(f"Peak System RAM Load:         {peak_system_load}%")
    log("="*70)

    # Evaluate ground truth recall for the processed S1 entities
    # Read S1 IDs processed from output file to filter ground truth
    log("Collecting processed S1 IDs for targeted Ground Truth evaluation...")
    processed_ids = set()
    with open(out_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.split('\t', 1)
            if p:
                processed_ids.add(p[0])

    recall, total_gt, recalled_gt, singletons = evaluate_ground_truth_recall(out_path, GT_PATH, filter_s1_ids=processed_ids)

    log("\n" + "="*80)
    log(f"SUMMARY REPORT: {args.country.upper()} PARTITION AT FULL SCALE")
    log("="*80)
    log(f"1. Output File:            {out_path}")
    log(f"2. Total S1 Records:       {total_s1:,}")
    log(f"3. Total Candidates:       {total_candidates:,}")
    log(f"4. Avg Candidates/Entity:  {avg_cands:.2f}")
    log(f"5. Pipeline Execution Time:{total_time/60:.2f} min ({total_time:.1f}s)")
    log(f"6. Peak Process WorkingSet:{peak_process_rss:.1f} MB")
    log(f"7. Peak System RAM Load:   {peak_system_load}%")
    log(f"8. Evaluated GT True Pairs:{total_gt:,}")
    log(f"9. True Pairs Recalled:    {recalled_gt:,}")
    log(f"10. Partition Recall:      {recall:.4%}")
    log("="*80)


if __name__ == "__main__":
    main()
