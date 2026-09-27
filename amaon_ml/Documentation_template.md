# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Team EntityResolvers  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We designed and implemented an end-to-end, production-scale Business Entity Resolution system achieving an official **Macro $F_{0.5}$ of 0.9457** and **97.06% micro-precision** on held-out validation data. Our solution combines an address-centric hybrid blocking pipeline (IDF-weighted inverted indexing + FAISS ANN embedding similarity) with a 15-feature LightGBM gradient boosted decision tree classifier, fine-tuned specifically for the precision-oriented $F_{0.5}$ metric with an optimal decision threshold of 0.895.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 10+ million record dataset revealed four primary matching failure patterns:
1. **Script Discrepancies (53.3% of initial blocking misses in India):** Business names frequently appear in regional Indic scripts (Devanagari, Tamil, Bengali) in one source and Latin script in another. However, address tokens consistently remain in Latin script and retain distinctive street names and PIN codes.
2. **Dense Keyword Inflation:** Generic terms (e.g., "Enterprises", "Trading", "Private Limited", "Capital") caused unweighted inverted indices to flood posting lists with false candidates, crowding out true matches.
3. **Singleton Distribution:** 8.5% of reference entities are true singletons with no match in candidate sources, which receive an all-or-nothing score (1.0 for correctly predicting empty, 0.0 for any false positive merge).
4. **Held-out Country Generalization:** The training set consists of US and India, while the test set introduces unseen geographic distributions (France).

### 2.2 Solution Strategy
**Approach Type:** Hybrid Multi-Stage Pipeline (Country Partitioned Blocking + Tabular Pairwise GBDT Matching + $F_{0.5}$ Threshold Calibration).  
**Core Innovation:** 
- **Address-Only Compound Blocking Keys:** Bigrams of significant address tokens and length-filtered locality tokens that recover cross-script entities without requiring brittle phonetic transliteration.
- **In-Stream Memory Capped Posting:** Bounded posting list growth (`STREAM_MAX_POSTING = 300`, `FINAL_MAX_POSTING = 120`) paired with smooth IDF weighting ($\text{score} += \log(1 + N / |P_k|)$) to eliminate common-key skew while guaranteeing strict memory safety.
- **Precision-Weighted Threshold Sweeping:** Rather than using the arbitrary 0.5 decision boundary, sweeping thresholds against the exact $F_{0.5}$ macro-average formula boosted the metric from 0.8916 to 0.9457.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. Standardized name prefixes and clean name token bigrams.
  2. Address-only compound keys (consecutive significant address token pairs).
  3. Standalone distinctive locality and alphanumeric tokens ($\ge 6$ characters).
  4. FAISS character n-gram (3-5 char) TF-IDF embeddings compressed via TruncatedSVD ($\text{dim}=48$) under Inner Product similarity (`IndexFlatIP`).
- **Candidate pairs generated:**
  - India: 18,945,458 candidates across 883,188 entities (~21.45 candidates/entity).
  - US: 27,376,495 candidates across 1,323,633 entities (~20.68 candidates/entity).
  - Test Set: ~20.9 candidates/entity maintained across all countries (France, US, India).
- **How true matches were protected:**
  - Complementary retrieval: Top-12 inverted index matches merged with Top-10 FAISS ANN semantic matches.
  - IDF weighting ensures specific street names or unique business words dominate over high-frequency corporate suffixes.
  - Combined candidate recall reached **88.2% on US** and **100% on the stratified validation slice**.

---

## 4. Matching Model

**Features used (15 total pairwise similarity features):**
- **Name features:** Levenshtein distance ratio, Partial ratio, Token sort ratio, Token set ratio, Jaro-Winkler similarity, Token Jaccard similarity, Length ratio.
- **Address features:** Levenshtein distance ratio, Token sort ratio, Token overlap Jaccard, Length ratio, Exact door/street number fractional match.
- **Cross & Interaction features:** Exact country match binary flag, Combined Name + Address Jaccard similarity, Count of shared blocking keys.

**Model type:** LightGBM Gradient Boosted Decision Trees (`binary_logloss`, `num_leaves=63`, `learning_rate=0.05`, `scale_pos_weight=6.2`, early stopping at 1000 rounds).  
**Top Feature Importances (Gain):**
1. `combined_jaccard` (20,321,884.3)
2. `num_shared_blocking_keys` (3,723,884.8)
3. `addr_number_match` (2,103,093.3)
4. `addr_token_overlap` (1,242,813.6)
5. `name_partial_ratio` (1,158,801.0)

**Threshold selection method:** Grid search sweep over decision thresholds $\in [0.10, 0.95]$ evaluated directly against Macro $F_{0.5}$. The optimal threshold was found at **`0.895`**, reflecting $F_{0.5}$'s 4:1 precision-over-recall penalty.

---

## 5. Results & Error Analysis

### Official Validation Metrics (25,000 Stratified Entities, 524,822 Candidate Pairs)
- **Macro $F_{0.5}$ (Official Metric):** **`0.9457`**
- **Micro Precision:** **`0.9706`** (97.06%)
- **Micro Recall:** **`0.9201`** (92.01%)
- **Singletons Correctly Identified:** **91.0%** (1,938 / 2,130)
- **Matched Entities Scoring $F_{0.5} \ge 0.5$:** **98.3%** (22,488 / 22,870)
- **Entities Scoring $F_{0.5} \ge 0.8$:** **92.1%** (23,035 / 25,000)

### Error Analysis
- **Common False Positives (Wrong Merges):** Co-located distinct businesses sharing identical addresses (e.g. multi-tenant office suites or strip malls where names share a broad category like "Dental Care" vs "Dental Clinic").
- **Common False Negatives (Missed Matches):** Businesses undergoing substantial name rebranding where address formatting was entirely unparsed or missing postal numbers.

---

## 6. Conclusion
By pairing an address-anchored, streaming hybrid blocking stage with a lightweight LightGBM matching model tuned to the $F_{0.5}$ metric, our solution achieves high macro $F_{0.5}$ (0.9457) and sub-second per-batch throughput while strictly adhering to open-source licensing and parameter constraints.

---

## Appendix

### A. Code Artefacts
The codebase is structured under `student_resource/code/business_entity_resolution/src/`:
- `blocking.py`: Core blocking key extractors, inverted index, and FAISS ANN routines.
- `run_full_blocking.py`: Production streaming candidate generation with in-stream memory caps.
- `make_split.py`: Stratified train/validation splitting preserving singleton and country distributions.
- `feature_engineering.py`: High-throughput pairwise similarity extraction with streaming TSV serialization.
- `train_model.py`: LightGBM training and automated Macro $F_{0.5}$ threshold sweep.
- `run_test_inference.py`: End-to-end streaming test-set inference, partition merging, and submission validation.
- `scorer.py`: Official Macro $F_{0.5}$ metric implementation.

### Reproducibility Commands:
```powershell
# 1. Generate full test predictions and submission files
python code\business_entity_resolution\src\run_test_inference.py --mode test --country all

# 2. Validate submission integrity
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```
