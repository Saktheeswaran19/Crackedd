# Amazon ML Challenge 2026 — Business Entity Resolution — Handoff Notes

This document is a complete status snapshot so you can pick up this project with
zero missing context. Read this fully before touching code.

Repo: (add your GitHub URL here once confirmed)

---

## 1. The Challenge, In Plain Terms

Given business listings from 3 sources (Source 1 = reference/dedup'd, Source 2
and Source 3 = messy, noisy), find which Source-2/3 records match each Source-1
entity. A Source-1 entity may have zero, one, or many true matches.

**Hard rules that shape every decision in this pipeline:**
- **No external lookups, ever.** No geocoding APIs, no business registry lookups,
  no internet calls of any kind in the pipeline. Disqualifying if violated.
- **Never hardcode country-specific logic.** Training data only has US and India;
  the real test set adds France. Any US/India-only regex or parsing rule will
  silently break on French records. Everything must be language/country-agnostic.
- **Scored metric is macro-averaged F_0.5 per Source-1 entity**, including
  singletons. Precision is weighted 2x over recall — a wrong merge hurts twice as
  much as a missed match. A correctly-predicted empty list (singleton) scores a
  full 1.0; a false merge on a singleton scores 0.0.
- **Two outputs required**, both graded:
  1. `matching_results.tsv` — final matches (scored on leaderboard via F_0.5)
  2. `candidate_pairs.tsv` — your blocking stage's output (graded separately —
     smaller candidate sets per entity score better, but only if recall holds)
- **Final model constraints:** MIT/Apache-2.0 license only, ≤8B parameters.
- Full submission is a zip with `output/`, `code/business_entity_resolution/`
  (src + README + requirements.txt), and a filled `Documentation_template.md`.

---

## 2. Repo Structure (as it exists now)

```
student_resource/
├── dataset/                          # given by Amazon, NOT in git (see .gitignore)
│   ├── train/
│   │   ├── train_source1.tsv         # 2,206,821 rows
│   │   ├── train_source2.tsv         # 5,034,616 rows
│   │   ├── train_source3.tsv         # 5,285,603 rows
│   │   └── train_ground_truth.tsv    # 2,206,821 rows
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/
│   └── validate_submission.py        # given by Amazon — run before every submission
├── output/                           # empty so far — final matching_results.tsv +
│                                      # candidate_pairs.tsv go here eventually
├── code/business_entity_resolution/
│   ├── requirements.txt              # pinned deps (pandas, sklearn, lightgbm,
│   │                                 # rapidfuzz, sentence-transformers, faiss-cpu,
│   │                                 # jellyfish, tqdm)
│   ├── README.md                     # (needs to be filled in properly still)
│   ├── src/
│   │   ├── eda.py                    # initial exploratory analysis script
│   │   ├── blocking.py               # core blocking logic (get_blocking_keys,
│   │   │                             # make_embedding_text) — imported by both
│   │   │                             # benchmark and production scripts
│   │   ├── benchmark_blocking.py     # early benchmark script (superseded)
│   │   ├── run_full_blocking.py      # PRODUCTION full-scale blocking runner —
│   │   │                             # see Section 5, this is where you resume
│   │   └── (scorer.py, make_split.py — planned but not yet written as
│   │        separate files; see Section 6)
│   └── notes/
│       └── blocking_failure_analysis.md   # THE KEY DOCUMENT — full v1→v2
│           failure analysis, root causes, fixes, and stability validation.
│           Read this in full before doing anything else. Directly reusable
│           in the final Documentation_template.md methodology section.
├── .agent/skills/                    # Antigravity agent skills
│   ├── entity-resolution/SKILL.md    # PROJECT-SPECIFIC RULES — encodes every
│   │                                 # constraint above so the AI agent doesn't
│   │                                 # violate fair-play or optimize the wrong
│   │                                 # metric. Read this too.
│   ├── data-scientist/
│   ├── data-engineer/
│   └── ai-engineer/
│       # (all three downloaded from github.com/rmyndharis/antigravity-skills —
│       #  general-purpose skills, not project-specific)
├── Documentation_template.md         # given by Amazon — fill in at the end
├── README.md                         # the challenge brief itself
└── .gitignore
```

---

## 3. EDA Findings (already done — real numbers, from `eda.py` output)

- **Source1: 2,206,821 | Source2: 5,034,616 | Source3: 5,285,603 | GT: 2,206,821**
- Country split — Source1: US 1,323,633 / India 883,188 (roughly 60/40)
- **Singletons: 123,247 / 2,206,821 (5.6%)** — most entities DO have matches
- Match count distribution: peaks at 2-4 matches per entity, up to 11 in rare cases
- Business name length: mean ~24 chars. Address length: mean ~52 chars.
- Eyeballed real matched pairs manually — confirmed the noise patterns described
  in the brief (abbreviations, punctuation, transliteration, DBA names) are real
  and present in the data.

---

## 4. Blocking Stage — DONE and VALIDATED (this is the big win so far)

### v1 (baseline): inverted index + ANN (TF-IDF char n-grams + SVD + FAISS)
- Combined recall at (top_inv=20, top_ann=15): **84.84%** on a 10k-entity sample
- 15.16% of true matches were missed — investigated via failure analysis

### Failure analysis (30 sampled misses, categorized)
- **(a) 10%** — clear DBA/trade-name mismatch (e.g. legal name vs. brand name),
  address sometimes still matches — largely unfixable without external lookups
- **(b) 53.3%** — THE BIG ONE. Name written in Indic script (Devanagari, Tamil,
  etc.) in one source, Latin English in another. Address fields, however,
  almost always stay in Latin script and match well. Fixable WITHOUT any
  transliteration by blocking on address tokens alone.
- **(c) 6.7%** — severe typo/OCR-style corruption (e.g. name glued into a URL)
- **(d) 30%** — shared blocking keys DID exist, but generic high-frequency
  tokens (e.g. "Capital", "First") got outranked by raw-count scoring —
  a scoring bug, not a data problem. Fixable with IDF weighting.

### v2 fixes implemented (in `blocking.py`)
1. **Address-only compound blocking keys** — bigrams of significant address
   tokens + individual distinctive locality tokens (≥6 chars), independent of
   any name signal. Recovers category (b) without touching transliteration.
2. **IDF-weighted inverted index scoring** — replaced raw count aggregation
   with `score += log(1 + N/|posting_list|)` per shared key, so rare/specific
   keys outrank generic ones. Fixes category (d).

### v2 results (validated, NOT a fluke — see below)
- Inverted index alone: 78.81% → **94.15%** recall
- Combined recall at (20,15): 84.84% → **95.49%**
- Misses dropped from 5,268 → **1,568** (70% reduction)
- Remaining misses are now dominated (>65%) by categories (a)/(c) — the
  genuine, honest recall ceiling, not fixable without external data.

### Stability validation (cross-sample check — this matters for credibility)
Reran the full v2 benchmark on a **completely disjoint** 10k-entity slice
(rows 10k-20k instead of 0-10k). Recall varied by **less than 0.3%** across
every (top_inv, top_ann) setting tested. This confirms the improvement
generalizes and isn't overfit to one lucky sample.

### LOCKED PRODUCTION CONFIG
**`top_inv = 12, top_ann = 10`** — chosen deliberately over the higher-recall
(20,15) setting: only ~1 point of recall given up (94.52% vs 95.49%) in
exchange for ~40% fewer candidates per entity (20.25 vs 33.02 avg). This
matters because the challenge explicitly rewards smaller candidate sets when
recall holds — this is the documented rationale, already reflected in
`run_full_blocking.py`'s defaults.

**→ Full details, all example pairs, and the math are in
`code/business_entity_resolution/notes/blocking_failure_analysis.md`. Read it.**

---

## 5. Where Things Actually Stand — Full-Scale Production Run

**This is the unfinished, blocking (pun intended) task.**

`run_full_blocking.py` was built to run the locked config against the FULL
training set (not just 10k samples), country-partitioned (US and India
separately) for memory safety, with streaming/incremental writes so it doesn't
need to hold everything in RAM at once.

### What happened
- First version had a real bug: posting lists had no size cap DURING
  streaming (only truncated at the very end), causing unbounded memory growth
  on the full ~10.3M-record candidate pool. This caused the machine to hit
  96%+ system RAM and start swap-thrashing (confirmed via Task Manager: only
  ~10% CPU despite high "activity" — classic disk-swap signature, not real
  compute).
- **Fixed version** (current `run_full_blocking.py`) caps posting lists at
  300 entries DURING streaming (not just at the end), added explicit garbage
  collection, tuple-based iteration instead of `.iterrows()` for speed, and
  live memory checkpoint logging via Windows ctypes API (`GlobalMemoryStatusEx`
  / `GetProcessMemoryInfo`) every 500k pool records / 200k S1 records.
- Ran with `python run_full_blocking.py --country India` (smaller partition,
  ~883k S1 entities, run first as a safety test before attempting US or `all`).
- **This run was healthy** — confirmed via direct file-size monitoring
  (`train_candidate_pairs_India.tsv` grew steadily, ~27-36 KB/s sustained,
  memory stayed bounded, no thrashing signature this time).
- **BUT it was killed manually** partway through due to the laptop's limited
  RAM making a multi-hour run impractical to babysit/risky to leave running.
  It reached **52.4 MB of output (~21% of the estimated ~249MB total)**
  before being stopped.
- **`train_candidate_pairs_India.tsv` as it currently exists is INCOMPLETE
  and should be deleted, not resumed from** — it only has partial S1 coverage
  and would give wrong recall numbers if evaluated as-is.

### What needs to happen next (this is the actual task to pick up)

1. **Delete the partial file** before rerunning:
   ```powershell
   Remove-Item code\business_entity_resolution\src\train_candidate_pairs_India.tsv
   ```

2. **Run on a machine with more RAM** if possible. The fix (streaming cap)
   works correctly — the remaining issue is pure hardware headroom for
   a multi-hour run on the full dataset, not a code bug. If a beefier
   machine (more RAM, ideally 16GB+ free) is available, this should run
   without needing to be babysat.

3. **Command to run** (from `student_resource/` root, venv activated):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python code\business_entity_resolution\src\run_full_blocking.py --country India
   ```
   Then separately, after India completes:
   ```powershell
   python code\business_entity_resolution\src\run_full_blocking.py --country US
   ```
   (Deliberately run as two separate invocations, not `--country all`, so a
   crash/interruption in one partition doesn't lose progress on the other.)

4. **How to tell if it's healthy vs. stalled**, without guessing:
   - Watch the output file grow over a fixed time window:
     ```powershell
     Get-Item code\business_entity_resolution\src\train_candidate_pairs_India.tsv | Select-Object Length, LastWriteTime
     ```
     Run it, wait 5 minutes, run again. If size is growing, it's healthy —
     regardless of what the terminal/console shows (Antigravity's agent panel
     output can be hard to find/scroll; the file growth is the ground truth).
   - A rough throughput estimate we measured: **~27-36 KB/s** sustained once
     past the initial vectorizer/SVD-fitting phase. At that rate, India's
     estimated ~249MB total output implies roughly **1.5-2.5 hours** for
     India alone; US (about 50% more records) proportionally longer — plan
     for **roughly 5-7 hours total** for both partitions run sequentially.
     This is an estimate, not a guarantee — rate can vary by machine.
   - If system RAM climbs steadily toward 95%+ AND CPU usage on the Python
     process stays low (under ~20%) at the same time, that's the swap-
     thrashing signature — kill it and revisit the streaming cap logic
     (currently `STREAM_MAX_POSTING = 300` in `run_full_blocking.py`; could
     be lowered further, e.g. to 150-200, to trade a small amount of recall
     for meaningfully less memory pressure if this recurs on the new machine).

5. **Once each partition completes**, the script auto-runs a ground-truth
   recall evaluation and prints a full summary report (S1 count, avg
   candidates/entity, peak memory, recall %). **Paste/save that output** —
   it's the final full-scale confirmation that the 94.5%-ish recall from the
   10k-sample benchmarks holds at true scale, and it belongs in the
   methodology doc.

---

## 6. What's NOT Started Yet (the rest of the project)

In order, once full-scale candidate generation is confirmed working:

1. **Write `scorer.py`** — the official F_0.5 macro-average formula, matching
   spec exactly (all-or-nothing 1.0/0.0 for singletons, standard F-beta
   otherwise). This was discussed and drafted in conversation but not yet
   saved as an actual file in the repo — write it before anything else,
   everything downstream depends on measuring against the real metric.

2. **Write `make_split.py`** — stratified train/validation split of Source-1
   entities, stratified by country AND singleton-vs-matched, so validation
   mirrors the true distribution (5.6% singletons, ~60/40 US/India). Also
   discussed but not yet saved as a file.

3. **Feature engineering** for the matching model — per candidate pair,
   compute: Levenshtein/Jaro-Winkler name similarity, token-set Jaccard,
   TF-IDF cosine (char n-gram), address component overlap, country exact-match
   flag, and reuse the embedding cosine from the blocking ANN stage. NOT
   STARTED.

4. **Train the matching model** — planned approach is a LightGBM/XGBoost
   classifier on the engineered features (satisfies the MIT/Apache +
   parameter-count constraints trivially, and is well-suited to tabular
   similarity features). NOT STARTED.

5. **Threshold tuning for F_0.5** — sweep the decision threshold against the
   real F_0.5 macro-average (NOT accuracy, NOT F1, NOT the default 0.5 cutoff)
   to find the precision-favoring operating point. NOT STARTED.

6. **Held-out-country generalization test** — deliberately remove one training
   country (e.g. India) during a test run to confirm the pipeline doesn't
   silently rely on country-specific logic, since the real test set includes
   France (absent from training entirely). NOT STARTED — important for
   avoiding a nasty surprise at submission time.

7. **Generate `matching_results.tsv` and `candidate_pairs.tsv`** in `output/`,
   run `utils/validate_submission.py` against them before any leaderboard
   upload. NOT STARTED.

8. **Fill in `Documentation_template.md`** — much of the blocking section can
   be lifted near-verbatim from `blocking_failure_analysis.md`. NOT STARTED.

---

## 7. Environment Setup (for a fresh machine)

```powershell
cd path\to\student_resource
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn lightgbm rapidfuzz sentence-transformers faiss-cpu jellyfish tqdm
pip freeze > code\business_entity_resolution\requirements.txt
```

If PowerShell blocks venv activation:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 8. Git / GitHub Status

Repo has been pushed to GitHub already. `.gitignore` excludes `dataset/`,
`.venv/`, generated candidate TSVs, and log files — **do not commit the raw
dataset or any `train_candidate_pairs*.tsv` output**, they're large and
regenerable from code + data.

If cloning fresh on a new machine, you'll need to place the `dataset/`
folder back manually (it's provided by Amazon separately, not in git) and
recreate the venv per Section 7.

---

## 9. Key Files to Read First, In Order

1. This file (you just did)
2. `code/business_entity_resolution/notes/blocking_failure_analysis.md` —
   the full technical story of the blocking stage, with real examples
3. `.agent/skills/entity-resolution/SKILL.md` — the hard constraints, so
   your AI assistant (if using Antigravity or similar) doesn't violate them
4. `code/business_entity_resolution/src/blocking.py` — the actual current
   blocking logic (locked config, address-key + IDF fixes)
5. `code/business_entity_resolution/src/run_full_blocking.py` — the
   production runner, currently mid-task (Section 5 above)
