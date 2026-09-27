---
name: entity-resolution
description: Hard constraints for the Amazon business entity resolution challenge — apply whenever writing, reviewing, or reasoning about blocking, matching, feature engineering, or output generation code in this repo.
---

# Amazon Entity Resolution Challenge — Project Constraints

## Non-negotiable rules

1. **No external data lookups, ever.** No geocoding APIs, no business registry
   lookups, no calling any internet service from pipeline code. All normalization
   and similarity logic must run entirely on the provided TSV files. If you catch
   yourself suggesting `requests.get(...)` to enrich an address or name, stop —
   that's disqualifying under the challenge's fair-play rules.

2. **Never hardcode country-specific logic.** Training data only has US and India;
   the test set adds France. Any regex, parser, or rule that only branches on
   `{"US", "India"}` will silently fail on French records. Treat `country` as an
   open string field. Prefer generic, language-agnostic string similarity
   (char n-grams, edit distance) over country-specific address grammars.

3. **Optimize for macro F_0.5 per Source-1 entity, not accuracy or F1.**
   Precision is weighted 2x over recall. When tuning a classification threshold,
   always sweep against the actual F_0.5 formula (see scorer.py), never default
   to 0.5 or optimize F1/accuracy as a proxy.

4. **Singletons matter as much as matches.** A Source-1 entity with zero true
   matches scores 1.0 for a correctly empty prediction and 0.0 for any false
   match. Never force a low-confidence match just to produce output. When no
   candidate clears the tuned threshold, emit an empty match list.

5. **Output format invariants — enforce these in code, not just at export time:**
   - Exactly one row per Source-1 test entity, no missing, no duplicates.
   - `matched_entity_ids` / `candidate_entity_ids`: empty string when no matches,
     never `NaN`/`None`/literal string `"nan"`.
   - No duplicate IDs within a single row's ID list.
   - IDs referenced must be S2-/S3- prefixed only — never S1- (no self-matches).
   - `candidate_pairs.tsv` must be a superset of `matching_results.tsv` per entity
     — every ID in the final matches must also appear in that entity's candidates.

6. **Candidate set size is graded, not just recall.** Smaller candidate sets per
   Source-1 entity score better in the final review, but only if recall doesn't
   collapse. When tuning blocking (top-K, similarity thresholds), always report
   the recall@K vs. candidate-count tradeoff — don't just maximize recall by
   inflating K.

7. **Final model constraints:** MIT or Apache-2.0 licensed only, ≤8B parameters.
   Prefer classical ML (gradient boosted trees on engineered features) or small
   open embedding models (e.g. MiniLM-class, Apache 2.0) over large hosted LLMs
   for the actual matching/scoring step.

## When reviewing or writing code in this repo

- Flag any dtype inference risk: entity IDs and postal-code-like strings must be
  read as `dtype=str`, never allowed to become int/float via pandas defaults.
- Flag any operation that loads all four ~500MB TSVs into memory simultaneously
  without a clear reason — prefer chunked/streaming reads where the join pattern
  allows it.
- Flag any file I/O that doesn't specify `encoding="utf-8"` explicitly.
- Before suggesting a "quick" full pairwise comparison between two sources for
  debugging, warn about the ~1.7M-entity scale of the real test set.