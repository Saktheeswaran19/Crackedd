"""
Train a LightGBM matching model on engineered features.

Run from student_resource/ root:
    python code\\business_entity_resolution\\src\\train_model.py
"""

import os
import sys
import argparse
import time
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from scorer import f_beta_score, macro_f_beta

SRC_DIR = os.path.dirname(__file__)
TRAIN_FEATURES = os.path.join(SRC_DIR, "train_features.tsv")
VAL_FEATURES = os.path.join(SRC_DIR, "val_features.tsv")
MODEL_PATH = os.path.join(SRC_DIR, "model.lgb")
CONFIG_PATH = os.path.join(SRC_DIR, "model_config.json")

# Features to use (must match feature_engineering.py output)
FEATURE_COLS = [
    'name_levenshtein', 'name_partial_ratio', 'name_token_sort_ratio',
    'name_token_set_ratio', 'name_jaro_winkler', 'name_jaccard', 'name_len_ratio',
    'addr_levenshtein', 'addr_token_sort_ratio', 'addr_token_overlap',
    'addr_len_ratio', 'addr_number_match',
    'country_exact', 'combined_jaccard', 'num_shared_blocking_keys'
]


def compute_macro_f05(val_df, threshold):
    """Compute macro F_0.5 at a given threshold."""
    s1_data = defaultdict(list)
    for row in val_df[['s1_id', 'candidate_id', 'pred_prob', 'label']].itertuples(index=False):
        s1_data[row[0]].append((row[1], float(row[2]), int(row[3])))

    scores = []
    for s1_id, pairs in s1_data.items():
        true_set = set(cid for cid, _, lbl in pairs if lbl == 1)
        pred_set = set(cid for cid, prob, _ in pairs if prob >= threshold)
        scores.append(f_beta_score(pred_set, true_set, beta=0.5))
    return float(np.mean(scores))


def tune_threshold(val_df):
    """Sweep thresholds and find the one maximizing macro F_0.5."""
    print("\n" + "="*60)
    print("THRESHOLD TUNING (optimizing macro F_0.5)")
    print("="*60)

    # Pre-group validation pairs by s1_id once for ultra-fast threshold sweeps
    print("Pre-grouping validation pairs for fast sweep...")
    from collections import defaultdict
    s1_data = defaultdict(list)
    for row in val_df[['s1_id', 'candidate_id', 'pred_prob', 'label']].itertuples(index=False):
        s1_data[row[0]].append((row[1], float(row[2]), int(row[3])))

    # Pre-extract true matches per S1 entity
    gt_per_s1 = {}
    for s1_id, pairs in s1_data.items():
        gt_per_s1[s1_id] = set(cid for cid, _, lbl in pairs if lbl == 1)

    def eval_at_threshold(thresh):
        scores = []
        for s1_id, pairs in s1_data.items():
            true_set = gt_per_s1[s1_id]
            pred_set = set(cid for cid, prob, _ in pairs if prob >= thresh)
            scores.append(f_beta_score(pred_set, true_set, beta=0.5))
        return float(np.mean(scores))

    best_thresh = 0.5
    best_f05 = 0.0
    results = []

    for thresh_int in range(10, 96, 2):  # 0.10, 0.12, ..., 0.94
        thresh = thresh_int / 100.0
        f05 = eval_at_threshold(thresh)
        results.append((thresh, f05))
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    # Fine-grained search around best
    for offset in range(-10, 11):
        thresh = best_thresh + offset * 0.005
        if thresh < 0.05 or thresh > 0.99:
            continue
        f05 = eval_at_threshold(thresh)
        results.append((thresh, f05))
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    print(f"\nBest threshold: {best_thresh:.3f}")
    print(f"Best macro F_0.5: {best_f05:.4f}")

    # Print a summary of the sweep
    seen_thresh = set()
    unique_results = []
    for t, s in results:
        t_round = round(t, 3)
        if t_round not in seen_thresh:
            seen_thresh.add(t_round)
            unique_results.append((t_round, s))
    unique_results.sort(key=lambda x: x[0])

    print("\nThreshold sweep summary:")
    for thresh, f05 in unique_results[::2]:  # step every other for concise output
        marker = " <<<" if abs(thresh - best_thresh) < 0.003 else ""
        print(f"  thresh={thresh:.3f}  macro F_0.5={f05:.4f}{marker}")

    return best_thresh, best_f05


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM matching model.")
    parser.add_argument("--train-features", default=TRAIN_FEATURES)
    parser.add_argument("--val-features", default=VAL_FEATURES)
    parser.add_argument("--model-output", default=MODEL_PATH)
    parser.add_argument("--skip-tune", action='store_true', help="Skip threshold tuning")
    args = parser.parse_args()

    t_start = time.time()

    # Load features
    print("Loading training features...")
    train_df = pd.read_csv(args.train_features, sep='\t')
    print(f"Train: {len(train_df):,} pairs, {train_df['label'].sum():,} positive")

    print("Loading validation features...")
    val_df = pd.read_csv(args.val_features, sep='\t')
    print(f"Val: {len(val_df):,} pairs, {val_df['label'].sum():,} positive")

    # Prepare training data
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['label'].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df['label'].values

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(1, n_pos)
    print(f"\nClass balance: pos={n_pos:,} neg={n_neg:,} scale_pos_weight={scale:.1f}")

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data)

    print("\nTraining LightGBM...")
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50)
    ]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks,
    )

    print(f"\nBest iteration: {model.best_iteration}")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])
    print("\nFeature Importance (gain):")
    for fname, imp in feat_imp:
        print(f"  {fname:30s} {imp:12.1f}")

    # Save model
    model.save_model(args.model_output)
    print(f"\nModel saved → {args.model_output}")

    # Predict on validation
    val_df = val_df.copy()
    val_df['pred_prob'] = model.predict(X_val)

    # Classification report at default 0.5 threshold
    y_pred_05 = (val_df['pred_prob'] >= 0.5).astype(int)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (threshold=0.5)")
    print("="*60)
    print(classification_report(y_val, y_pred_05, target_names=['non-match', 'match']))

    f05_default = compute_macro_f05(val_df, 0.5)
    print(f"Macro F_0.5 at threshold 0.5: {f05_default:.4f}")

    # Threshold tuning
    if not args.skip_tune:
        best_thresh, best_f05 = tune_threshold(val_df)
    else:
        best_thresh = 0.5
        best_f05 = f05_default

    # Save config
    config = {
        'best_threshold': best_thresh,
        'best_macro_f05': best_f05,
        'best_iteration': model.best_iteration,
        'feature_cols': FEATURE_COLS,
        'feature_importance': {f: float(i) for f, i in feat_imp},
        'params': params,
    }
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved → {CONFIG_PATH}")

    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model: {args.model_output}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best threshold: {best_thresh:.3f}")
    print(f"Macro F_0.5 (val): {best_f05:.4f}")
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
