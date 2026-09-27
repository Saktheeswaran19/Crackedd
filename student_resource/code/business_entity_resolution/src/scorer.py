"""F_0.5 macro-average scorer, matching the official spec exactly."""
import numpy as np

def f_beta_score(pred_set: set, true_set: set, beta: float = 0.5) -> float:
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set) if true_set else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)

def macro_f_beta(predictions: dict, ground_truth: dict, beta: float = 0.5) -> float:
    """predictions, ground_truth: {source1_id: set(matched_ids)}"""
    scores = [
        f_beta_score(predictions.get(s1_id, set()), true_set, beta)
        for s1_id, true_set in ground_truth.items()
    ]
    return float(np.mean(scores))