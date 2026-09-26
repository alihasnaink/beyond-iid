from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def as_numpy(values):
    return values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)


def threshold_from_validation(validation_scores, quantile=0.95):
    """95% of known validation samples are accepted when score <= threshold."""
    return float(np.quantile(as_numpy(validation_scores), quantile))


def known_accuracy(logits, labels):
    logits, labels = as_numpy(logits), as_numpy(labels)
    return float((logits.argmax(1) == labels).mean())


def auroc(known_scores, unknown_scores):
    known_scores, unknown_scores = as_numpy(known_scores), as_numpy(unknown_scores)
    target = np.r_[np.zeros(len(known_scores)), np.ones(len(unknown_scores))]
    return float(roc_auc_score(target, np.r_[known_scores, unknown_scores]))


def evaluate_unknownness(validation_scores, known_test_scores, near_scores, far_scores, threshold=None):
    validation_scores, known_test_scores = as_numpy(validation_scores), as_numpy(known_test_scores)
    near_scores, far_scores = as_numpy(near_scores), as_numpy(far_scores)
    threshold = threshold_from_validation(validation_scores) if threshold is None else float(threshold)
    all_unknown = np.r_[near_scores, far_scores]
    return {
        "threshold": threshold,
        "auroc_near": auroc(known_test_scores, near_scores),
        "auroc_far": auroc(known_test_scores, far_scores),
        "auroc_all": auroc(known_test_scores, all_unknown),
        "known_test_acceptance": float((known_test_scores <= threshold).mean()),
        "near_unknown_rejection": float((near_scores > threshold).mean()),
        "far_unknown_rejection": float((far_scores > threshold).mean()),
        "all_unknown_rejection": float((all_unknown > threshold).mean()),
        "near_fpr_at_95_tpr": float((near_scores <= threshold).mean()),
        "far_fpr_at_95_tpr": float((far_scores <= threshold).mean()),
        "all_fpr_at_95_tpr": float((all_unknown <= threshold).mean()),
    }
