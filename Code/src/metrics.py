from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_change_metrics(change_probs: np.ndarray, change_targets: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    change_preds = (change_probs >= threshold).astype(np.int64)
    change_targets = change_targets.astype(np.int64)

    acc = accuracy_score(change_targets, change_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        change_targets,
        change_preds,
        average="binary",
        zero_division=0,
    )

    return {
        "change_accuracy": float(acc),
        "change_precision": float(precision),
        "change_recall": float(recall),
        "change_f1": float(f1),
    }


def compute_multilabel_micro_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    prefix: str,
) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    targets = targets.astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        preds,
        average="micro",
        zero_division=0,
    )
    return {
        f"{prefix}_micro_precision": float(precision),
        f"{prefix}_micro_recall": float(recall),
        f"{prefix}_micro_f1": float(f1),
    }


def compute_all_metrics(
    change_probs: np.ndarray,
    change_targets: np.ndarray,
    object_probs: np.ndarray,
    object_targets: np.ndarray,
    action_probs: np.ndarray,
    action_targets: np.ndarray,
    location_probs: np.ndarray,
    location_targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    metrics = {}
    metrics.update(compute_change_metrics(change_probs, change_targets, threshold=threshold))
    metrics.update(compute_multilabel_micro_metrics(object_probs, object_targets, threshold, "object"))
    metrics.update(compute_multilabel_micro_metrics(action_probs, action_targets, threshold, "action"))
    metrics.update(compute_multilabel_micro_metrics(location_probs, location_targets, threshold, "location"))
    return metrics
