"""
Metrics computation for DR-TBAC-ZT++ evaluation.

Covers:
- Binary classification metrics (trust prediction, PERMIT/DENY correctness)
- RL training metrics (episode return, convergence tracking)
- Federated learning metrics (global model accuracy, communication cost)
- System metrics (throughput, latency)
"""

import time
from collections import deque
from typing import Sequence

import numpy as np

# Optional: sklearn for richer classification metrics
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
) -> dict:
    """
    Compute standard binary-classification metrics.

    Parameters
    ----------
    y_true : array-like of int    Ground-truth labels (0/1).
    y_pred : array-like of int    Predicted labels (0/1).
    y_prob : array-like of float  Predicted probabilities for class 1 (optional).

    Returns
    -------
    dict with accuracy, precision, recall, f1, auc (if y_prob provided),
    confusion_matrix, fpr (false-positive rate), fnr (false-negative rate).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if _SKLEARN:
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        cm   = confusion_matrix(y_true, y_pred).tolist()
    else:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        acc  = (tp + tn) / max(len(y_true), 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        cm   = [[tn, fp], [fn, tp]]

    results = {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "confusion_matrix": cm,
    }

    # False positive / negative rates from confusion matrix
    tn_v, fp_v, fn_v, tp_v = (
        cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    )
    results["fpr"] = round(fp_v / max(fp_v + tn_v, 1), 4)
    results["fnr"] = round(fn_v / max(fn_v + tp_v, 1), 4)

    if y_prob is not None and _SKLEARN:
        try:
            results["auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            results["auc"] = None

    return results


# ---------------------------------------------------------------------------
# RL Training Metrics
# ---------------------------------------------------------------------------

class EpisodeTracker:
    """
    Rolling statistics tracker for RL training episodes.

    Parameters
    ----------
    window : int   Size of the sliding window for moving averages.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.returns:  deque[float] = deque(maxlen=window)
        self.lengths:  deque[int]   = deque(maxlen=window)
        self.losses:   deque[float] = deque(maxlen=window)
        self.epsilons: list[float]  = []
        self.episode   = 0

    def record(
        self,
        episode_return: float,
        episode_length: int,
        loss:           float | None = None,
        epsilon:        float | None = None,
    ) -> None:
        """Record statistics for one completed episode."""
        self.episode += 1
        self.returns.append(episode_return)
        self.lengths.append(episode_length)
        if loss is not None:
            self.losses.append(loss)
        if epsilon is not None:
            self.epsilons.append(epsilon)

    def summary(self) -> dict:
        """Return a dict of rolling averages for the current window."""
        r = np.array(self.returns) if self.returns else np.array([0.0])
        l = np.array(self.lengths) if self.lengths else np.array([0])
        q = np.array(self.losses)  if self.losses  else np.array([float("nan")])
        return {
            "episode":        self.episode,
            "mean_return":    round(float(r.mean()), 4),
            "max_return":     round(float(r.max()),  4),
            "min_return":     round(float(r.min()),  4),
            "mean_length":    round(float(l.mean()), 2),
            "mean_loss":      round(float(q.mean()), 6) if not np.isnan(q).all() else None,
            "last_epsilon":   round(self.epsilons[-1], 4) if self.epsilons else None,
        }


# ---------------------------------------------------------------------------
# Federated Learning Metrics
# ---------------------------------------------------------------------------

def aggregate_fl_metrics(client_metrics: list[dict]) -> dict:
    """
    Aggregate per-client FL round metrics into global statistics.

    Parameters
    ----------
    client_metrics : list of dicts, each with keys:
        ``num_examples``, ``accuracy``, ``loss`` (and optionally others).

    Returns
    -------
    Weighted-average global metrics dict.
    """
    if not client_metrics:
        return {}

    total_examples = sum(m.get("num_examples", 1) for m in client_metrics)
    weighted_acc   = sum(
        m.get("accuracy", 0) * m.get("num_examples", 1) for m in client_metrics
    ) / max(total_examples, 1)
    weighted_loss  = sum(
        m.get("loss", 0) * m.get("num_examples", 1) for m in client_metrics
    ) / max(total_examples, 1)

    return {
        "num_clients":       len(client_metrics),
        "total_examples":    total_examples,
        "global_accuracy":   round(weighted_acc,  4),
        "global_loss":       round(weighted_loss, 6),
    }


# ---------------------------------------------------------------------------
# System / Throughput Metrics
# ---------------------------------------------------------------------------

class ThroughputMeter:
    """Measures request throughput and latency over a sliding window."""

    def __init__(self, window: int = 1000):
        self._latencies: deque[float] = deque(maxlen=window)
        self._count = 0
        self._start = time.monotonic()

    def record(self, latency_seconds: float) -> None:
        """Record one request with its latency (seconds)."""
        self._latencies.append(latency_seconds)
        self._count += 1

    def stats(self) -> dict:
        elapsed = time.monotonic() - self._start
        lats = np.array(self._latencies) * 1000  # convert to ms
        return {
            "total_requests":  self._count,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_rps":  round(self._count / max(elapsed, 1e-9), 2),
            "mean_latency_ms": round(float(lats.mean()), 3) if len(lats) else 0.0,
            "p50_latency_ms":  round(float(np.percentile(lats, 50)), 3) if len(lats) else 0.0,
            "p95_latency_ms":  round(float(np.percentile(lats, 95)), 3) if len(lats) else 0.0,
            "p99_latency_ms":  round(float(np.percentile(lats, 99)), 3) if len(lats) else 0.0,
        }
