"""Multi-label evaluation metrics (numpy and scikit-learn only).

Conventions:
  * ``y_true`` and ``y_pred`` are binary matrices of shape (N, K).
  * ``y_prob`` holds sigmoid probabilities of shape (N, K).
  * Classes with a single observed value in ``y_true`` have undefined AUROC and
    are reported as NaN and excluded from macro averages.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def class_column_slug(name: str) -> str:
    """Turn a class name into a safe column suffix, e.g. ``"No Finding" -> "no_finding"``."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _as_matrix(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (samples x classes), got shape {arr.shape}")
    return arr


def _broadcast_thresholds(thresholds: float | Sequence[float] | np.ndarray, num_classes: int) -> np.ndarray:
    thr = np.asarray(thresholds, dtype=float)
    return np.broadcast_to(thr, (num_classes,)).copy()


def binarize(probs: np.ndarray, thresholds: float | Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert probabilities to 0/1 predictions using a scalar or per-class threshold."""
    probs = _as_matrix(probs, "probs")
    thr = _broadcast_thresholds(thresholds, probs.shape[1])
    return (probs >= thr).astype(np.int64)


def per_class_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    y_true = _as_matrix(y_true, "y_true")
    y_prob = _as_matrix(y_prob, "y_prob")
    result = np.full(y_true.shape[1], np.nan)
    for k in range(y_true.shape[1]):
        column = y_true[:, k]
        if column.min() == column.max():
            continue
        result[k] = roc_auc_score(column, y_prob[:, k])
    return result


def per_class_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    y_true = _as_matrix(y_true, "y_true")
    y_prob = _as_matrix(y_prob, "y_prob")
    result = np.full(y_true.shape[1], np.nan)
    for k in range(y_true.shape[1]):
        column = y_true[:, k]
        if column.sum() == 0:
            continue
        result[k] = average_precision_score(column, y_prob[:, k])
    return result


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean per-class AUROC over classes where it is defined."""
    values = per_class_auroc(y_true, y_prob)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of per-class F1 scores (zero division counted as 0)."""
    return float(f1_score(_as_matrix(y_true, "y_true"), _as_matrix(y_pred, "y_pred"), average="macro", zero_division=0))


def hamming_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of individual label decisions that are correct (per-label accuracy)."""
    return float((_as_matrix(y_true, "y_true") == _as_matrix(y_pred, "y_pred")).mean())


def exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples whose complete label set is predicted exactly (strict)."""
    return float(np.all(_as_matrix(y_true, "y_true") == _as_matrix(y_pred, "y_pred"), axis=1).mean())


def per_class_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: Sequence[str],
    thresholds: float | Sequence[float] | np.ndarray = 0.5,
) -> pd.DataFrame:
    """Per-class support, confusion counts, precision, recall, specificity, F1, AUROC and AP."""
    y_true = _as_matrix(y_true, "y_true").astype(np.int64)
    y_prob = _as_matrix(y_prob, "y_prob")
    if y_true.shape != y_prob.shape or y_true.shape[1] != len(classes):
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_prob {y_prob.shape}, classes {len(classes)}")

    thr = _broadcast_thresholds(thresholds, len(classes))
    y_pred = binarize(y_prob, thr)
    confusion = multilabel_confusion_matrix(y_true, y_pred)  # shape (K, 2, 2): [[tn, fp], [fn, tp]]
    auroc = per_class_auroc(y_true, y_prob)
    average_precision = per_class_average_precision(y_true, y_prob)

    rows = []
    for k, cls in enumerate(classes):
        tn, fp, fn, tp = (int(v) for v in confusion[k].ravel())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "class": cls,
            "support": int(y_true[:, k].sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "threshold": float(thr[k]),
            "auroc": float(auroc[k]),
            "average_precision": float(average_precision[k]),
        })
    return pd.DataFrame(rows)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, default: float = 0.5) -> np.ndarray:
    """Choose each class's decision threshold to maximise F1 on the supplied data.

    Call this with the **validation** split only. Tuning on the test split would leak
    test information into the reported results. Classes with no positives or no
    negatives keep ``default``.
    """
    y_true = _as_matrix(y_true, "y_true").astype(np.int64)
    y_prob = _as_matrix(y_prob, "y_prob")
    thresholds = np.full(y_true.shape[1], float(default))
    for k in range(y_true.shape[1]):
        column = y_true[:, k]
        if column.sum() == 0 or column.sum() == len(column):
            continue
        precision, recall, cut = precision_recall_curve(column, y_prob[:, k])
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        # precision/recall have one more entry than the thresholds array
        best = int(np.nanargmax(f1[:-1]))
        thresholds[k] = float(cut[best])
    return thresholds


def rank_experiments(
    results: pd.DataFrame,
    baseline_id: str = "E0",
    max_class_auroc_drop: float = 0.02,
) -> pd.DataFrame:
    """Rank experiments on validation criteria, guarding against class-specific degradation.

    Expected columns: ``experiment_id``, ``val_macro_auroc``, ``val_macro_f1_tuned``,
    ``val_loss``, and one ``auroc_<class_slug>`` column per class.

    Ranking order:
      1. ``eligible``: no class AUROC falls more than ``max_class_auroc_drop`` below the baseline.
      2. Higher validation macro AUROC (threshold-free, robust to calibration).
      3. Higher validation macro F1 at tuned thresholds (informational, optimistic on validation).
      4. Lower validation loss.
    """
    if "experiment_id" not in results.columns:
        raise ValueError("results must contain an 'experiment_id' column")
    baseline = results.loc[results["experiment_id"] == baseline_id]
    if baseline.empty:
        raise ValueError(f"Baseline experiment '{baseline_id}' not found in results")

    df = results.copy()
    auroc_cols = [c for c in df.columns if c.startswith("auroc_")]
    if not auroc_cols:
        raise ValueError("results must contain per-class 'auroc_<class>' columns")
    baseline_values = baseline[auroc_cols].iloc[0].to_numpy(dtype=float)
    drops = baseline_values[None, :] - df[auroc_cols].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        df["max_class_auroc_drop"] = np.where(np.isfinite(drops).any(axis=1), np.nanmax(drops, axis=1), np.nan)
    df["eligible"] = df["max_class_auroc_drop"] <= max_class_auroc_drop

    df = df.sort_values(
        by=["eligible", "val_macro_auroc", "val_macro_f1_tuned", "val_loss"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df
