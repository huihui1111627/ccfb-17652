"""Classification metrics with explicit per-class breakdown."""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(y_true, y_pred, n_classes):
    """Per-class precision / recall / f1 / support (one-vs-rest)."""
    cm = confusion(y_true, y_pred, n_classes)
    rows = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = int(cm[c, :].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": c, "precision": float(prec),
                     "recall": float(rec), "f1": float(f1),
                     "support": support})
    return rows, cm


def evaluate(y_true, logits, key_class=None):
    y_pred = logits.argmax(1)
    n_classes = int(logits.shape[1])
    rows, cm = per_class_metrics(y_true, y_pred, n_classes)
    out = {
        "accuracy": float((y_pred == y_true).mean()),
        "per_class": rows,
        "confusion": cm.tolist(),
    }
    if key_class is not None:
        out["key_recall"] = rows[key_class]["recall"]
        out["key_precision"] = rows[key_class]["precision"]
        out["key_f1"] = rows[key_class]["f1"]
        out["key_support"] = rows[key_class]["support"]
    return out
