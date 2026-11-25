"""Calibration-set construction.

The ranges that fix every quantization grid are estimated from a
calibration set, so a non-representative set directly biases the model.
This module builds deliberately different calibration sets whose effects
the analysis compares head to head.
"""
from __future__ import annotations

import numpy as np


def calibration_variants(X, y, key_class, n=128, seed=0):
    """Return named calibration subsets of X.

    variants:
      stratified  — class proportions match the training data (recommended)
      head        — first n rows, no shuffling; can be non-representative
      key_heavy   — half the rows are the rare key class (range skewed
                    toward key-class activations)
      key_free    — key class removed entirely (its activation magnitudes
                    are unseen at calibration -> clipping at inference)
      tiny        — only 16 rows, noisy range estimates
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    key = idx[y == key_class]
    other = idx[y != key_class]
    out = {}

    # stratified
    rng.shuffle(key); rng.shuffle(other)
    p_key = len(key) / len(idx)
    n_key = max(1, int(round(n * p_key)))
    take_key = np.tile(key, n_key // len(key) + 1)[:n_key] if len(key) else key
    take_other = np.tile(other, (n - n_key) // len(other) + 1)[:n - n_key]
    s = np.concatenate([take_key, take_other])
    rng.shuffle(s)
    out["stratified"] = X[s]

    out["head"] = X[:n]

    half = n // 2
    kh = np.tile(key, half // len(key) + 1)[:half] if len(key) else key
    oh = np.tile(other, (n - half) // len(other) + 1)[:n - half]
    s2 = np.concatenate([kh, oh]); rng.shuffle(s2)
    out["key_heavy"] = X[s2]

    kf = rng.permutation(other)[:min(n, len(other))]
    out["key_free"] = X[kf]
    out["tiny"] = X[rng.choice(len(X), size=min(16, len(X)), replace=False)]
    return out
