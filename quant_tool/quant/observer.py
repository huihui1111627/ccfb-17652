"""Calibration observers: how the numeric range of each tensor is chosen.

The range decision is the single most important PTQ choice, so it is kept
explicit and swappable:

* MinMaxObserver      — full [min, max] of the calibration tensors.
  No clipping; one giant activation can waste most of the grid.
* PercentileObserver — clip to central (lo, 100-lo)% percentiles.
  Trades a little clipping error for a much tighter grid; used to show how
  a poorly chosen calibration sample (or an outlier-heavy one) changes the
  result.

Observers only collect statistics; the actual grid mapping lives in
quant/kernel.py so the integer path is independent of how ranges were set.
"""
from __future__ import annotations

import numpy as np


class MinMaxObserver:
    method = "minmax"

    def __init__(self):
        self.vmin = np.inf
        self.vmax = -np.inf

    def observe(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.vmin = min(self.vmin, float(x.min()))
        self.vmax = max(self.vmax, float(x.max()))

    def range_(self):
        if not np.isfinite(self.vmin):
            return -1.0, 1.0
        return self.vmin, self.vmax


class PercentileObserver:
    method = "percentile"

    def __init__(self, clip_percentile=99.9):
        self.clip = clip_percentile
        self._vals = []

    def observe(self, x):
        # subsample to keep memory bounded
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size > 20000:
            rng = np.random.default_rng(0)
            x = rng.choice(x, 20000, replace=False)
        self._vals.append(x)

    def range_(self):
        vals = np.concatenate(self._vals) if self._vals else np.array([-1, 1])
        lo = np.percentile(vals, 100.0 - self.clip)
        hi = np.percentile(vals, self.clip)
        if hi <= lo:
            hi = lo + 1e-6
        return float(lo), float(hi)


def make_observer(method: str):
    if method == "minmax":
        return MinMaxObserver()
    if method.startswith("pct"):
        # "pct99.9" -> PercentileObserver(99.9)
        p = float(method[3:]) if len(method) > 3 else 99.9
        return PercentileObserver(p)
    raise ValueError(f"unknown observer method {method!r}")
