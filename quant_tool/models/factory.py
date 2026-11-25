"""Model factories: four small pure-NumPy networks."""
from __future__ import annotations

import numpy as np

from .layers import (Conv2d, Flatten, Linear, MeanPool, ReLU,
                     SelfAttentionBlock, Sequential)


def build_mlp(in_dim, n_classes=2, hidden=(64, 32), seed=0):
    rng = np.random.default_rng(seed)
    layers, d = [], in_dim
    for h in hidden:
        layers += [Linear(d, h, rng), ReLU()]
        d = h
    layers.append(Linear(d, n_classes, rng))
    return Sequential(layers)


def build_cnn(n_classes=10, seed=0):
    """Conv(1->16,3x3) -> ReLU -> Conv(16->32,3x3) -> ReLU
    -> Flatten -> Linear(2048->64) -> ReLU -> Linear(64->10)."""
    rng = np.random.default_rng(seed)
    return Sequential([
        Conv2d(1, 16, 3, 3, rng, pad=1), ReLU(),
        Conv2d(16, 32, 3, 3, rng, pad=1), ReLU(),
        Flatten(),
        Linear(32 * 8 * 8, 64, rng), ReLU(),
        Linear(64, n_classes, rng),
    ])


def build_attention(n_classes=10, dim=32, seed=0):
    """Embed 8-d row tokens -> single-head self-attention -> mean pool
    -> FFN Linear(32->32)+ReLU -> Linear(32->10)."""
    rng = np.random.default_rng(seed)
    return Sequential([
        Linear(8, dim, rng),
        SelfAttentionBlock(dim, rng),
        MeanPool(),
        Linear(dim, dim, rng), ReLU(),
        Linear(dim, n_classes, rng),
    ])


FACTORIES = {
    "mlp_robust": lambda: build_mlp(61, 2, (64, 32), seed=1),
    "mlp_sensitive": lambda: build_mlp(61, 2, (64, 32), seed=2),
    "cnn_sensitive": lambda: build_cnn(10, seed=3),
    "attn_sensitive": lambda: build_attention(10, 32, seed=4),
}
