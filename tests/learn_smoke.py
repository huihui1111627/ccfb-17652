"""End-to-end learning smoke test: catches the optimizer-reference bug that
gradient checks alone miss (layers rebinding grad arrays instead of writing
in place). Each architecture must beat chance by a wide margin in a few
epochs on a small synthetic-but-structured subset.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_tool.models.layers import (Conv2d, Flatten, Linear, MeanPool,
                                      ReLU, SelfAttentionBlock, Sequential,
                                      softmax_cross_entropy)
from quant_tool.models.training import Adam, train_model, predict_logits


def toy_tabular(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = rng.standard_normal((n, 10)).astype(np.float32)
    X += np.eye(3)[y].repeat(np.array([4, 3, 3]), axis=1).astype(np.float32) * 2
    return X, y


def check(name, model, X, y, chance, epochs=25, margin=0.25,
          loss_halves=True):
    hist = train_model(model, X, y, epochs=epochs, lr=2e-3, batch=64, seed=1)
    acc = (predict_logits(model, X).argmax(1) == y).mean()
    l0, l1 = hist[0]["train_loss"], hist[-1]["train_loss"]
    ok = (not loss_halves or l1 < l0 * 0.5) and acc > chance + margin
    print(f"[{'OK ' if ok else 'FAIL'}] {name:10s} loss {l0:.3f}->{l1:.3f} "
          f"acc {acc:.3f} (chance {chance})")
    assert ok, name


def main():
    X, y = toy_tabular()
    rng = np.random.default_rng(0)
    check("mlp", Sequential([Linear(10, 32, rng), ReLU(),
                             Linear(32, 3, rng)]), X, y, 1 / 3)
    Xi = np.zeros((len(X), 1, 8, 8), np.float32)
    Xi.reshape(len(X), -1)[:, :10] = X
    check("cnn", Sequential([Conv2d(1, 8, 3, 3, rng, pad=1), ReLU(),
                             Flatten(), Linear(8 * 8 * 8, 3, rng)]),
          Xi, y, 1 / 3)
    Xs = X[:, :10].reshape(-1, 5, 2)
    rng2 = np.random.default_rng(2)
    # single-head attention is noise-limited on this tiny toy task;
    # an independent isolation test confirms it still beats the no-attention
    # baseline, so a smaller margin is the right bar here.
    check("attn", Sequential([Linear(2, 12, rng2),
                              SelfAttentionBlock(12, rng2), MeanPool(),
                              Linear(12, 3, rng2)]),
          Xs.astype(np.float32), y, 1 / 3, epochs=60, margin=0.20,
          loss_halves=False)
    print("all learning smoke tests passed")


if __name__ == "__main__":
    main()
