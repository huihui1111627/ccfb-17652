"""Numerical gradient checks for every layer type."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_tool.models.layers import (Conv2d, Linear, ReLU, Flatten,
                                      SelfAttentionBlock, MeanPool,
                                      Sequential, softmax_cross_entropy)


def check(name, model, x, y, eps=1e-3, tol=2e-2):
    logits = model.forward(x, train=True)
    loss, dlog = softmax_cross_entropy(logits, y)
    model.backward(dlog)
    params, grads = model.all_params_grads()
    worst = 0.0
    for p, g in zip(params, grads):
        # check up to 6 random entries per tensor
        rng = np.random.default_rng(0)
        idx = np.stack(np.where(np.ones_like(p))).T
        sel = rng.choice(len(idx), size=min(6, len(idx)), replace=False)
        for ii in idx[sel]:
            ii = tuple(ii)
            old = p[ii]
            p[ii] = old + eps
            lp, _ = softmax_cross_entropy(model.forward(x, train=False), y)
            p[ii] = old - eps
            lm, _ = softmax_cross_entropy(model.forward(x, train=False), y)
            p[ii] = old
            num = (lp - lm) / (2 * eps)
            err = abs(num - g[ii]) / max(1.0, abs(num), abs(g[ii]))
            worst = max(worst, err)
    status = "OK " if worst < tol else "FAIL"
    print(f"[{status}] {name:18s} worst relative grad err = {worst:.2e}")
    assert worst < tol, name


def main():
    rng = np.random.default_rng(0)
    # MLP
    mlp = Sequential([Linear(8, 12, rng), ReLU(), Linear(12, 5, rng)])
    check("mlp", mlp, rng.standard_normal((7, 8)).astype(np.float32),
          rng.integers(0, 5, 7))
    # CNN
    cnn = Sequential([Conv2d(1, 4, 3, 3, rng, pad=1), ReLU(),
                      Conv2d(4, 6, 3, 3, rng, pad=1), ReLU(),
                      Flatten(), Linear(6 * 8 * 8, 5, rng)])
    check("cnn", cnn, rng.standard_normal((5, 1, 8, 8)).astype(np.float32),
          rng.integers(0, 5, 5), eps=2e-3, tol=3e-2)
    # Attention
    attn = Sequential([Linear(8, 10, rng), SelfAttentionBlock(10, rng),
                       MeanPool(), Linear(10, 5, rng)])
    check("attention", attn,
          rng.standard_normal((6, 4, 8)).astype(np.float32),
          rng.integers(0, 5, 6), eps=2e-3, tol=3e-2)
    print("all gradient checks passed")


if __name__ == "__main__":
    main()
