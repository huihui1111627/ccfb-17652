"""Adam trainer + FP-equivalent outlier-channel injection.

Outlier injection principle (provably exact in floating point)
--------------------------------------------------------------
For a Linear/Conv layer L1 feeding (optionally through ReLU) a later
Linear L2, pick output channel c and a positive scale s:

    W1[c] *= s,  b1[c] *= s        # wide outlier channel at L1 output
    W2[:, c] /= s                  # inverse scale at L2's input column

ReLU between the two is exact for positive s:
    relu(s * x) = s * relu(x)  for every x when s > 0.
For a Conv layer, channel c covers a contiguous H*W block of the flattened
Linear input, so the same rule applies column-block-wise.

The transformed network computes bit-for-bit (up to floating-point
rounding) the SAME function; ``verify_fp_equivalence`` asserts this on real
data. Integer quantization, however, sees per-channel ranges s times wider
for the picked channels — the outlier-drags-down-a-layer failure mode the
toolkit is built to detect.

Attention blocks contain softmax + residual + LayerNorm, for which no
channel-wise exact scaling exists; the attention model therefore receives
its outlier channels at the feed-forward Linear pair after the block,
while the block's own quantization sensitivity is measured directly by the
layer-wise analysis.
"""
from __future__ import annotations

import numpy as np

from .layers import Sequential, softmax_cross_entropy


class Adam:
    def __init__(self, params, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8,
                 weight_decay=5e-5):
        self.params, self.grads = params, grads
        self.lr, self.b1, self.b2, self.eps, self.wd = (
            lr, b1, b2, eps, weight_decay)
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.b1, self.b2
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        for p, g, m, v in zip(self.params, self.grads, self.m, self.v):
            if self.wd:
                g = g + self.wd * p
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            p -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)

    def zero_grad(self):
        for g in self.grads:
            g.fill(0.0)


def predict_logits(model: Sequential, X, batch=256):
    outs = []
    for i in range(0, len(X), batch):
        outs.append(model.forward(X[i:i + batch], train=False))
    return np.concatenate(outs, axis=0)


def accuracy(model, X, y, batch=256):
    return float((predict_logits(model, X, batch).argmax(1) == y).mean())


def train_model(model: Sequential, X, y, epochs=60, lr=1e-3, batch=64,
                seed=0, X_val=None, y_val=None, verbose=False):
    rng = np.random.default_rng(seed)
    params, grads = model.all_params_grads()
    opt = Adam(params, grads, lr=lr)
    n = len(X)
    history = []
    for ep in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            logits = model.forward(X[idx], train=True)
            loss, dlog = softmax_cross_entropy(logits, y[idx])
            model.backward(dlog)
            opt.step()
            total += loss * len(idx)
        row = {"epoch": ep, "train_loss": total / n}
        if X_val is not None:
            row["val_acc"] = accuracy(model, X_val, y_val)
        history.append(row)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d} loss {row['train_loss']:.4f}"
                  + (f" val_acc {row['val_acc']:.4f}" if X_val is not None
                     else ""))
    return history


def _next_linear(model, start_idx):
    for l in model.layers[start_idx:]:
        if l.name == "linear":
            return l
    raise ValueError("no Linear found after index %d" % start_idx)


def inject_linear_outliers(model, layer_idx, channels, scale=8.0):
    """Exact outlier pair on Linear ``layer_idx`` -> next Linear."""
    l1 = model.layers[layer_idx]
    assert l1.name == "linear" and scale > 0
    l2 = _next_linear(model, layer_idx + 1)
    channels = list(channels)
    for c in channels:
        l1.W[c] *= scale
        l1.b[c] *= scale
        l2.W[:, c] /= scale
    return [{"layer_index": layer_idx, "kind": "linear",
             "channels": [int(c) for c in channels], "scale": float(scale)}]


def inject_conv_outliers(model, layer_idx, channels, scale=8.0, inp_size=8):
    """Exact outlier pair on Conv2d ``layer_idx``.

    The compensation depends on what consumes the conv's output:
      * another Conv2d (ReLU between them is exact for positive s):
        divide that conv's weights for *input* channel c by s;
      * a Linear after Flatten: divide the contiguous H*W input-column
        block belonging to channel c by s.
    Both follow relu(s*x) = s*relu(x) for s > 0, so the FP function is
    unchanged (verified numerically by the caller).
    """
    conv = model.layers[layer_idx]
    assert conv.name == "conv" and scale > 0
    channels = list(channels)
    consumer = None
    for l in model.layers[layer_idx + 1:]:
        if l.name in ("conv", "linear"):
            consumer = l
            break
    if consumer is None:
        raise ValueError("conv has no conv/linear consumer")

    if consumer.name == "conv":
        for c in channels:
            conv.W[c] *= scale
            conv.b[c] *= scale
            consumer.W[:, c] /= scale          # input-channel weights
    else:
        flat_idx = next(i for i, l in enumerate(model.layers)
                        if l.name == "flatten")
        hout = (inp_size + 2 * conv.pad - conv.kh) // conv.stride + 1
        wout = (inp_size + 2 * conv.pad - conv.kw) // conv.stride + 1
        block = hout * wout
        for c in channels:
            conv.W[c] *= scale
            conv.b[c] *= scale
            consumer.W[:, c * block:(c + 1) * block] /= scale
    return [{"layer_index": layer_idx, "kind": "conv",
             "channels": [int(c) for c in channels], "scale": float(scale)}]


def verify_fp_equivalence(logits_before, logits_after, tol=2e-4):
    """Assert two logit sets agree to float32 rounding scale."""
    d = np.abs(logits_before - logits_after)
    scale = np.abs(logits_before).max() + 1e-8
    rel = d.max() / scale
    if rel > tol:
        raise AssertionError(
            f"FP equivalence violated: max abs diff {d.max():.3e}, "
            f"relative {rel:.3e} > {tol}")
    return {"max_abs_diff": float(d.max()), "relative_diff": float(rel)}
