"""Pure-NumPy layers: forward + backward, no autodiff.

Every trainable layer exposes ``params`` / ``grads`` lists of matching
array references. Activations needed for backprop are cached on the layer
during ``forward`` and cleared by ``new_epoch``.
"""
from __future__ import annotations

import numpy as np


def _xavier(rng, fan_in, fan_out):
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_out, fan_in)).astype(np.float32)


class Layer:
    name = "layer"

    def forward(self, x, train=True):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def params(self):
        return []

    def grads(self):
        return []


class Linear(Layer):
    name = "linear"

    def __init__(self, fan_in, fan_out, rng):
        self.W = _xavier(rng, fan_in, fan_out)
        self.b = np.zeros(fan_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x, train=True):
        self._x = x
        return x @ self.W.T + self.b

    def backward(self, grad):
        # grad (..., out); flatten leading dims (e.g. N*T for attention)
        x = self._x
        lead = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1])
        gf = grad.reshape(-1, grad.shape[-1])
        # in-place: Adam holds references to self.dW / self.db
        np.matmul(gf.T, xf, out=self.dW)
        np.sum(gf, axis=0, out=self.db)
        return (gf @ self.W).reshape(*lead, self.W.shape[1])

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU(Layer):
    name = "relu"

    def forward(self, x, train=True):
        self._mask = x > 0
        return np.maximum(x, 0.0)

    def backward(self, grad):
        return grad * self._mask


class Flatten(Layer):
    name = "flatten"

    def forward(self, x, train=True):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._shape)


def im2col(x, kh, kw, stride, pad):
    """x (N,C,H,W) padded -> cols (N, C*kh*kw, L)."""
    n, c, h, w = x.shape
    if pad:
        xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=x.dtype)
        xp[:, :, pad:pad + h, pad:pad + w] = x
    else:
        xp = x
    hout = (xp.shape[2] - kh) // stride + 1
    wout = (xp.shape[3] - kw) // stride + 1
    cols = np.zeros((n, c, kh, kw, hout, wout), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * hout
        for j in range(kw):
            j_max = j + stride * wout
            cols[:, :, i, j, :, :] = xp[:, :, i:i_max:stride, j:j_max:stride]
    return cols.reshape(n, c * kh * kw, hout * wout)


def col2im(cols, shape, kh, kw, stride, pad):
    n, c, h, w = shape
    hout = (h + 2 * pad - kh) // stride + 1
    wout = (w + 2 * pad - kw) // stride + 1
    cols = cols.reshape(n, c, kh, kw, hout, wout)
    hp, wp = h + 2 * pad, w + 2 * pad
    xp = np.zeros((n, c, hp, wp), dtype=cols.dtype)
    for i in range(kh):
        i_max = i + stride * hout
        for j in range(kw):
            j_max = j + stride * wout
            xp[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]
    if pad:
        return xp[:, :, pad:pad + h, pad:pad + w]
    return xp


class Conv2d(Layer):
    name = "conv"

    def __init__(self, in_c, out_c, kh, kw, rng, stride=1, pad=0):
        self.in_c, self.out_c, self.kh, self.kw = in_c, out_c, kh, kw
        self.stride, self.pad = stride, pad
        self.W = _xavier(rng, in_c * kh * kw, out_c).reshape(
            out_c, in_c, kh, kw).copy()
        self.b = np.zeros(out_c, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, train=True):
        n = x.shape[0]
        self._x_shape = x.shape
        col = im2col(x, self.kh, self.kw, self.stride, self.pad)  # N,K,L
        self._col = col
        out = (self.W.reshape(self.out_c, -1) @ col
               + self.b[None, :, None])  # N,O,L
        hout = (x.shape[2] + 2 * self.pad - self.kh) // self.stride + 1
        wout = (x.shape[3] + 2 * self.pad - self.kw) // self.stride + 1
        return out.reshape(n, self.out_c, hout, wout)

    def backward(self, grad):
        n = grad.shape[0]
        go = grad.reshape(n, self.out_c, -1)               # N,O,L
        np.einsum("nol,nkl->ok", go, self._col,
                  out=self.dW.reshape(self.out_c, -1))
        np.sum(go, axis=(0, 2), out=self.db)
        W2 = self.W.reshape(self.out_c, -1)
        dcol = W2.T @ go                                   # N,K,L
        return col2im(dcol, self._x_shape, self.kh, self.kw,
                      self.stride, self.pad)

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class LayerNorm(Layer):
    """LayerNorm over the last axis (used inside attention blocks)."""
    name = "layernorm"

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.eps = eps

    def forward(self, x, train=True):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._inv = 1.0 / np.sqrt(var + self.eps)
        self._hat = (x - mu) * self._inv
        return self.gamma * self._hat + self.beta

    def backward(self, grad):
        d = grad.shape[-1]
        axes = tuple(range(grad.ndim - 1))
        np.sum(grad * self._hat, axis=axes, out=self.dgamma)
        np.sum(grad, axis=axes, out=self.dbeta)
        # standard LN gradient
        g = grad * self.gamma
        mean_gh = (g * self._hat).mean(axis=-1, keepdims=True)
        return self._inv / d * (d * g - g.sum(axis=-1, keepdims=True)
                                - self._hat * d * mean_gh)

    def params(self):
        return [self.gamma, self.beta]

    def grads(self):
        return [self.dgamma, self.dbeta]


class SelfAttentionBlock(Layer):
    """Single-head self-attention + residual + LayerNorm.

    Input/output: (N, T, D). Q/K/V are three separate Linear layers.
    """
    name = "attn"

    def __init__(self, dim, rng):
        self.dim = dim
        self.Wq = Linear(dim, dim, rng)
        self.Wk = Linear(dim, dim, rng)
        self.Wv = Linear(dim, dim, rng)
        self.ln = LayerNorm(dim)
        self._cache = {}

    def forward(self, x, train=True):
        n, t, d = x.shape
        Q = self.Wq.forward(x)
        K = self.Wk.forward(x)
        V = self.Wv.forward(x)
        scale = 1.0 / np.sqrt(d)
        scores = Q @ K.transpose(0, 2, 1) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        a = e / e.sum(axis=-1, keepdims=True)
        ctx = a @ V
        h = x + ctx
        y = self.ln.forward(h)
        if train:
            self._cache = dict(x=x, Q=Q, K=K, V=V, a=a, h=h, scale=scale)
        return y

    def backward(self, dy):
        c = self._cache
        x, Q, K, V, a, scale = c["x"], c["Q"], c["K"], c["V"], c["a"], c["scale"]
        dh = self.ln.backward(dy)
        dV = a.transpose(0, 2, 1) @ dh
        da = dh @ V.transpose(0, 2, 1)
        dscores = a * (da - (da * a).sum(axis=-1, keepdims=True))
        dQ = dscores @ K * scale
        dK = dscores.transpose(0, 2, 1) @ Q * scale
        dq_x = self.Wq.backward(dQ)
        dk_x = self.Wk.backward(dK)
        dv_x = self.Wv.backward(dV)
        return dh + dq_x + dk_x + dv_x

    def params(self):
        return self.Wq.params() + self.Wk.params() + self.Wv.params() + \
            self.ln.params()

    def grads(self):
        return self.Wq.grads() + self.Wk.grads() + self.Wv.grads() + \
            self.ln.grads()


class MeanPool(Layer):
    """Mean over the time axis: (N,T,D) -> (N,D)."""
    name = "meanpool"

    def forward(self, x, train=True):
        self._t = x.shape[1]
        return x.mean(axis=1)

    def backward(self, grad):
        return np.repeat(grad[:, None, :] / self._t, self._t, axis=1)


def softmax_cross_entropy(logits, y):
    """Fused softmax CE. Returns (loss, dlogits)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    grad = p.copy()
    grad[np.arange(n), y] -= 1.0
    return loss.astype(np.float32), (grad / n).astype(np.float32)


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def all_params_grads(self):
        ps, gs = [], []
        for layer in self.layers:
            ps += layer.params()
            gs += layer.grads()
        return ps, gs

    def named_layers(self):
        counts = {}
        out = []
        for layer in self.layers:
            k = layer.name
            counts[k] = counts.get(k, 0) + 1
            out.append((f"{k}{counts[k]}", layer))
        return out
