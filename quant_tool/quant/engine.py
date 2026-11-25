"""Quantized inference engine: calibrated, mixed-precision, integer-simulated.

The engine never mutates the trained FP model. It walks the same layer list
and replaces each weight-bearing layer's math with either

  * the integer path (bits 8 or 4):
        quantized weights (per-channel symmetric), affine-quantized
        activations, int64 GEMM/conv accumulation, integer bias add and a
        fixed-point multiply-and-shift requantization; or
  * the float path (bits 32): dequantized float GEMM, used for layers the
        mixed-precision search decides to keep in full precision.

Softmax and LayerNorm have no useful integer GEMM form at this scale, so
they run in a small float microkernel between explicit quantize/dequantize
boundaries — the same hybrid boundary real engines use. Every matmul,
including the attention Q@K^T and A@V, is integer on the quantized grids.

Calibration observes every tensor boundary (and attention internals) on a
user-chosen sample set, so "how the calibration set was chosen" is an
explicit, swappable input rather than an implicit default.
"""
from __future__ import annotations

import numpy as np

from .kernel import (activation_params, dequantize_activation,
                     quantize_activation, quantize_weights,
                     dequantize_weights, requant_params, requantize_int)
from .observer import make_observer
from ..models.layers import im2col

FP32 = 32


class QuantConfig:
    """Per-layer precision. ``weight_bits`` maps top-level layer index ->
    32/8/4. Activation bits default to 8 and can be overridden per layer."""

    def __init__(self, weight_bits=None, act_bits=8, act_bits_override=None,
                 weight_granularity="per_tensor"):
        self.weight_bits = dict(weight_bits or {})
        self.act_bits_default = act_bits
        self.act_bits_override = dict(act_bits_override or {})
        # "per_tensor" (outlier-sensitive) or "per_channel" (robust)
        self.weight_granularity = weight_granularity

    def wbits(self, idx):
        return int(self.weight_bits.get(idx, 8))

    def abits(self, idx):
        return int(self.act_bits_override.get(idx, self.act_bits_default))

    def label(self):
        vals = sorted(set(self.weight_bits.values()))
        if vals == [8]:
            return "all-int8"
        if vals == [4]:
            return "all-int4"
        return "mixed:" + ",".join(f"L{i}={b}" for i, b in
                                   sorted(self.weight_bits.items()))


class QuantizedEngine:
    def __init__(self, model, calib_X, observer_method="minmax",
                 config=None, batch=128):
        self.model = model
        self.batch = batch
        self.observer_method = observer_method
        self.config = config or QuantConfig()
        self.layers = model.layers
        # boundary observers: keyed ("out", idx) and attn internals
        self.obs = {}
        self._calibrate(calib_X)
        self._prepare()

    # ------------------------------------------------------------------ #
    # Calibration: float forward that records every tensor boundary
    # ------------------------------------------------------------------ #
    def _new_obs(self):
        return make_observer(self.observer_method)

    def _record(self, key, x):
        if key not in self.obs:
            self.obs[key] = self._new_obs()
        self.obs[key].observe(x)

    def _calibrate(self, X):
        for s in range(0, len(X), self.batch):
            xb = X[s:s + self.batch]
            self._record(("in", 0), xb)
            for i, layer in enumerate(self.layers):
                if layer.name == "attn":
                    xb = self._calib_attn(layer, xb, i)
                else:
                    xb = layer.forward(xb, train=False)
                self._record(("out", i), xb)

    def _calib_attn(self, attn, x, idx):
        n, t, d = x.shape
        Q = attn.Wq.forward(x)
        K = attn.Wk.forward(x)
        V = attn.Wv.forward(x)
        scale = 1.0 / np.sqrt(d)
        scores = Q @ K.transpose(0, 2, 1) * scale
        e = np.exp(scores - scores.max(-1, keepdims=True))
        probs = e / e.sum(-1, keepdims=True)
        ctx = probs @ V
        h = x + ctx
        y = attn.ln.forward(h)
        for sub, ten in (("Q", Q), ("K", K), ("V", V),
                         ("scores", scores), ("probs", probs),
                         ("ctx", ctx), ("h", h)):
            self._record(("attn", idx, sub), ten)
        return y

    def _qparams(self, key, bits):
        vmin, vmax = self.obs[key].range_()
        return activation_params(vmin, vmax, bits)

    # ------------------------------------------------------------------ #
    # Weight quantization / integer GEMM machinery
    # ------------------------------------------------------------------ #
    def _prepare(self):
        self.qw = {}          # idx -> (qW int32, scales float64) or None
        self.act_qp = {}      # boundary key -> (scale, zp, bits)
        for i, layer in enumerate(self.layers):
            if layer.name in ("linear", "conv"):
                b = self.config.wbits(i)
                if b != FP32:
                    self.qw[i] = quantize_weights(
                        layer.W, b,
                        granularity=self.config.weight_granularity)
            if layer.name == "attn":
                for sub in ("Wq", "Wk", "Wv"):
                    lin = getattr(layer, sub)
                    b = self.config.wbits(i)  # block shares one precision
                    self.qw[(i, sub)] = (
                        quantize_weights(
                            lin.W, b,
                            granularity=self.config.weight_granularity)
                        if b != FP32 else None)

    def _boundary_qp(self, key, bits):
        if key not in self.act_qp:
            self.act_qp[key] = (*self._qparams(key, bits), bits)
        return self.act_qp[key]

    def _align_grid(self, q, from_key, to_key, bits):
        """Integer requantization between two activation grids.

        Values in ``q`` live on ``from_key``'s grid; map them onto
        ``to_key``'s grid using a fixed-point multiply-and-shift, keeping
        everything in integer arithmetic. If both grids share a scale the
        call is an identity after the zero-point shift.
        """
        s_f, zp_f, _ = self._boundary_qp(from_key, bits)
        s_t, zp_t, _ = self._boundary_qp(to_key, bits)
        if abs(s_f - s_t) < 1e-12:
            out = q.astype(np.int64) - np.int64(zp_f) + np.int64(zp_t)
        else:
            M0, sh = requant_params(s_f, 1.0, s_t)
            centered = q.astype(np.int64) - np.int64(zp_f)
            out = requantize_int(centered, M0, sh, zp_t, bits)
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        return np.clip(out, qmin, qmax).astype(np.int32)

    def _to_int(self, x_float, key, bits):
        scale, zp, _ = self._boundary_qp(key, bits)
        return quantize_activation(x_float, scale, zp, bits), scale, zp

    def _int_linear(self, x_float, lin, idx, in_key, out_key, sub=None):
        """Integer (or float) affine GEMM for a Linear layer."""
        wkey = idx if sub is None else (idx, sub)
        b = self.config.wbits(idx)
        ab = self.config.abits(idx)
        if b == FP32:
            y = lin.forward(x_float, train=False)
            so, zp_o = self._boundary_qp(out_key, ab)[:2]
            return quantize_activation(y, so, zp_o, ab)
        qW, sw = self.qw[wkey]
        sx, zp_x, _ = self._boundary_qp(in_key, ab)
        so, zp_o, _ = self._boundary_qp(out_key, ab)
        qx, _, _ = self._to_int(x_float, in_key, ab)
        xc = qx.astype(np.int64) - np.int64(zp_x)
        acc = xc @ qW.T.astype(np.int64)                  # (..., O)
        qb = np.round(lin.b / (sw * sx)).astype(np.int64)
        acc = acc + qb
        # per-channel requant multipliers
        M0 = np.empty(len(sw), dtype=np.int64)
        sh = np.empty(len(sw), dtype=np.int64)
        for c in range(len(sw)):
            M0[c], sh[c] = requant_params(sw[c], sx, so)
        out = self._requant_channels(acc, M0, sh, zp_o, ab)
        return out

    @staticmethod
    def _requant_channels(acc, M0, shift, zp_o, bits):
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        a = acc.astype(np.int64)
        out = np.empty(a.shape, dtype=np.int64)
        for c in range(a.shape[-1]):
            prod = a[..., c] * np.int64(M0[c])
            sh = int(shift[c])
            v = prod if sh == 0 else (
                prod + np.int64(1 << (sh - 1))) >> sh
            out[..., c] = v
        out = out + np.int64(zp_o)
        return np.clip(out, qmin, qmax).astype(np.int32)

    def _int_conv(self, x_float, conv, idx, in_key, out_key):
        b = self.config.wbits(idx)
        ab = self.config.abits(idx)
        n = x_float.shape[0]
        if b == FP32:
            y = conv.forward(x_float, train=False)
            so, zp_o = self._boundary_qp(out_key, ab)[:2]
            return quantize_activation(y, so, zp_o, ab)
        qW, sw = self.qw[idx]
        sx, zp_x, _ = self._boundary_qp(in_key, ab)
        so, zp_o, _ = self._boundary_qp(out_key, ab)
        qx, _, _ = self._to_int(x_float, in_key, ab)
        col = im2col(qx, conv.kh, conv.kw, conv.stride, conv.pad)
        xc = col.astype(np.int64) - np.int64(zp_x)          # (N,K,L)
        W2 = qW.reshape(conv.out_c, -1).astype(np.int64)
        acc = np.einsum("ok,nkl->nol", W2, xc)              # (N,O,L)
        qb = np.round(conv.b / (sw * sx)).astype(np.int64)
        acc = acc + qb[None, :, None]
        M0 = np.empty(conv.out_c, dtype=np.int64)
        sh = np.empty(conv.out_c, dtype=np.int64)
        for c in range(conv.out_c):
            M0[c], sh[c] = requant_params(sw[c], sx, so)
        out = self._requant_channels(
            acc.transpose(0, 2, 1), M0, sh, zp_o, ab)      # (N,L,O)
        out = out.transpose(0, 2, 1)
        hout = (x_float.shape[2] + 2 * conv.pad - conv.kh) // conv.stride + 1
        wout = (x_float.shape[3] + 2 * conv.pad - conv.kw) // conv.stride + 1
        return out.reshape(n, conv.out_c, hout, wout)

    def _int_attn(self, attn, x_float, idx):
        """Hybrid attention: integer Q/K/V GEMMs and attention matmuls,
        float softmax and LayerNorm between quantized boundaries."""
        ab = self.config.abits(idx)
        in_key = ("out", idx - 1) if idx > 0 else ("in", 0)
        qx, sx, zpx = self._to_int(x_float, in_key, ab)
        x_int = qx
        # Q, K, V via integer linears
        qQ = self._attn_linear(attn.Wq, x_float, idx, "Wq", in_key,
                               ("attn", idx, "Q"))
        qK = self._attn_linear(attn.Wk, x_float, idx, "Wk", in_key,
                               ("attn", idx, "K"))
        qV = self._attn_linear(attn.Wv, x_float, idx, "Wv", in_key,
                               ("attn", idx, "V"))
        sQ, zpQ, _ = self._boundary_qp(("attn", idx, "Q"), ab)
        sK, zpK, _ = self._boundary_qp(("attn", idx, "K"), ab)
        sV, zpV, _ = self._boundary_qp(("attn", idx, "V"), ab)
        sS, zpS, _ = self._boundary_qp(("attn", idx, "scores"), ab)
        sP, zpP, _ = self._boundary_qp(("attn", idx, "probs"), ab)
        sC, zpC, _ = self._boundary_qp(("attn", idx, "ctx"), ab)
        # integer Q@K^T
        Qc = qQ.astype(np.int64) - np.int64(zpQ)
        Kc = qK.astype(np.int64) - np.int64(zpK)
        raw = Qc @ Kc.transpose(0, 2, 1)                    # (N,T,T)
        d = attn.dim
        # raw accumulator is in units of sQ*sK (per element, /sqrt(d) below)
        M0, sh = requant_params(sQ * sK / np.sqrt(d), 1.0, sS)
        qS = requantize_int(raw, M0, sh, zpS, ab)
        scores = dequantize_activation(qS, sS, zpS)
        e = np.exp(scores - scores.max(-1, keepdims=True))
        probs = (e / e.sum(-1, keepdims=True)).astype(np.float32)
        qP = quantize_activation(probs, sP, zpP, ab)
        Pc = qP.astype(np.int64) - np.int64(zpP)
        Vc = qV.astype(np.int64) - np.int64(zpV)
        raw_ctx = Pc @ Vc                                    # (N,T,d)
        M0c, shc = requant_params(sP * sV, 1.0, sC)
        qC = requantize_int(raw_ctx, M0c, shc, zpC, ab)
        # residual: align ctx grid to input grid, integer add
        M0a, sha = requant_params(sC, 1.0, sx)
        qC_aligned = requantize_int(
            qC.astype(np.int64) - np.int64(zpC), M0a, sha, zpx, ab)
        qh = np.clip(x_int.astype(np.int64) +
                     qC_aligned.astype(np.int64),
                     -(1 << (ab - 1)), (1 << (ab - 1)) - 1).astype(np.int32)
        h = dequantize_activation(qh, sx, zpx)
        y = attn.ln.forward(h, train=False)
        so, zp_o, _ = self._boundary_qp(("out", idx), ab)
        return quantize_activation(y, so, zp_o, ab)

    def _attn_linear(self, lin, x_float, idx, sub, in_key, out_key):
        b = self.config.wbits(idx)
        ab = self.config.abits(idx)
        if b == FP32:
            y = lin.forward(x_float, train=False)
            so, zp_o = self._boundary_qp(out_key, ab)[:2]
            return quantize_activation(y, so, zp_o, ab)
        qW, sw = self.qw[(idx, sub)]
        sx, zp_x, _ = self._boundary_qp(in_key, ab)
        so, zp_o, _ = self._boundary_qp(out_key, ab)
        qx, _, _ = self._to_int(x_float, in_key, ab)
        lead = qx.shape[:-1]
        xc = qx.reshape(-1, qx.shape[-1]).astype(np.int64) - np.int64(zp_x)
        acc = xc @ qW.T.astype(np.int64)
        qb = np.round(lin.b / (sw * sx)).astype(np.int64)
        acc = acc + qb
        M0 = np.empty(len(sw), dtype=np.int64)
        sh = np.empty(len(sw), dtype=np.int64)
        for c in range(len(sw)):
            M0[c], sh[c] = requant_params(sw[c], sx, so)
        out = self._requant_channels(acc, M0, sh, zp_o, ab)
        return out.reshape(*lead, -1)

    # ------------------------------------------------------------------ #
    # Public forward
    # ------------------------------------------------------------------ #
    def forward(self, X, trace=False):
        """If trace=True, also returns per-layer (fp output, int output)."""
        fp_trace, q_trace = [], []
        x_fp = X
        x_q = None          # current tensor on an integer grid
        for i, layer in enumerate(self.layers):
            in_key = ("out", i - 1) if i > 0 else ("in", 0)
            out_key = ("out", i)
            ab = self.config.abits(i)
            name = layer.name
            if name == "linear":
                x_q = self._int_linear(x_fp, layer, i, in_key, out_key)
                x_fp_new = layer.forward(x_fp, train=False)
                x_fp = x_fp_new
            elif name == "conv":
                x_q = self._int_conv(x_fp, layer, i, in_key, out_key)
                x_fp = layer.forward(x_fp, train=False)
            elif name == "attn":
                x_q = self._int_attn(layer, x_fp, i)
                x_fp = self._calib_attn(layer, x_fp, i)
            elif name == "relu":
                # integer relu on the *input* grid: clamp at its zero point
                _, zp_in, _ = self._boundary_qp(in_key, ab)
                x_q = np.maximum(x_q, np.int32(zp_in))
                x_q = self._align_grid(x_q, in_key, out_key, ab)
                x_fp = layer.forward(x_fp, train=False)
            elif name == "flatten":
                x_q = x_q.reshape(x_q.shape[0], -1)
                x_q = self._align_grid(x_q, in_key, out_key, ab)
                x_fp = layer.forward(x_fp, train=False)
            elif name == "meanpool":
                # integer-only mean: sum then rounded integer division by T
                t = x_q.shape[1]
                s = x_q.astype(np.int64).sum(axis=1)
                x_q = ((s + np.where(s >= 0, t // 2, -((t - 1) // 2)))
                       // t).astype(np.int32)
                x_q = self._align_grid(x_q, in_key, out_key, ab)
                x_fp = layer.forward(x_fp, train=False)
            else:
                raise ValueError(f"unsupported layer {name}")
            if trace:
                fp_trace.append(x_fp.copy())
                q_trace.append(
                    dequantize_activation(x_q, *self._boundary_qp(
                        out_key, ab)[:2]).copy())
        # final logits back to float
        s, zp, _ = self._boundary_qp(("out", len(self.layers) - 1),
                                     self.config.abits(len(self.layers) - 1))
        logits_q = dequantize_activation(x_q, s, zp)
        if trace:
            return logits_q, fp_trace, q_trace
        return logits_q

    def predict(self, X):
        outs = []
        for s in range(0, len(X), self.batch):
            outs.append(self.forward(X[s:s + self.batch]))
        return np.concatenate(outs, axis=0)
