"""Layer- and channel-level quantization sensitivity analysis.

Four complementary views, each answering one question from the brief:

* layer SQNR          — how much noise each layer boundary carries.
* single-layer ablation — exactly which layer, quantized alone, costs the
                          most accuracy / key-class recall.
* channel outliers    — the specific output channels whose unusually wide
                          range coarsens the shared (per-tensor) grid.
* error propagation   — for one sample, where the per-layer output gap
                          first appears and where it is amplified.
"""
from __future__ import annotations

import numpy as np

from ..quant.engine import QuantizedEngine, QuantConfig, FP32
from ..quant.kernel import sqnr
from ..metrics import evaluate
from ..models.training import predict_logits

WEIGHT_LAYERS = ("linear", "conv", "attn")


def weight_layer_indices(model):
    return [i for i, l in enumerate(model.layers) if l.name in WEIGHT_LAYERS]


# --------------------------------------------------------------------------
# Float activation capture (for per-channel ranges and SQNR reference)
# --------------------------------------------------------------------------

def capture_activations(model, X, batch=128):
    """Float forward capturing each layer's output, stacked over samples."""
    captured = [[] for _ in model.layers]
    for s in range(0, len(X), batch):
        x = X[s:s + batch]
        for i, layer in enumerate(model.layers):
            x = layer.forward(x, train=False)
            captured[i].append(x.copy())
    return [np.concatenate(c, axis=0) for c in captured]


# --------------------------------------------------------------------------
# Layer SQNR from an engine trace
# --------------------------------------------------------------------------

def layer_sqnr(engine, X):
    """Per-layer SQNR (dB) between float and dequantized integer outputs."""
    _, fp_trace, q_trace = engine.forward(X[:min(len(X), 256)], trace=True)
    rows = []
    for i, (fp, q) in enumerate(zip(fp_trace, q_trace)):
        rows.append({"layer": i, "name": engine.layers[i].name,
                     "sqnr_db": sqnr(fp, q)})
    return rows


# --------------------------------------------------------------------------
# Single-layer ablation
# --------------------------------------------------------------------------

def single_layer_ablation(model, calib, X, y, key_class, probe_bits=4,
                          granularity="per_tensor", observer="minmax"):
    """Quantize one weight layer at probe_bits with everything else FP32.

    The resulting accuracy/key-recall gap vs the FP model isolates that
    layer's own sensitivity. Also reports the all-INT8 and all-probe
    baselines for context.
    """
    fp_metrics = evaluate(y, predict_logits(model, X), key_class)
    widx = weight_layer_indices(model)

    def run(cfg):
        eng = QuantizedEngine(model, calib, observer, cfg)
        return evaluate(y, eng.predict(X), key_class)

    base8 = QuantConfig({i: 8 for i in widx}, 8,
                        weight_granularity=granularity)
    all8 = run(base8)
    allprobe = run(QuantConfig({i: probe_bits for i in widx}, 8,
                               weight_granularity=granularity))
    rows = []
    for i in widx:
        cfg = QuantConfig({j: (probe_bits if j == i else FP32)
                           for j in widx}, 8,
                          weight_granularity=granularity)
        m = run(cfg)
        rows.append({
            "layer": i, "name": model.layers[i].name,
            "acc": m["accuracy"],
            "acc_drop": m["accuracy"] - fp_metrics["accuracy"],
            "key_recall": m["key_recall"],
            "key_recall_drop": m["key_recall"] - fp_metrics["key_recall"],
        })
    rows.sort(key=lambda r: r["acc_drop"])
    return {
        "fp": fp_metrics, "all_int8": all8,
        f"all_int{probe_bits}": allprobe,
        "ablation": rows,
    }


# --------------------------------------------------------------------------
# Per-channel outlier detection
# --------------------------------------------------------------------------

def channel_outliers(model, X, top_k=8):
    """Find output channels with anomalously wide weight/activation range.

    Two signals per output channel:
      w_ratio: channel weight abs-max divided by the layer median — a
               per-tensor grid is set by the largest, so a high ratio means
               this channel steals grid levels from all the others.
      a_ratio: same idea on observed float activation magnitudes.
    """
    acts = capture_activations(model, X[:256])
    rows = []
    for i, layer in enumerate(model.layers):
        if layer.name not in ("linear", "conv"):
            continue
        W = layer.W
        wmax = np.max(np.abs(W.reshape(W.shape[0], -1)), axis=1)  # (O,)
        wmed = np.median(wmax) + 1e-12
        # activation channels: last axis for linear (N,O), axis1 for conv
        a = acts[i]
        if layer.name == "linear":
            amax = np.max(np.abs(a), axis=tuple(range(a.ndim - 1)))
        else:
            amax = np.max(np.abs(a), axis=(0, 2, 3))
        amed = np.median(amax) + 1e-12
        order = np.argsort(-(wmax / wmed))[:top_k]
        for c in order:
            rows.append({
                "layer": i, "name": layer.name, "channel": int(c),
                "w_absmax": float(wmax[c]),
                "w_ratio_vs_median": float(wmax[c] / wmed),
                "act_absmax": float(amax[c]),
                "act_ratio_vs_median": float(amax[c] / amed),
                "outlier": bool(wmax[c] / wmed > 3.0 or
                                amax[c] / amed > 3.0),
            })
    return rows


# --------------------------------------------------------------------------
# Single-sample error propagation
# --------------------------------------------------------------------------

def trace_sample(engine, x):
    """Layer-by-layer float vs integer outputs for one sample.

    Returns per-layer: SQNR, mean/max absolute gap, and the cosine
    similarity between flattened float and integer tensors. The first
    layer where SQNR drops sharply (or cosine collapses) localizes where
    quantization error is introduced; a falling cosine downstream shows
    where it is amplified.
    """
    logits_q, fp_trace, q_trace = engine.forward(x[None], trace=True)
    rows = []
    for i, (fp, q) in enumerate(zip(fp_trace, q_trace)):
        f, qq = fp.ravel().astype(np.float64), q.ravel().astype(np.float64)
        cos = float(np.dot(f, qq) / (np.linalg.norm(f) *
                                     np.linalg.norm(qq) + 1e-12))
        rows.append({
            "layer": i, "name": engine.layers[i].name,
            "sqnr_db": sqnr(fp, q),
            "mean_abs_gap": float(np.mean(np.abs(f - qq))),
            "max_abs_gap": float(np.max(np.abs(f - qq))),
            "cosine": cos,
            "fp_flat": f.tolist(),
            "q_flat": qq.tolist(),
        })
    return rows, logits_q[0]
