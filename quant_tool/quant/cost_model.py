"""Model size and integer-compute cost model.

Both numbers are derived from the chosen per-layer precision, not measured
wall-clock (the simulation runs in NumPy), and are reported in the same
units a deployment review uses:

* size: bytes occupied by quantized weights plus per-channel scales and
  integer biases, layer by layer.
* integer ops: MAC count (multiply-accumulate) split by weight precision,
  plus an estimated cycle count using representative per-precision integer
  throughputs (configurable), so two mixed-precision plans under the same
  accuracy budget can be compared on speed, not just size.
"""
from __future__ import annotations

import numpy as np

from .kernel import signed_bounds

# Representative relative cycle cost per MAC. INT8 is the baseline; INT4
# MACs are cheaper on SIMD/accelerator hardware; FP32 is far more expensive.
DEFAULT_CYCLES_PER_MAC = {4: 0.5, 8: 1.0, 32: 4.0}
BYTES_PER_PARAM = {4: 0.5, 8: 1.0, 32: 4.0}  # 4-bit packed = 0.5 byte


def layer_weight_count(layer):
    if layer.name in ("linear", "conv"):
        return int(layer.W.size), int(layer.b.size)
    if layer.name == "attn":
        n = 0
        for sub in ("Wq", "Wk", "Wv"):
            lin = getattr(layer, sub)
            n += lin.W.size + lin.b.size
        n += layer.ln.gamma.size + layer.ln.beta.size
        return n, 0
    if layer.name == "layernorm":
        return int(layer.gamma.size + layer.beta.size), 0
    return 0, 0


def layer_macs(layer, input_shape):
    """MAC count for one sample given the layer's input shape."""
    name = layer.name
    if name == "linear":
        return int(np.prod(input_shape[:-1])) * layer.W.shape[1] * \
            layer.W.shape[0]
    if name == "conv":
        hout = (input_shape[2] + 2 * layer.pad - layer.kh) // layer.stride + 1
        wout = (input_shape[3] + 2 * layer.pad - layer.kw) // layer.stride + 1
        return hout * wout * layer.out_c * layer.in_c * layer.kh * layer.kw
    if name == "attn":
        n, t, d = input_shape
        # three projections + QK^T + A@V
        return 3 * t * d * d + 2 * t * t * d
    if name == "meanpool":
        return int(np.prod(input_shape))
    return 0


def size_and_cost(model, config, input_shape, cycles_per_mac=None):
    """Returns per-layer rows and totals for a QuantConfig."""
    cpm = cycles_per_mac or DEFAULT_CYCLES_PER_MAC
    rows = []
    shape = input_shape
    total_bytes = total_macs = total_cycles = 0.0
    for i, layer in enumerate(model.layers):
        n_w, n_b = layer_weight_count(layer)
        bits = config.wbits(i) if layer.name in ("linear", "conv", "attn") \
            else 32
        macs = layer_macs(layer, shape)
        # weight bytes at the chosen precision; scales/biases stay 32-bit
        w_bytes = n_w * BYTES_PER_PARAM.get(bits, 4.0)
        scale_bytes = n_b * 4.0 if n_w else 0.0
        bias_bytes = n_b * 4.0
        layer_bytes = w_bytes + scale_bytes + bias_bytes
        cyc = macs * cpm.get(bits, 4.0)
        rows.append({
            "layer": i, "name": layer.name, "bits": bits,
            "weights": n_w, "macs": macs,
            "bytes": float(layer_bytes),
            "cycles": float(cyc),
        })
        total_bytes += layer_bytes
        total_macs += macs
        total_cycles += cyc
        # update shape
        if layer.name == "linear":
            shape = (*shape[:-1], layer.W.shape[0])
        elif layer.name == "conv":
            hout = (shape[2] + 2 * layer.pad - layer.kh) // layer.stride + 1
            wout = (shape[3] + 2 * layer.pad - layer.kw) // layer.stride + 1
            shape = (shape[0], layer.out_c, hout, wout)
        elif layer.name == "flatten":
            shape = (shape[0], int(np.prod(shape[1:])))
        elif layer.name == "meanpool":
            shape = (shape[0], shape[2])
        elif layer.name == "attn":
            pass
    return {
        "layers": rows,
        "total_bytes": float(total_bytes),
        "total_macs": int(total_macs),
        "estimated_cycles": float(total_cycles),
    }
