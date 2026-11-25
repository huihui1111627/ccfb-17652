"""Quantization primitives and the integer-only compute path.

Two tensor schemes, deliberately kept close to what an embedded inference
engine actually does:

Weights (Linear / Conv), symmetric per output channel
    q = round(w / scale) clipped to [-2^(b-1), 2^(b-1)-1], zero point 0.
    One scale per output channel — the basis for per-channel outlier
    analysis and for mixed precision per layer.

Activations, asymmetric affine with integer zero point
    q = round(x / scale) + zp,  clipped to the b-bit signed grid.
    Range (vmin, vmax) comes from a calibration observer. This mirrors
    TFLite-style uint/int activation quantization.

GEMM and convolution run on the integer grids: products and accumulations
use int64 to emulate the hardware integer accumulator without overflow, and
the requantization back to the next layer's activation grid is an integer
multiply-and-shift (M0, shift), not a floating-point multiply.
A float reference path is kept solely to cross-check the integer path.
"""
from __future__ import annotations

import numpy as np


def signed_bounds(bits):
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


# --------------------------------------------------------------------------
# Weight quantization: symmetric, per output channel
# --------------------------------------------------------------------------

def weight_scales(w, bits, channel_axis=0, granularity="per_channel"):
    """Weight scales.

    granularity="per_channel": one symmetric scale per output channel
        (axis 0) — robust to outlier channels.
    granularity="per_tensor":  one shared scale for the whole tensor —
        a single wide channel then coarsens every other channel's grid,
        which is the outlier-drags-down-a-layer failure mode.
    """
    qmin, qmax = signed_bounds(bits)
    if granularity == "per_channel":
        axis = tuple(i for i in range(w.ndim) if i != channel_axis)
        wmax = np.max(np.abs(w), axis=axis)                  # (O,)
    elif granularity == "per_tensor":
        wmax = np.full(w.shape[channel_axis], np.max(np.abs(w)),
                       dtype=np.float64)
    else:
        raise ValueError(granularity)
    wmax = np.maximum(wmax, 1e-8)
    return (wmax / qmax).astype(np.float64)


def quantize_weights(w, bits, scales=None, granularity="per_channel"):
    """Returns (q int32, scales float64 per output channel)."""
    if scales is None:
        scales = weight_scales(w, bits, granularity=granularity)
    qmin, qmax = signed_bounds(bits)
    s = scales.reshape((-1,) + (1,) * (w.ndim - 1))
    q = np.round(w / s).clip(qmin, qmax).astype(np.int32)
    return q, scales


def dequantize_weights(q, scales):
    s = scales.reshape((-1,) + (1,) * (q.ndim - 1))
    return (q.astype(np.float32) * s.astype(np.float32))


# --------------------------------------------------------------------------
# Activation quantization: asymmetric affine
# --------------------------------------------------------------------------

def activation_params(vmin, vmax, bits):
    """Returns (scale, zero_point) mapping [vmin,vmax] onto the signed grid."""
    qmin, qmax = signed_bounds(bits)
    vmin = float(min(vmin, 0.0))
    vmax = float(max(vmax, 0.0))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    scale = (vmax - vmin) / (qmax - qmin)
    zp = int(round(qmin - vmin / scale))
    zp = max(qmin, min(qmax, zp))
    return float(scale), zp


def quantize_activation(x, scale, zp, bits):
    qmin, qmax = signed_bounds(bits)
    q = np.round(x / scale).astype(np.int64) + zp
    return np.clip(q, qmin, qmax).astype(np.int32)


def dequantize_activation(q, scale, zp):
    return (q.astype(np.float32) - zp) * np.float32(scale)


# --------------------------------------------------------------------------
# Integer requantization: emulated fixed-point multiply-and-shift
# --------------------------------------------------------------------------

def requant_params(sw, sx, so, mult_bits=31):
    """Integer multiplier M0 and right-shift for acc-scale -> output-scale.

    The integer accumulator is in units of sw*sx. We want units of so, so
    the real multiplier is M = sw*sx/so (always > 0). It is represented as

        M = M0 / 2^shift,   2^(mult_bits-1) <= M0 < 2^mult_bits,  shift >= 0

    matching the TFLite fixed-point convention. ``out = round(acc*M0) >>
    shift`` then equals round(acc*M) up to integer rounding error.
    """
    M = (sw * sx) / so
    shift = 0
    M0 = M
    target_lo = 1 << (mult_bits - 1)
    target_hi = 1 << mult_bits
    while M0 < target_lo:
        M0 *= 2.0
        shift += 1
    while M0 >= target_hi:
        M0 /= 2.0
        shift -= 1
    if shift < 0:                      # defensive: M <= 1 must give shift>=0
        M0 *= 2.0 ** (-shift)
        shift = 0
    return int(round(M0)), int(shift)


def requantize_int(acc, M0, shift, zp_out, bits, add_bias_int=0):
    """acc int64 -> output grid int32 using only integer ops.

    out = round(acc * M0 / 2^shift) + zp_out with round-half-away-from-zero
    implemented via an integer rounding constant; M0 > 0 by construction.
    """
    qmin, qmax = signed_bounds(bits)
    a = acc.astype(np.int64) + np.int64(add_bias_int)
    prod = a * np.int64(M0)
    if shift == 0:
        scaled = prod
    else:
        rounded = prod + np.int64(1 << (shift - 1))
        scaled = rounded >> shift
    out = scaled + np.int64(zp_out)
    return np.clip(out, qmin, qmax).astype(np.int32)


# --------------------------------------------------------------------------
# SQNR helper used all over the analysis layer
# --------------------------------------------------------------------------

def sqnr(ref, q):
    """Signal-to-quantization-noise ratio in dB, per-tensor."""
    ref = ref.astype(np.float64)
    noise = ref - q.astype(np.float64)
    p_sig = float(np.mean(ref ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise < 1e-20:
        return 99.0
    return float(10.0 * np.log10(p_sig / p_noise))
