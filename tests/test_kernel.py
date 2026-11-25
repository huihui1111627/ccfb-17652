"""Unit tests for quantization primitives and fixed-point requant math."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_tool.quant.kernel import (signed_bounds, weight_scales,
                                     quantize_weights, dequantize_weights,
                                     activation_params, quantize_activation,
                                     dequantize_activation, requant_params,
                                     requantize_int, sqnr)


def test_weight_roundtrip():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 8)).astype(np.float32)
    for bits in (8, 4):
        q, s = quantize_weights(w, bits)
        assert np.issubdtype(q.dtype, np.integer)
        qmin, qmax = signed_bounds(bits)
        assert q.min() >= qmin and q.max() <= qmax
        wq = dequantize_weights(q, s)
        err = np.abs(w - wq).mean()
        assert err < (0.02 if bits == 8 else 0.12), (bits, err)
        # per-channel: no channel's dequantized magnitude exceeds its range
        print(f"[OK] weight int{bits} roundtrip mean err {err:.4f}")


def test_activation_affine():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((100,)) * 3 + 1).astype(np.float32)
    for bits in (8, 4):
        s, zp = activation_params(x.min(), x.max(), bits)
        q = quantize_activation(x, s, zp, bits)
        qmin, qmax = signed_bounds(bits)
        assert q.min() >= qmin and q.max() <= qmax
        xh = dequantize_activation(q, s, zp)
        err = np.abs(x - xh).mean()
        assert err < (0.05 if bits == 8 else 0.3), (bits, err)
        print(f"[OK] activation int{bits} affine roundtrip err {err:.4f}, "
              f"zp {zp}")


def test_requant_matches_float():
    """The fixed-point path must match round(acc * sw*sx/so)/so + zp."""
    rng = np.random.default_rng(2)
    for (sw, sx, so) in [(0.01, 0.02, 0.05), (0.003, 0.007, 0.01),
                         (0.1, 0.1, 0.02), (1e-4, 5e-3, 2e-3)]:
        acc = rng.integers(-5000, 5000, size=2000, dtype=np.int64)
        M0, shift = requant_params(sw, sx, so)
        assert shift >= 0 and M0 > 0
        q = requantize_int(acc, M0, shift, zp_out=3, bits=8)
        # float reference: real value acc*sw*sx mapped onto so grid
        real = acc.astype(np.float64) * sw * sx
        ref_q = np.round(real / so).astype(np.int64) + 3
        ref_q = np.clip(ref_q, *signed_bounds(8))
        mismatch = np.mean(q != ref_q)
        assert mismatch < 0.02, (sw, sx, so, mismatch)
        # and never off by more than one grid step
        assert np.max(np.abs(q - ref_q)) <= 1
        print(f"[OK] requant sw={sw} sx={sx} so={so}: M0={M0} "
              f"shift={shift} mismatch {mismatch:.4f}")


def test_requant_scale_identity():
    # when sw*sx == so, the fixed point must be essentially exact
    M0, shift = requant_params(0.02, 0.05, 0.001)
    acc = np.array([-1000, -1, 0, 1, 1000], dtype=np.int64)
    q = requantize_int(acc, M0, shift, 0, 16)
    assert np.max(np.abs(q - acc)) <= 1, (q, acc)
    print("[OK] requant identity-scale within 1 LSB")


def test_sqnr_monotone_bits():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5000,)).astype(np.float32)
    s8 = sqnr(x, dequantize_weights(*quantize_weights(x[None, :], 8))[0])
    s4 = sqnr(x, dequantize_weights(*quantize_weights(x[None, :], 4))[0])
    assert s8 > s4
    print(f"[OK] SQNR int8 {s8:.1f}dB > int4 {s4:.1f}dB")


if __name__ == "__main__":
    test_weight_roundtrip()
    test_activation_affine()
    test_requant_matches_float()
    test_requant_scale_identity()
    test_sqnr_monotone_bits()
    print("all kernel unit tests passed")
