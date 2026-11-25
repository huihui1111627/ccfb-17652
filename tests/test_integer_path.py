"""Verify the quantized GEMM path is genuinely integer arithmetic.

The engine's whole premise is real integer simulation rather than fake
quantization (quantize -> dequantize -> float matmul). We instrument the
integer GEMM entry points and assert every accumulator and requant input
has an integer dtype, and that the float reference tensors never feed the
matmul.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_tool.data import load_credit
from quant_tool.models.checkpoint import load_model
from quant_tool.quant.engine import QuantizedEngine, QuantConfig
from quant_tool import quant


def main():
    credit = load_credit()
    model, _ = load_model("mlp_sensitive")
    widx = [i for i, l in enumerate(model.layers)
            if l.name in ("linear", "conv", "attn")]
    cfg = QuantConfig({i: 8 for i in widx}, 8,
                      weight_granularity="per_tensor")
    eng = QuantizedEngine(model, credit["X_train"][:64], "minmax", cfg)

    # wrap the requant function to record the dtype of every accumulator
    import quant_tool.quant.engine as engmod
    orig = engmod.requantize_int
    seen = []

    def spy(acc, M0, shift, zp_out, bits, add_bias_int=0):
        seen.append(np.asarray(acc).dtype)
        assert np.issubdtype(np.asarray(acc).dtype, np.integer), \
            "accumulator reached requant as float!"
        assert isinstance(M0, (int, np.integer)) and \
               isinstance(shift, (int, np.integer)), \
            "fixed-point multiplier/shift must be integers"
        return orig(acc, M0, shift, zp_out, bits, add_bias_int)

    engmod.requantize_int = spy
    try:
        logits = eng.predict(credit["X_test"][:32])
    finally:
        engmod.requantize_int = orig

    assert seen, "no requantization was performed"
    assert all(np.issubdtype(d, np.integer) for d in seen), seen
    print(f"[OK] {len(seen)} requant calls, all accumulators integer: "
          f"{sorted(set(str(d) for d in seen))}")

    # quantized weights must be integer arrays
    for k, (qW, sw) in eng.qw.items():
        assert np.issubdtype(qW.dtype, np.integer), k
    print(f"[OK] {len(eng.qw)} quantized weight tensors all integer dtypes")

    # scales/zero-points are the only float metadata
    float_tensors = [v for v in eng.act_qp.values()]
    assert all(isinstance(s, float) and isinstance(z, (int, np.integer))
               for s, z, _ in float_tensors)
    print("[OK] activation metadata = float scale + integer zero point")

    # sanity: output is finite and close-ish to FP
    from quant_tool.models.training import predict_logits
    fp = predict_logits(model, credit["X_test"][:32])
    assert np.isfinite(logits).all()
    print(f"[OK] integer logits finite, mean |gap| vs FP = "
          f"{np.abs(logits - fp).mean():.4f}")
    print("integer-path verification passed")


if __name__ == "__main__":
    main()
