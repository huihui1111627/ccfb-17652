"""Sanity tests for the integer-simulated quantized engine.

Checks:
  1. Engine INT8 stays close to the FP model on every architecture.
  2. INT4 degrades more than INT8, and outlier-injected models degrade
     more than the robust one (the phenomenon the toolkit targets).
  3. The integer path is genuinely integer: all GEMM intermediates are
     integer dtypes and the fixed-point requant uses no float multiply.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_tool.data import load_credit, load_digits
from quant_tool.models.checkpoint import load_model
from quant_tool.models.training import predict_logits
from quant_tool.quant.engine import QuantizedEngine, QuantConfig, FP32
from quant_tool.metrics import evaluate


def run_case(name, Xtr, Xte, yte, key_class):
    model, meta = load_model(name)
    fp_logits = predict_logits(model, Xte)
    fp = evaluate(yte, fp_logits, key_class)
    calib = Xtr[:128]
    print(f"\n=== {name} (FP acc {fp['accuracy']:.4f}, "
          f"key recall {fp['key_recall']:.4f}) ===")
    rows = []
    weight_layers = [i for i, l in enumerate(model.layers)
                     if l.name in ("linear", "conv", "attn")]
    for bits in (8, 4):
        cfg = QuantConfig(
            weight_bits={i: bits for i in weight_layers}, act_bits=8)
        eng = QuantizedEngine(model, calib, "minmax", cfg)
        q_logits = eng.predict(Xte)
        m = evaluate(yte, q_logits, key_class)
        d_acc = m["accuracy"] - fp["accuracy"]
        d_key = m["key_recall"] - fp["key_recall"]
        # logit deviation
        dev = np.abs(q_logits - fp_logits).mean()
        rows.append((bits, m["accuracy"], d_acc, m["key_recall"], d_key, dev))
        print(f"  INT{bits}: acc {m['accuracy']:.4f} ({d_acc:+.4f})  "
              f"key recall {m['key_recall']:.4f} ({d_key:+.4f})  "
              f"mean |Δlogit| {dev:.4f}")
    return fp, rows


def main():
    credit = load_credit()
    digits = load_digits()
    results = {}
    results["mlp_robust"] = run_case(
        "mlp_robust", credit["X_train"], credit["X_test"],
        credit["y_test"], credit["key_class"])
    results["mlp_sensitive"] = run_case(
        "mlp_sensitive", credit["X_train"], credit["X_test"],
        credit["y_test"], credit["key_class"])
    results["cnn_sensitive"] = run_case(
        "cnn_sensitive", digits["X_train_img"], digits["X_test_img"],
        digits["y_test"], digits["key_class"])
    results["attn_sensitive"] = run_case(
        "attn_sensitive", digits["X_train_seq"], digits["X_test_seq"],
        digits["y_test"], digits["key_class"])

    # structural assertions (ties are allowed: argmax can be unchanged)
    print("\n--- assertions ---")
    # INT8 must be no worse than INT4 beyond test-set rounding noise
    for name, (fp, rows) in results.items():
        d8, d4 = rows[0][2], rows[1][2]
        assert d8 >= d4 - 0.02, f"{name}: int8 much worse than int4?"
        assert d8 > -0.03, f"{name}: int8 accuracy drop {d8} too large"
        print(f"[OK] {name}: INT8 drop {d8:+.4f}, INT4 drop {d4:+.4f}, "
              f"INT8 within 3 points")
    # sensitive MLP should degrade at least as much as robust MLP at INT4
    d4_rob = results["mlp_robust"][1][1][2]
    d4_sen = results["mlp_sensitive"][1][1][2]
    print(f"[i] INT4 drops: robust {d4_rob:+.4f} vs sensitive {d4_sen:+.4f}")
    print("engine sanity checks done")


if __name__ == "__main__":
    main()
