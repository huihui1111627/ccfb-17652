"""Train all four models on real data and save FP32 checkpoints.

Run:
    python3 -I tools/run_iso.py experiments/01_train_models.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_tool.data import load_credit, load_digits
from quant_tool.models.factory import build_mlp, build_cnn, build_attention
from quant_tool.models.training import (train_model, accuracy,
                                        inject_linear_outliers,
                                        inject_conv_outliers,
                                        verify_fp_equivalence,
                                        predict_logits)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "experiments", "models")
os.makedirs(OUT, exist_ok=True)


def save_checkpoint(path, model, meta):
    params, _ = model.all_params_grads()
    np.savez(os.path.join(path), *[p.astype(np.float32) for p in params],
             meta=np.array([__import__("json").dumps(meta)]))


def main():
    credit = load_credit()
    digits = load_digits()
    summary = {}

    # ---------- 1. MLP robust (credit-g) ----------
    print("[1/4] mlp_robust on credit-g")
    m = build_mlp(credit["X_train"].shape[1], 2, (64, 32), seed=1)
    train_model(m, credit["X_train"], credit["y_train"], epochs=80,
                lr=1e-3, batch=64, seed=11,
                X_val=credit["X_test"], y_val=credit["y_test"], verbose=True)
    acc = accuracy(m, credit["X_test"], credit["y_test"])
    save_checkpoint(os.path.join(OUT, "mlp_robust.npz"), m,
                    {"arch": "mlp", "in_dim": 61, "hidden": [64, 32],
                     "dataset": "credit-g"})
    summary["mlp_robust"] = acc
    print(f"  test acc {acc:.4f}\n")

    # ---------- 2. MLP sensitive (credit-g) + outlier channels ----------
    print("[2/4] mlp_sensitive on credit-g (outlier injection)")
    m2 = build_mlp(61, 2, (64, 32), seed=2)
    train_model(m2, credit["X_train"], credit["y_train"], epochs=80,
                lr=1e-3, batch=64, seed=12,
                X_val=credit["X_test"], y_val=credit["y_test"], verbose=True)
    before = predict_logits(m2, credit["X_test"])
    rng = np.random.default_rng(7)
    # Strong, late outlier channels. Layer 2 is the last hidden Linear
    # (64->32), directly feeding the logits layer, so its wide channels are
    # no longer diluted by another nonlinear block. Large positive scales
    # widen the per-tensor weight grid and the activation range without
    # changing the FP function (exact pair transform, verified below).
    chans1 = sorted(rng.choice(64, size=5, replace=False))
    rep1 = inject_linear_outliers(m2, 0, chans1, scale=12.0)
    chans2 = sorted(rng.choice(32, size=6, replace=False))
    rep2 = inject_linear_outliers(m2, 2, chans2, scale=16.0)
    after = predict_logits(m2, credit["X_test"])
    eq = verify_fp_equivalence(before, after)
    print(f"  FP equivalence: {eq}")
    acc2 = accuracy(m2, credit["X_test"], credit["y_test"])
    save_checkpoint(os.path.join(OUT, "mlp_sensitive.npz"), m2,
                    {"arch": "mlp", "in_dim": 61, "hidden": [64, 32],
                     "dataset": "credit-g",
                     "outlier_report": rep1 + rep2})
    summary["mlp_sensitive"] = acc2
    print(f"  test acc {acc2:.4f}\n")

    # ---------- 3. CNN sensitive (digits) + outlier feature maps ----------
    print("[3/4] cnn_sensitive on digits (outlier feature maps)")
    c = build_cnn(10, seed=3)
    train_model(c, digits["X_train_img"], digits["y_train"], epochs=60,
                lr=1e-3, batch=64, seed=13,
                X_val=digits["X_test_img"], y_val=digits["y_test"],
                verbose=True)
    before = predict_logits(c, digits["X_test_img"])
    rng2 = np.random.default_rng(8)
    cchans = sorted(rng2.choice(32, size=4, replace=False))
    crep = inject_conv_outliers(c, 2, cchans, scale=10.0, inp_size=8)
    # also widen a few channels of the first conv
    cchans0 = sorted(rng2.choice(16, size=3, replace=False))
    crep0 = inject_conv_outliers(c, 0, cchans0, scale=8.0, inp_size=8)
    after = predict_logits(c, digits["X_test_img"])
    eqc = verify_fp_equivalence(before, after)
    print(f"  FP equivalence: {eqc}")
    acc3 = accuracy(c, digits["X_test_img"], digits["y_test"])
    save_checkpoint(os.path.join(OUT, "cnn_sensitive.npz"), c,
                    {"arch": "cnn", "dataset": "digits",
                     "outlier_report": crep0 + crep})
    summary["cnn_sensitive"] = acc3
    print(f"  test acc {acc3:.4f}\n")

    # ---------- 4. Attention (digits sequences) + FFN outliers ----------
    print("[4/4] attn_sensitive on digits sequences")
    a = build_attention(10, 32, seed=4)
    train_model(a, digits["X_train_seq"], digits["y_train"], epochs=90,
                lr=1e-3, batch=64, seed=14,
                X_val=digits["X_test_seq"], y_val=digits["y_test"],
                verbose=True)
    before = predict_logits(a, digits["X_test_seq"])
    # FFN Linear is layer index 3 (MeanPool=2, Linear 32->32 = 3)
    arep = inject_linear_outliers(a, 3, [3, 11, 24], scale=9.0)
    after = predict_logits(a, digits["X_test_seq"])
    eqa = verify_fp_equivalence(before, after)
    print(f"  FP equivalence: {eqa}")
    acc4 = accuracy(a, digits["X_test_seq"], digits["y_test"])
    save_checkpoint(os.path.join(OUT, "attn_sensitive.npz"), a,
                    {"arch": "attn", "dim": 32, "dataset": "digits",
                     "outlier_report": arep})
    summary["attn_sensitive"] = acc4
    print(f"  test acc {acc4:.4f}\n")

    print("All models saved to", OUT)
    for k, v in summary.items():
        print(f"  {k:15s} acc={v:.4f}")


if __name__ == "__main__":
    main()
