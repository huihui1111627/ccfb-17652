"""Run the full quantization analysis for every trained model.

Produces experiments/results/results.json consumed by the web app:
  * FP32 / all-INT8 / all-INT4 / tool-selected mixed / per-channel INT8
  * overall and per-class (esp. key-class) metrics for every scheme
  * model size and integer-cycle cost for every scheme
  * calibration-set variant comparison (how a bad calibration set hurts)
  * per-layer SQNR, single-layer ablation, channel outliers
  * mixed-precision Pareto front and the cheapest feasible plan
  * per-sample layer-by-layer error traces

Run:
    python3 -I tools/run_iso.py experiments/02_run_analysis.py
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from quant_tool.data import load_credit, load_digits
from quant_tool.models.checkpoint import load_model
from quant_tool.models.training import predict_logits
from quant_tool.quant.engine import QuantizedEngine, QuantConfig
from quant_tool.quant.cost_model import size_and_cost
from quant_tool.metrics import evaluate
from quant_tool.analysis.sensitivity import (
    weight_layer_indices, layer_sqnr, single_layer_ablation,
    channel_outliers, trace_sample)
from quant_tool.analysis.mixed import search_mixed_precision, uniform_plan
from quant_tool.analysis.calibration import calibration_variants

OUT = os.path.join(ROOT, "experiments", "results")
os.makedirs(OUT, exist_ok=True)

ACC_BUDGET, KEY_BUDGET = 0.02, 0.05


def model_dataset(name, credit, digits):
    if name.startswith("mlp"):
        return (credit["X_train"], credit["X_test"], credit["y_test"],
                credit["key_class"], credit["class_names"],
                (1, credit["X_train"].shape[1]))
    if name == "cnn_sensitive":
        return (digits["X_train_img"], digits["X_test_img"],
                digits["y_test"], digits["key_class"],
                digits["class_names"], (1, 1, 8, 8))
    return (digits["X_train_seq"], digits["X_test_seq"], digits["y_test"],
            digits["key_class"], digits["class_names"], (1, 8, 8))


def scheme_row(label, cfg, model, calib, X, y, key, shape, granularity):
    eng = QuantizedEngine(model, calib, "minmax", cfg)
    m = evaluate(y, eng.predict(X), key)
    fp = evaluate(y, predict_logits(model, X), key)
    cost = size_and_cost(model, cfg, shape)
    return {
        "label": label,
        "weight_bits": {str(k): v for k, v in cfg.weight_bits.items()},
        "granularity": granularity,
        "accuracy": m["accuracy"],
        "acc_drop": m["accuracy"] - fp["accuracy"],
        "key_recall": m["key_recall"],
        "key_recall_drop": m["key_recall"] - fp["key_recall"],
        "per_class": m["per_class"],
        "total_bytes": cost["total_bytes"],
        "estimated_cycles": cost["estimated_cycles"],
        "per_layer_cost": cost["layers"],
    }


def pick_samples(y, key, n_each=2):
    """Representative sample indices: key-class and ordinary class."""
    rng = np.random.default_rng(5)
    key_idx = np.where(y == key)[0]
    other_idx = np.where(y != key)[0]
    sel = list(rng.choice(key_idx, min(n_each, len(key_idx)), replace=False))
    sel += list(rng.choice(other_idx, min(n_each, len(other_idx)),
                           replace=False))
    return sorted(int(i) for i in sel)


def analyze(name, credit, digits):
    print(f"\n=== analyzing {name} ===")
    Xtr, Xte, yte, key, class_names, shape = model_dataset(
        name, credit, digits)
    model, meta = load_model(name)
    widx = weight_layer_indices(model)
    calib = Xtr[:128]
    fp = evaluate(yte, predict_logits(model, Xte), key)

    # ---- uniform schemes (per-tensor) ----
    schemes = {}
    for bits, lab in ((8, "all_int8"), (4, "all_int4")):
        cfg = QuantConfig({i: bits for i in widx}, 8,
                          weight_granularity="per_tensor")
        schemes[lab] = scheme_row(lab, cfg, model, calib, Xte, yte, key,
                                  shape, "per_tensor")
        print(f"  {lab}: acc {schemes[lab]['accuracy']:.3f} "
              f"({schemes[lab]['acc_drop']:+.3f}) key "
              f"{schemes[lab]['key_recall']:.3f} "
              f"({schemes[lab]['key_recall_drop']:+.3f})")

    # per-channel INT8 to show the outlier remediation
    cfg_pc = QuantConfig({i: 8 for i in widx}, 8,
                         weight_granularity="per_channel")
    schemes["per_channel_int8"] = scheme_row(
        "per_channel_int8", cfg_pc, model, calib, Xte, yte, key, shape,
        "per_channel")

    # ---- calibration set comparison (all-INT8 per-tensor) ----
    cal_sets = calibration_variants(Xtr, _train_y(name, credit, digits), key,
                                    n=128)
    calib_compare = {}
    for vname, vX in cal_sets.items():
        cfg = QuantConfig({i: 8 for i in widx}, 8,
                          weight_granularity="per_tensor")
        eng = QuantizedEngine(model, vX, "minmax", cfg)
        m = evaluate(yte, eng.predict(Xte), key)
        calib_compare[vname] = {
            "n_calib": int(len(vX)),
            "accuracy": m["accuracy"],
            "acc_drop": m["accuracy"] - fp["accuracy"],
            "key_recall": m["key_recall"],
            "key_recall_drop": m["key_recall"] - fp["key_recall"],
        }
    print("  calibration variants:",
          {k: round(v["key_recall_drop"], 3)
           for k, v in calib_compare.items()})

    # ---- range-determination method (observer) comparison, all-INT8 ----
    observer_compare = {}
    for method in ("minmax", "pct99.9", "pct99.5", "pct99.0"):
        cfg = QuantConfig({i: 8 for i in widx}, 8,
                          weight_granularity="per_tensor")
        eng = QuantizedEngine(model, calib, method, cfg)
        m = evaluate(yte, eng.predict(Xte), key)
        observer_compare[method] = {
            "accuracy": m["accuracy"],
            "acc_drop": m["accuracy"] - fp["accuracy"],
            "key_recall": m["key_recall"],
            "key_recall_drop": m["key_recall"] - fp["key_recall"],
        }
    print("  observer methods:",
          {k: round(v["key_recall_drop"], 3)
           for k, v in observer_compare.items()})

    # ---- layer sensitivity ----
    abl = single_layer_ablation(model, calib, Xte, yte, key, probe_bits=4)
    eng8 = QuantizedEngine(
        model, calib, "minmax",
        QuantConfig({i: 8 for i in widx}, 8,
                    weight_granularity="per_tensor"))
    sqnr_rows = layer_sqnr(eng8, Xte)
    outliers = channel_outliers(model, calib)

    # ---- mixed precision search ----
    search = search_mixed_precision(
        model, calib, Xte, yte, key, ACC_BUDGET, KEY_BUDGET,
        granularity="per_tensor", input_shape=shape)
    mixed = None
    if search["cheapest_feasible"]:
        ch = search["cheapest_feasible"]
        cfg = QuantConfig(
            {int(k): v for k, v in ch["weight_bits"].items()}, 8,
            weight_granularity="per_tensor")
        mixed = scheme_row("mixed", cfg, model, calib, Xte, yte, key, shape,
                           "per_tensor")
        print(f"  mixed plan {mixed['weight_bits']}: acc "
              f"{mixed['acc_drop']:+.3f} key {mixed['key_recall_drop']:+.3f} "
              f"cycles {int(mixed['estimated_cycles'])}")
    pareto = [{k: (v if not isinstance(v, dict) else v) for k, v in
               p.items() if k != "per_layer"} for p in search["pareto"]]
    # keep several cheapest feasible alternatives so different selections
    # under the same budget can be compared side by side
    top_feasible = []
    for p in search["all_feasible"][:8]:
        top_feasible.append({
            "weight_bits": {str(k): v for k, v in p["weight_bits"].items()},
            "acc_drop": p["acc_drop"], "key_recall_drop": p["key_recall_drop"],
            "total_bytes": p["total_bytes"],
            "estimated_cycles": p["estimated_cycles"],
        })

    # ---- per-sample traces (all-INT8 engine) ----
    sample_ids = pick_samples(yte, key)
    traces = []
    for sid in sample_ids:
        rows, logits_q = trace_sample(eng8, Xte[sid])
        logits_fp = predict_logits(model, Xte[sid][None])[0]
        # keep trace compact: drop raw flat arrays beyond layer summary,
        # but retain them for the last few layers where it matters
        for r in rows:
            if len(r["fp_flat"]) > 64:
                r["fp_flat"] = r["fp_flat"][:64]
                r["q_flat"] = r["q_flat"][:64]
        traces.append({
            "sample_id": sid,
            "label": int(yte[sid]),
            "is_key": bool(yte[sid] == key),
            "pred_fp": int(logits_fp.argmax()),
            "pred_q": int(logits_q.argmax()),
            "logit_gap": float(np.abs(logits_fp - logits_q).mean()),
            "trace": rows,
        })

    return {
        "meta": {"name": name, "checkpoint_meta": meta,
                 "dataset": meta.get("dataset"),
                 "key_class": int(key), "class_names": class_names,
                 "input_shape": list(shape)},
        "fp": fp,
        "schemes": schemes,
        "mixed": mixed,
        "calibration_compare": calib_compare,
        "observer_compare": observer_compare,
        "layer_sensitivity": {
            "ablation": abl["ablation"],
            "all_int8": {"accuracy": abl["all_int8"]["accuracy"],
                         "key_recall": abl["all_int8"]["key_recall"]},
            "all_int4": {"accuracy": abl["all_int4"]["accuracy"],
                         "key_recall": abl["all_int4"]["key_recall"]},
            "sqnr": sqnr_rows,
        },
        "outliers": outliers,
        "mixed_search": {
            "n_evaluated": search["n_evaluated"],
            "n_feasible": search["n_feasible"],
            "acc_budget": ACC_BUDGET, "key_budget": KEY_BUDGET,
            "pareto": [{**p, "weight_bits": {
                str(k): v for k, v in p["weight_bits"].items()}}
                for p in pareto],
            "top_feasible": top_feasible,
        },
        "sample_traces": traces,
    }


def _train_y(name, credit, digits):
    if name.startswith("mlp"):
        return credit["y_train"]
    return digits["y_train"]


def main():
    credit = load_credit()
    digits = load_digits()
    results = {}
    for name in ("mlp_robust", "mlp_sensitive", "cnn_sensitive",
                 "attn_sensitive"):
        results[name] = analyze(name, credit, digits)
    path = os.path.join(OUT, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print("\nwrote", path, f"({os.path.getsize(path)//1024} KB)")


if __name__ == "__main__":
    main()
