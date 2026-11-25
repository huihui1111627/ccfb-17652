"""Mixed-precision plan search under dual accuracy constraints.

Every weight layer independently chooses FP32 / INT8 / INT4. A plan is
feasible only if BOTH hold on the evaluation set:

    accuracy drop        <= acc_budget
    key-class recall drop <= key_budget     (the risk-control constraint)

Among feasible plans we keep the Pareto front over estimated integer
cycles and model bytes, so different selections under the same budget can
be compared on speed and size rather than a single number.

All our small models have <= 4 weight layers (3^4 = 81 plans), so the
default is exact full enumeration; a beam search handles larger layer
counts.
"""
from __future__ import annotations

import itertools
import numpy as np

from ..quant.engine import QuantizedEngine, QuantConfig
from ..quant.cost_model import size_and_cost
from ..metrics import evaluate
from ..models.training import predict_logits
from .sensitivity import weight_layer_indices

CHOICES = (4, 8, 32)


def _all_configs(layer_indices):
    for combo in itertools.product(CHOICES, repeat=len(layer_indices)):
        yield QuantConfig(dict(zip(layer_indices, combo)), act_bits=8,
                          weight_granularity="per_tensor")


def _beam_configs(layer_indices, width=12):
    """Beam over layer order, pruning to the `width` cheapest prefixes per
    achieved precision signature."""
    beams = [{}]
    for li in layer_indices:
        nxt = []
        for partial in beams:
            for b in CHOICES:
                cfg = dict(partial)
                cfg[li] = b
                nxt.append(cfg)
        # keep beam diverse: one entry per (bits-multiset) is hard; instead
        # cap by total precision cost deterministically
        nxt.sort(key=lambda c: sum(c.values()))
        beams = nxt[:width]
    for c in beams:
        yield QuantConfig(c, act_bits=8, weight_granularity="per_tensor")


def pareto_front(plans, keys=("estimated_cycles", "total_bytes")):
    """Keep plans not dominated on both cost dimensions (lower is better)."""
    front = []
    for p in plans:
        dominated = False
        for q in plans:
            if q is p:
                continue
            if all(q[k] <= p[k] for k in keys) and \
               any(q[k] < p[k] for k in keys):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: (p["estimated_cycles"], p["total_bytes"]))
    return front


def search_mixed_precision(model, calib, X, y, key_class,
                           acc_budget=0.02, key_budget=0.05,
                           granularity="per_tensor", observer="minmax",
                           input_shape=None, beam_width=12):
    fp = evaluate(y, predict_logits(model, X), key_class)
    widx = weight_layer_indices(model)
    configs = (_all_configs(widx) if len(widx) <= 6
               else _beam_configs(widx, beam_width))

    feasible, infeasible = [], []
    for cfg in configs:
        eng = QuantizedEngine(model, calib, observer, cfg)
        m = evaluate(y, eng.predict(X), key_class)
        cost = size_and_cost(
            model, cfg, input_shape or _first_shape(X))
        plan = {
            "weight_bits": dict(cfg.weight_bits),
            "accuracy": m["accuracy"],
            "acc_drop": m["accuracy"] - fp["accuracy"],
            "key_recall": m["key_recall"],
            "key_recall_drop": m["key_recall"] - fp["key_recall"],
            "total_bytes": cost["total_bytes"],
            "estimated_cycles": cost["estimated_cycles"],
            "per_layer": cost["layers"],
            "feasible": (m["accuracy"] - fp["accuracy"] >= -acc_budget and
                         m["key_recall"] - fp["key_recall"] >= -key_budget),
        }
        (feasible if plan["feasible"] else infeasible).append(plan)

    front = pareto_front(feasible)
    # cheapest feasible plan = first Pareto point (fewest cycles)
    cheapest = front[0] if front else None
    return {
        "fp": fp,
        "acc_budget": acc_budget,
        "key_budget": key_budget,
        "n_evaluated": len(feasible) + len(infeasible),
        "n_feasible": len(feasible),
        "pareto": front,
        "cheapest_feasible": cheapest,
        "all_feasible": sorted(feasible,
                               key=lambda p: p["estimated_cycles"]),
        "infeasible": sorted(infeasible,
                             key=lambda p: -p["estimated_cycles"]),
    }


def _first_shape(X):
    return (1,) + X.shape[1:]


def uniform_plan(model, bits, calib, X, y, key_class, granularity,
                 observer="minmax"):
    """Convenience: all weight layers at one precision, fully evaluated."""
    widx = weight_layer_indices(model)
    cfg = QuantConfig({i: bits for i in widx}, act_bits=8,
                      weight_granularity=granularity)
    eng = QuantizedEngine(model, calib, observer, cfg)
    m = evaluate(y, eng.predict(X), key_class)
    fp = evaluate(y, predict_logits(model, X), key_class)
    cost = size_and_cost(model, cfg, _first_shape(X))
    return {
        "label": f"all-int{bits}" if bits != 32 else "fp32",
        "weight_bits": dict(cfg.weight_bits),
        "accuracy": m["accuracy"],
        "acc_drop": m["accuracy"] - fp["accuracy"],
        "key_recall": m["key_recall"],
        "key_recall_drop": m["key_recall"] - fp["key_recall"],
        "per_class": m["per_class"],
        "total_bytes": cost["total_bytes"],
        "estimated_cycles": cost["estimated_cycles"],
    }
