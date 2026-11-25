"""Rebuild trained models from checkpoints."""
from __future__ import annotations

import json
import os
import numpy as np

from .factory import build_mlp, build_cnn, build_attention

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "experiments", "models")


def load_model(name, model_dir=MODEL_DIR):
    """name in {mlp_robust, mlp_sensitive, cnn_sensitive, attn_sensitive}."""
    path = os.path.join(model_dir, name + ".npz")
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    arch = meta["arch"]
    if arch == "mlp":
        model = build_mlp(meta["in_dim"], 2, meta["hidden"],
                          seed=hash(name) & 0xffff)
    elif arch == "cnn":
        model = build_cnn(10, seed=hash(name) & 0xffff)
    elif arch == "attn":
        model = build_attention(10, meta["dim"], seed=hash(name) & 0xffff)
    else:
        raise ValueError(arch)
    params, _ = model.all_params_grads()
    keys = [k for k in z.files if k != "meta"]
    # np.savez positional: arr_0, arr_1, ... in params() order
    for p, k in zip(params, sorted(keys, key=lambda s: int(s.split("_")[1]))):
        p[:] = z[k]
    return model, meta
