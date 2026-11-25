"""Real datasets with built-in minority / key-class structure.

Three datasets, all downloaded once into an isolated data cache:

1. ``credit-g`` (OpenML id 31): German credit risk, 1000 rows, mixed
   categorical/numeric features. Key class = "bad" (high credit risk),
   ~30% of rows. Used for the MLP (risk-control scenario from the brief).

2. ``digits`` (sklearn, UCI pen-based handwritten digits): 1797 8x8
   grayscale images. Used for the CNN (image view) and the attention model
   (sequence-of-pixels view). We subsample a "rare key digit" so its share
   is small, mirroring the risk-control setup on visual data.

Everything is normalized to float32; one-hot is generated only for
categorical columns. No file in the data cache is ever imported.
"""
from __future__ import annotations

import os
import numpy as np

# __file__ = workspace/quant_tool/data.py: two dirnames -> workspace root
DATA_CACHE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data_cache")
os.makedirs(DATA_CACHE, exist_ok=True)


def _train_test_split(n, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    return idx[n_test:], idx[:n_test]


def load_credit(seed=0):
    """German credit dataset.

    Returns dict with keys:
      X_train/X_test: float32 (n, d), standardized using train stats
      y_train/y_test: int {0,1}, 1 = BAD (high risk, the key class)
      feature_names, key_class=1, name
    """
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=31, as_frame=True, data_home=DATA_CACHE,
                      parser="auto")
    df = ds.frame
    target = df.pop(ds.target.name)
    # 1 = bad/high-risk, 0 = good
    cats = sorted(target.unique())  # ['bad', 'good'] lexically
    y = (target.astype(str).to_numpy() == "bad").astype(np.int64)

    cols = []
    names = []
    for col in df.columns:
        v = df[col]
        if v.dtype.kind in "OUSb" or str(v.dtype).startswith("category"):
            vals = v.astype(str).to_numpy()
            levels = sorted(np.unique(vals))
            oh = (vals[:, None] == np.array(levels)[None, :]).astype(np.float32)
            cols.append(oh)
            names.extend(f"{col}={lv}" for lv in levels)
        else:
            cols.append(v.to_numpy(dtype=np.float32)[:, None])
            names.append(str(col))
    X = np.concatenate(cols, axis=1).astype(np.float32)

    tr, te = _train_test_split(len(X), 0.25, seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    X = (X - mu) / sd
    return {
        "name": "credit-g",
        "X_train": X[tr], "X_test": X[te],
        "y_train": y[tr], "y_test": y[te],
        "feature_names": names,
        "key_class": 1,
        "class_names": ["good", "bad"],
    }


def load_digits(key_digit=8, key_frac=0.08, seed=0):
    """Handwritten digits with one rare, key digit.

    The key digit's train/test rows are subsampled (without touching the
    other classes) so it makes up roughly ``key_frac`` of the data — a
    stand-in for rare high-risk samples. Test set uses the same subsampling
    so recall on the rare class is estimable.

    Returns both the image view (n,1,8,8) for the CNN and the sequence view
    (n,8,8) — 8 rows treated as a length-8 sequence of 8-d tokens — for the
    attention model.
    """
    from sklearn.datasets import load_digits as _ld
    d = _ld()
    X_img = (d.data.astype(np.float32) / 16.0)  # (n,64) in [0,1]
    y = d.target.astype(np.int64)
    rng = np.random.default_rng(seed)

    keep = np.ones(len(y), dtype=bool)
    key_idx = np.where(y == key_digit)[0]
    other_n = int(np.sum(y != key_digit))
    n_keep = max(20, int(round(key_frac * other_n / (1 - key_frac))))
    drop = rng.choice(key_idx, size=len(key_idx) - min(n_keep, len(key_idx)),
                      replace=False)
    keep[drop] = False
    X_img, y = X_img[keep], y[keep]

    tr, te = _train_test_split(len(y), 0.25, seed + 1)
    n = X_img.reshape(-1, 1, 8, 8)
    seq = X_img.reshape(-1, 8, 8)
    return {
        "name": f"digits(key={key_digit},frac~{key_frac})",
        "X_train_img": n[tr], "X_test_img": n[te],
        "X_train_seq": seq[tr], "X_test_seq": seq[te],
        "y_train": y[tr], "y_test": y[te],
        "key_class": int(key_digit),
        "class_names": [str(i) for i in range(10)],
    }


if __name__ == "__main__":
    # Smoke test / dataset fetcher, run through tools/run_iso.py.
    c = load_credit()
    print("credit:", c["X_train"].shape, c["X_test"].shape,
          "bad train share:", round(c["y_train"].mean(), 3))
    d = load_digits()
    print("digits:", d["X_train_img"].shape, d["X_test_img"].shape,
          "key share train:",
          round((d["y_train"] == d["key_class"]).mean(), 3))
