from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ..data.tfrecord import make_train_patch_dataset

def collect_patch_samples(train_pat: str, stats: dict, k: int, batch_train: int, train_pos: int, train_neg: int, max_batches: int, seed=42):
    ds = make_train_patch_dataset(
        train_pat, stats=stats,
        batch_size=int(batch_train),
        k=int(k),
        n_pos=int(train_pos),
        n_neg=int(train_neg),
        shuffle_files=True,
        repeat=True,
        seed=int(seed),
    )
    Xs, ys = [], []
    it = iter(ds)
    for _ in range(int(max_batches)):
        xb, yb, _sw = next(it)
        Xs.append(xb.numpy().reshape(xb.shape[0], -1).astype(np.float32, copy=False))
        ys.append(yb.numpy().astype(np.int32, copy=False))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y

def train_lr(train_pat: str, stats: dict, k: int, cfg: dict):
    X, y = collect_patch_samples(
        train_pat, stats, k,
        batch_train=cfg["batch_train"],
        train_pos=cfg["train_pos"], train_neg=cfg["train_neg"],
        max_batches=cfg["max_train_batches"],
        seed=cfg.get("seed", 42)
    )
    model = LogisticRegression(
        C=float(cfg.get("C", 1.0)),
        max_iter=int(cfg.get("max_iter", 200)),
        n_jobs=-1,
        solver="lbfgs"
    )
    model.fit(X, y)
    return model

def train_rf(train_pat: str, stats: dict, k: int, cfg: dict):
    X, y = collect_patch_samples(
        train_pat, stats, k,
        batch_train=cfg["batch_train"],
        train_pos=cfg["train_pos"], train_neg=cfg["train_neg"],
        max_batches=cfg["max_train_batches"],
        seed=cfg.get("seed", 42)
    )
    model = RandomForestClassifier(
        n_estimators=int(cfg.get("n_estimators", 600)),
        max_depth=None if cfg.get("max_depth") in (None, "None") else int(cfg["max_depth"]),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
        max_features=cfg.get("max_features", "log2"),  # IMPORTANT
        n_jobs=-1,
        random_state=int(cfg.get("seed", 42)),
    )
    model.fit(X, y)
    return model
