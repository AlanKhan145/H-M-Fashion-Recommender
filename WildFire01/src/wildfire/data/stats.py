from __future__ import annotations
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from .constants import DATA_SIZE, ENV11

AUTOTUNE = tf.data.AUTOTUNE

def _feature_spec(size: int, keys: list[str]):
    return {k: tf.io.FixedLenFeature([size, size], tf.float32) for k in keys}

def _make_env_stack_parser(size: int, env_keys: list[str]):
    spec = _feature_spec(size, env_keys)
    def _parse(ex):
        p = tf.io.parse_single_example(ex, spec)
        return tf.stack([p[k] for k in env_keys], axis=-1)  # (H,W,F)
    return _parse

def _reservoir_update(reservoir: np.ndarray, seen: int, new_vals: np.ndarray, rng: np.random.Generator):
    K = reservoir.shape[0]
    new_vals = np.asarray(new_vals, dtype=np.float64)
    if new_vals.size == 0:
        return reservoir, seen
    if seen < K:
        fill = min(K - seen, new_vals.size)
        reservoir[seen:seen+fill] = new_vals[:fill]
        seen += fill
        new_vals = new_vals[fill:]
        if new_vals.size == 0:
            return reservoir, seen
    for v in new_vals:
        j = rng.integers(0, seen + 1)
        if j < K:
            reservoir[j] = v
        seen += 1
    return reservoir, seen

def compute_train_stats(
    train_pattern: str,
    save_json: str,
    size: int = DATA_SIZE,
    env_keys: list[str] = ENV11,
    p_low: float = 0.5,
    p_high: float = 99.5,
    sample_per_tile: int = 512,
    reservoir_size: int = 200_000,
    max_tiles: int | None = None,
    batch_tiles: int = 8,
    seed: int = 42,
    force: bool = False,
) -> dict:
    save_path = Path(save_json)
    if save_path.exists() and not force:
        with open(save_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: tuple(v) for k, v in raw.items()}

    parse = _make_env_stack_parser(size, env_keys)
    ds = tf.data.Dataset.list_files(train_pattern, shuffle=False)
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
    if max_tiles is not None:
        ds = ds.take(int(max_tiles))
    ds = ds.batch(int(batch_tiles)).prefetch(AUTOTUNE)

    rng = np.random.default_rng(seed)
    reservoirs = {k: np.empty(reservoir_size, np.float64) for k in env_keys}
    seen = {k: 0 for k in env_keys}

    F = len(env_keys)
    for xb in ds:
        x = xb.numpy().astype(np.float64)  # (B,H,W,F)
        B, H, W, _ = x.shape
        HW = H * W
        take = min(sample_per_tile, HW)
        x2 = x.reshape(B, HW, F)
        for b in range(B):
            idx = rng.integers(0, HW, size=take, endpoint=False)
            samp = x2[b, idx, :]
            for f, k in enumerate(env_keys):
                v = samp[:, f]
                v = v[np.isfinite(v)]
                reservoirs[k], seen[k] = _reservoir_update(reservoirs[k], seen[k], v, rng)

    bounds = {}
    for k in env_keys:
        n = min(seen[k], reservoir_size)
        arr = reservoirs[k][:n]
        lo = float(np.percentile(arr, p_low))
        hi = float(np.percentile(arr, p_high))
        if hi <= lo:
            hi = lo + 1e-6
        bounds[k] = (lo, hi)

    # mean/std after clip
    sum_ = np.zeros(F, np.float64)
    sumsq = np.zeros(F, np.float64)
    cnt = np.zeros(F, np.int64)
    lo = np.array([bounds[k][0] for k in env_keys], np.float64)
    hi = np.array([bounds[k][1] for k in env_keys], np.float64)

    ds2 = tf.data.Dataset.list_files(train_pattern, shuffle=False)
    ds2 = ds2.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds2 = ds2.map(parse, num_parallel_calls=AUTOTUNE)
    if max_tiles is not None:
        ds2 = ds2.take(int(max_tiles))
    ds2 = ds2.batch(int(batch_tiles)).prefetch(AUTOTUNE)

    for xb in ds2:
        x = xb.numpy().astype(np.float64)
        x = np.clip(x, lo.reshape(1,1,1,F), hi.reshape(1,1,1,F))
        flat = x.reshape(-1, F)
        m = np.isfinite(flat)
        for f in range(F):
            vv = flat[m[:, f], f]
            if vv.size:
                sum_[f] += vv.sum(dtype=np.float64)
                sumsq[f] += (vv*vv).sum(dtype=np.float64)
                cnt[f] += vv.size

    mean = sum_ / np.maximum(cnt, 1)
    var  = sumsq / np.maximum(cnt, 1) - mean * mean
    std  = np.sqrt(np.maximum(var, 1e-12))

    stats = {}
    for f, k in enumerate(env_keys):
        mn, mx = bounds[k]
        stats[k] = (float(mn), float(mx), float(mean[f]), float(std[f] if std[f] > 0 else 1.0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({k: list(v) for k, v in stats.items()}, f, ensure_ascii=False, indent=2)

    return stats
