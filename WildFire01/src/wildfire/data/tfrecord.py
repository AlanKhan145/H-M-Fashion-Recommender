from __future__ import annotations
import tensorflow as tf
import numpy as np
from pathlib import Path

from .constants import DATA_SIZE, ENV11, PREV_KEY, NEXT_KEY, INPUT_FEATURES, OUTPUT_FEATURES, C_IN

AUTOTUNE = tf.data.AUTOTUNE

def feature_spec(size: int, keys: list[str]):
    return {k: tf.io.FixedLenFeature([size, size], tf.float32) for k in keys}

def load_stats(stats_json: str) -> dict:
    import json
    with open(stats_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}

def clip_norm_tf(x: tf.Tensor, key: str, stats: dict) -> tf.Tensor:
    mn, mx, mean, std = stats[key]
    x = tf.clip_by_value(x, mn, mx)
    return (x - mean) / (std + 1e-6)

def parse_to_tile(example_proto: tf.Tensor, stats: dict):
    keys = INPUT_FEATURES + OUTPUT_FEATURES
    p = tf.io.parse_single_example(example_proto, feature_spec(DATA_SIZE, keys))

    x_env = tf.stack([clip_norm_tf(p[k], k, stats) for k in ENV11], axis=-1)  # (64,64,11)

    prev_raw = tf.ensure_shape(p[PREV_KEY], [DATA_SIZE, DATA_SIZE])
    ndvi_raw = tf.ensure_shape(p["NDVI"],   [DATA_SIZE, DATA_SIZE])
    y_raw    = tf.ensure_shape(p[NEXT_KEY], [DATA_SIZE, DATA_SIZE])

    prev_bin = tf.where(prev_raw > 0.0, 1.0, 0.0)
    prev_bin = tf.expand_dims(prev_bin, axis=-1)

    x = tf.concat([x_env, prev_bin], axis=-1)
    x = tf.ensure_shape(x, [DATA_SIZE, DATA_SIZE, C_IN])
    return x, prev_raw, ndvi_raw, y_raw

def tile_dataset(file_pattern: str | list[str], stats: dict, max_tiles=None, shuffle_files=False, seed=42):
    if isinstance(file_pattern, (list, tuple)):
        files = list(file_pattern)
    else:
        files = tf.io.gfile.glob(str(file_pattern))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy TFRecord files: {file_pattern}")

    ds_files = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_files:
        ds_files = ds_files.shuffle(len(files), seed=seed, reshuffle_each_iteration=False)

    cycle = min(16, len(files))
    ds = ds_files.interleave(lambda f: tf.data.TFRecordDataset(f), cycle_length=cycle,
                             num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.map(lambda ex: parse_to_tile(ex, stats), num_parallel_calls=AUTOTUNE)
    if max_tiles is not None:
        ds = ds.take(int(max_tiles))
    return ds.prefetch(AUTOTUNE)

# -------- patch sampling (balanced) --------
def valid_core_mask(prev_raw, y_raw, ndvi_raw, k, use_fuel_mask=False, ndvi_thr_raw=0.0):
    H = tf.shape(y_raw)[0]; W = tf.shape(y_raw)[1]
    r = k // 2
    valid = tf.logical_and(y_raw >= 0.0, prev_raw >= 0.0)
    yy = tf.range(H)[:, None]
    xx = tf.range(W)[None, :]
    core = tf.logical_and(tf.logical_and(yy >= r, yy < H - r),
                          tf.logical_and(xx >= r, xx < W - r))
    valid = tf.logical_and(valid, core)
    if use_fuel_mask:
        valid = tf.logical_and(valid, ndvi_raw > ndvi_thr_raw)
    return valid

def gather_patches_at_flat_idx(x, flat_idx, k):
    H = tf.shape(x)[0]
    W = tf.shape(x)[1]
    r = k // 2
    xpad = tf.pad(x, [[r, r], [r, r], [0, 0]])
    ci = flat_idx // W
    cj = flat_idx % W
    centers = tf.stack([ci + r, cj + r], axis=-1)
    offs = tf.range(-r, r + 1, dtype=tf.int32)
    gy, gx = tf.meshgrid(offs, offs, indexing="ij")
    offsets = tf.stack([gy, gx], axis=-1)
    idx2 = centers[:, None, None, :] + offsets[None, :, :, :]
    patches = tf.gather_nd(xpad, idx2)
    return patches

def sample_patches_from_tile(x, prev_raw, ndvi_raw, y_raw, k=7, n_pos=256, n_neg=256):
    valid = valid_core_mask(prev_raw, y_raw, ndvi_raw, k=k)
    y_bin = tf.cast(y_raw > 0.0, tf.float32)
    valid_flat = tf.reshape(valid, [-1])
    y_flat = tf.reshape(y_bin, [-1])

    pos_idx = tf.cast(tf.where(tf.logical_and(valid_flat, y_flat > 0.5))[:, 0], tf.int32)
    neg_idx = tf.cast(tf.where(tf.logical_and(valid_flat, y_flat <= 0.5))[:, 0], tf.int32)
    pool = tf.cast(tf.where(valid_flat)[:, 0], tf.int32)
    pool = tf.cond(tf.shape(pool)[0] > 0, lambda: pool, lambda: tf.range(tf.size(y_flat), dtype=tf.int32))

    def take_k(idxs, k_take):
        n = tf.shape(idxs)[0]
        idxs2 = tf.cond(n > 0, lambda: idxs, lambda: pool)
        n2 = tf.shape(idxs2)[0]
        ridx = tf.random.uniform([k_take], 0, n2, dtype=tf.int32)
        return tf.gather(idxs2, ridx)

    idx_take = tf.concat([take_k(pos_idx, n_pos), take_k(neg_idx, n_neg)], axis=0)
    x_p = gather_patches_at_flat_idx(x, idx_take, k)
    y_p = tf.gather(y_flat, idx_take)
    sw  = tf.ones_like(y_p, tf.float32)
    return x_p, y_p, sw

def make_train_patch_dataset(
    file_pattern: str,
    stats: dict,
    batch_size=256,
    k=7,
    n_pos=256,
    n_neg=256,
    shuffle_files=True,
    shuffle_buf=32768,
    repeat=True,
    seed=42
):
    files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle_files, seed=seed)
    ds = files.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTOTUNE, deterministic=False)
    if shuffle_buf and shuffle_buf > 0:
        ds = ds.shuffle(shuffle_buf, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda ex: parse_to_tile(ex, stats), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, prev, ndvi, y: sample_patches_from_tile(x, prev, ndvi, y, k=k, n_pos=n_pos, n_neg=n_neg),
                num_parallel_calls=AUTOTUNE)
    ds = ds.unbatch()
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds

# -------- inference maps (keras model gives logits) --------
def infer_core_logit_map_valid(x_tile, model, k=7, batch_patches=8192):
    x = tf.convert_to_tensor(x_tile, tf.float32)
    xt = x[None, ...]
    patches = tf.image.extract_patches(
        images=xt, sizes=[1, k, k, 1], strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1], padding="VALID"
    )
    Hc = int(tf.shape(patches)[1].numpy())
    Wc = int(tf.shape(patches)[2].numpy())
    C  = int(tf.shape(xt)[-1].numpy())
    patches = tf.reshape(patches, [-1, k, k, C])
    n = int(tf.shape(patches)[0].numpy())

    outs = []
    bs = int(batch_patches)
    for s in range(0, n, bs):
        y = model(patches[s:s+bs], training=False)
        outs.append(tf.reshape(y, [-1]))
    logit_flat = tf.concat(outs, axis=0)
    return tf.reshape(logit_flat, [Hc, Wc])

def infer_full_map_logit_same(x_tile, model, k=7, batch_patches=8192):
    x = tf.convert_to_tensor(x_tile, tf.float32)
    xt = x[None, ...]
    patches = tf.image.extract_patches(
        images=xt, sizes=[1, k, k, 1], strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1], padding="SAME"
    )
    H = int(tf.shape(patches)[1].numpy())
    W = int(tf.shape(patches)[2].numpy())
    C = int(tf.shape(xt)[-1].numpy())
    patches = tf.reshape(patches, [-1, k, k, C])
    n = int(tf.shape(patches)[0].numpy())

    outs = []
    bs = int(batch_patches)
    for s in range(0, n, bs):
        y = model(patches[s:s+bs], training=False)
        outs.append(tf.reshape(y, [-1]))
    logit_flat = tf.concat(outs, axis=0)
    return tf.reshape(logit_flat, [H, W]).numpy()
