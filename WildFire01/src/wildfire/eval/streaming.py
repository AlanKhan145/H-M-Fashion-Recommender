from __future__ import annotations
import numpy as np
import tensorflow as tf
from ..data.tfrecord import valid_core_mask, infer_core_logit_map_valid, tile_dataset
from ..models.sklearn_wrappers import sklearn_predict_proba, prob_to_logit

def _confusion_from_probs(y_true_np, p_prob_np, thr=0.5):
    pred = (p_prob_np >= thr)
    y1 = (y_true_np == 1)
    y0 = ~y1
    tp = int(np.sum(pred & y1))
    fp = int(np.sum(pred & y0))
    fn = int(np.sum((~pred) & y1))
    tn = int(np.sum((~pred) & y0))
    return tp, fp, fn, tn

def _metrics_from_conf(tp, fp, fn, tn):
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    acc       = (tp + tn) / (tp + fp + fn + tn + eps)
    return dict(IoU=float(iou), F1=float(f1), Precision=float(precision), Recall=float(recall), Accuracy=float(acc))

def full_eval_metrics_stream_keras(file_pattern, stats, model, k=7, thr_prob=0.5, max_tiles=None, batch_patches=8192):
    ds = tile_dataset(file_pattern, stats=stats, max_tiles=max_tiles)
    r = int(k) // 2
    TP = FP = FN = TN = 0
    N_valid = 0

    for x, prev_raw, ndvi_raw, y_raw in ds:
        valid = valid_core_mask(prev_raw, y_raw, ndvi_raw, k=int(k))
        logit_core = infer_core_logit_map_valid(x, model, k=int(k), batch_patches=int(batch_patches))
        y_core = y_raw[r:-r, r:-r] if r > 0 else y_raw
        v_core = valid[r:-r, r:-r] if r > 0 else valid

        yb = tf.boolean_mask(tf.cast(y_core > 0.0, tf.int32), v_core)
        lb = tf.boolean_mask(logit_core, v_core)
        if int(tf.size(yb).numpy()) == 0:
            continue

        pb = tf.math.sigmoid(tf.cast(lb, tf.float32)).numpy().astype(np.float32, copy=False)
        y_np = yb.numpy().astype(np.int32, copy=False)

        tp, fp, fn, tn = _confusion_from_probs(y_np, pb, thr=float(thr_prob))
        TP += tp; FP += fp; FN += fn; TN += tn
        N_valid += int(y_np.shape[0])

    m = _metrics_from_conf(TP, FP, FN, TN)
    return {**m, "TP": TP, "FP": FP, "FN": FN, "TN": TN, "N_valid": N_valid, "thr_prob": float(thr_prob)}

def _infer_core_prob_map_valid_sklearn(x_tile, sk_model, k=7, batch_patches=8192):
    # tile -> extract VALID patches -> predict proba -> reshape to (H-2r, W-2r)
    x = tf.convert_to_tensor(x_tile, tf.float32)[None, ...]  # (1,H,W,C)
    patches = tf.image.extract_patches(
        images=x, sizes=[1, k, k, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID"
    )
    Hc = int(tf.shape(patches)[1].numpy()); Wc = int(tf.shape(patches)[2].numpy())
    C  = int(tf.shape(x)[-1].numpy())
    patches = tf.reshape(patches, [-1, k*k*C]).numpy().astype(np.float32, copy=False)

    n = patches.shape[0]
    bs = int(batch_patches)
    outs = []
    for s in range(0, n, bs):
        p = sklearn_predict_proba(sk_model, patches[s:s+bs])
        outs.append(p)
    pflat = np.concatenate(outs, axis=0)
    return pflat.reshape(Hc, Wc).astype(np.float32, copy=False)

def full_eval_metrics_stream_sklearn(file_pattern, stats, sk_model, k=7, thr_prob=0.5, max_tiles=None, batch_patches=8192):
    ds = tile_dataset(file_pattern, stats=stats, max_tiles=max_tiles)
    r = int(k) // 2
    TP = FP = FN = TN = 0
    N_valid = 0

    for x, prev_raw, ndvi_raw, y_raw in ds:
        valid = valid_core_mask(prev_raw, y_raw, ndvi_raw, k=int(k))
        prob_core = _infer_core_prob_map_valid_sklearn(x, sk_model, k=int(k), batch_patches=int(batch_patches))

        y_core = y_raw[r:-r, r:-r] if r > 0 else y_raw
        v_core = valid[r:-r, r:-r] if r > 0 else valid

        yb = tf.boolean_mask(tf.cast(y_core > 0.0, tf.int32), v_core).numpy().astype(np.int32, copy=False)
        pb = tf.boolean_mask(tf.convert_to_tensor(prob_core), v_core).numpy().astype(np.float32, copy=False)

        if yb.size == 0:
            continue

        tp, fp, fn, tn = _confusion_from_probs(yb, pb, thr=float(thr_prob))
        TP += tp; FP += fp; FN += fn; TN += tn
        N_valid += int(yb.shape[0])

    m = _metrics_from_conf(TP, FP, FN, TN)
    return {**m, "TP": TP, "FP": FP, "FN": FN, "TN": TN, "N_valid": N_valid, "thr_prob": float(thr_prob)}
