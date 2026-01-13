from __future__ import annotations
import gc
import numpy as np
import tensorflow as tf
import keras
from keras import callbacks

from ..models.patch_gnn import build_patch_gnn, WeightedBCEFromLogits
from ..data.constants import C_IN
from ..data.tfrecord import make_train_patch_dataset
from ..eval.streaming import full_eval_metrics_stream_keras

def cleanup_tf():
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

class StreamSplitMetrics(callbacks.Callback):
    def __init__(self, prefix, file_pattern, stats, k, max_tiles=64, batch_patches=8192, thr_prob=0.5, every=1, verbose=1):
        super().__init__()
        self.prefix = str(prefix)
        self.file_pattern = file_pattern
        self.stats = stats
        self.k = int(k)
        self.max_tiles = max_tiles
        self.batch_patches = int(batch_patches)
        self.thr_prob = float(thr_prob)
        self.every = int(every)
        self.verbose = int(verbose)
        self.hist = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.every != 0:
            return
        res = full_eval_metrics_stream_keras(
            self.file_pattern, stats=self.stats, model=self.model,
            k=self.k, thr_prob=self.thr_prob,
            max_tiles=self.max_tiles, batch_patches=self.batch_patches
        )
        logs[f"{self.prefix}_iou"] = res["IoU"]
        logs[f"{self.prefix}_f1"]  = res["F1"]
        logs[f"{self.prefix}_precision"] = res["Precision"]
        logs[f"{self.prefix}_recall"]    = res["Recall"]
        logs[f"{self.prefix}_acc_stream"] = res["Accuracy"]
        self.hist.append({"epoch": epoch+1, **res})
        if self.verbose:
            print(f"[Stream{self.prefix.upper()}] ep={epoch+1:03d} IoU={res['IoU']:.4f} F1={res['F1']:.4f} "
                  f"P={res['Precision']:.4f} R={res['Recall']:.4f} Acc={res['Accuracy']*100:.2f}% n={res['N_valid']}")

def build_compile_model(cfg: dict):
    model = build_patch_gnn(
        k=int(cfg["k_patch"]),
        c=C_IN,
        d_hidden=int(cfg["d_hidden"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
        use_pos_emb=True
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(cfg["lr"])),
        loss=WeightedBCEFromLogits(pos_weight=float(cfg["pos_weight"])),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.0, name="acc_patch")],
    )
    return model

def run_train_trial(cfg: dict, budget: dict, stats: dict, train_pat: str, val_pat: str, thr_prob=0.5, verbose_fit=0):
    train_ds = make_train_patch_dataset(
        train_pat, stats=stats,
        batch_size=int(budget["batch_train"]),
        k=int(cfg["k_patch"]),
        n_pos=int(budget["train_pos"]),
        n_neg=int(budget["train_neg"]),
        shuffle_files=True,
        repeat=True
    )
    model = build_compile_model(cfg)

    monitor = str(budget.get("monitor", "val_f1")).lower()
    cb_val = StreamSplitMetrics(
        prefix="val", file_pattern=val_pat, stats=stats, k=int(cfg["k_patch"]),
        max_tiles=int(budget.get("val_max_tiles", 64)),
        batch_patches=int(budget.get("val_batch_patches", 8192)),
        thr_prob=float(thr_prob), every=int(budget.get("val_every", 1)),
        verbose=1 if verbose_fit else 0
    )

    cb_tr = StreamSplitMetrics(
        prefix="tr", file_pattern=train_pat, stats=stats, k=int(cfg["k_patch"]),
        max_tiles=int(budget.get("train_eval_max_tiles", 32)),
        batch_patches=int(budget.get("train_eval_batch_patches", budget.get("val_batch_patches", 8192))),
        thr_prob=float(thr_prob), every=int(budget.get("train_eval_every", 1)),
        verbose=1 if verbose_fit else 0
    )

    cbs = [
        cb_tr,
        cb_val,
        callbacks.EarlyStopping(monitor=monitor, mode="max", patience=int(budget.get("es_patience", 3)),
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor=monitor, mode="max", factor=0.5, patience=int(budget.get("rlrop_patience", 2)),
                                   min_lr=1e-6, verbose=0),
    ]

    hist = model.fit(
        train_ds,
        steps_per_epoch=int(budget["steps_per_epoch"]),
        epochs=int(budget["epochs"]),
        callbacks=cbs,
        verbose=verbose_fit
    )

    def _best(cb_hist, key="F1"):
        return max(cb_hist, key=lambda r: r.get(key, -1.0)) if cb_hist else None

    best_val = _best(cb_val.hist, "F1")
    best_tr  = _best(cb_tr.hist, "F1")

    summary = {
        "best_tr_f1": best_tr["F1"] if best_tr else np.nan,
        "best_tr_iou": best_tr["IoU"] if best_tr else np.nan,
        "best_tr_precision": best_tr["Precision"] if best_tr else np.nan,
        "best_tr_recall": best_tr["Recall"] if best_tr else np.nan,
        "best_tr_acc_stream": best_tr["Accuracy"] if best_tr else np.nan,

        "best_val_f1": best_val["F1"] if best_val else np.nan,
        "best_val_iou": best_val["IoU"] if best_val else np.nan,
        "best_val_precision": best_val["Precision"] if best_val else np.nan,
        "best_val_recall": best_val["Recall"] if best_val else np.nan,
        "best_val_acc_stream": best_val["Accuracy"] if best_val else np.nan,

        "epochs_ran": int(len(hist.history.get("loss", []))),
        "monitor": monitor,
        "thr_prob": float(thr_prob),
    }
    return summary, model
