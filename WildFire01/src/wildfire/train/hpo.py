from __future__ import annotations
import math, gc
import numpy as np
import pandas as pd

from .gnn_train import run_train_trial, cleanup_tf

def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def _approx_cost(cfg):
    k = int(cfg["k_patch"]); L = int(cfg["n_layers"]); d = int(cfg["d_hidden"])
    return (k*k)*L*(d*d)

def sample_configs(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    K_CHOICES    = np.array([5, 7, 9, 11])
    L_CHOICES    = np.array([4, 6, 8, 10])
    D_CHOICES    = np.array([96, 128, 160, 192])
    DROP_CHOICES = np.array([0.0, 0.05, 0.1])
    POSW_CHOICES = np.array([1.0, 2.0, 4.0, 9.0])
    LR_MIN, LR_MAX = 2e-5, 3e-4
    COST_CAP = 11*11*10*(192*192)

    cfgs = []
    while len(cfgs) < n:
        cfg = {
            "k_patch": int(rng.choice(K_CHOICES)),
            "n_layers": int(rng.choice(L_CHOICES)),
            "d_hidden": int(rng.choice(D_CHOICES)),
            "dropout": float(rng.choice(DROP_CHOICES)),
            "lr": _loguniform(rng, LR_MIN, LR_MAX),
            "pos_weight": float(rng.choice(POSW_CHOICES)),
        }
        if _approx_cost(cfg) <= COST_CAP:
            cfgs.append(cfg)
    return cfgs

def _trial_score(row):
    iou = row.get("best_val_iou", np.nan)
    f1  = row.get("best_val_f1", np.nan)
    acc = row.get("best_val_acc_stream", np.nan)
    if not (np.isfinite(iou) and np.isfinite(f1)):
        return np.nan
    score = float(iou) * (0.85 + 0.15 * float(f1))
    if np.isfinite(acc) and acc < 0.10:
        score *= 0.85
    return score

def successive_halving_hpo(
    train_pat: str,
    val_pat: str,
    stats: dict,
    budgets: list[dict],
    n_initial=64,
    eta=4,
    seed=42,
    verbose_fit=0,
    thr_prob=0.5
):
    alive = sample_configs(int(n_initial), seed=int(seed))
    all_rows = []

    for rung, budget in enumerate(budgets):
        # đảm bảo min epochs = 4
        if int(budget.get("epochs", 0)) < 4:
            budget["epochs"] = 4

        print(f"\n===== HPO RUNG {rung} | alive={len(alive)} =====")
        scored = []

        for i, cfg in enumerate(alive, 1):
            cleanup_tf()
            try:
                summ, _model = run_train_trial(cfg, budget, stats, train_pat, val_pat, thr_prob=thr_prob, verbose_fit=verbose_fit)
                row = {"status": "ok", "rung": rung, **cfg, **budget, **summ}
            except Exception as e:
                row = {"status": "error", "rung": rung, "error": f"{type(e).__name__}: {str(e)[:200]}", **cfg, **budget}
            row["score"] = _trial_score(row)
            all_rows.append(row)

            ok = (row["status"] == "ok") and np.isfinite(row["score"])
            print(f"  [{i:02d}/{len(alive):02d}] {'ok' if ok else 'fail'} k={cfg['k_patch']} L={cfg['n_layers']} d={cfg['d_hidden']} score={row['score']}")
            if ok:
                scored.append((row["score"], cfg))

        df = pd.DataFrame(all_rows)
        if not scored:
            return None, df[df["status"].astype(str).eq("ok")].copy()

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = max(1, math.ceil(len(scored) / int(eta)))
        alive = [cfg for _, cfg in scored[:keep]]
        print(f"--> keep={keep} | best_score={scored[0][0]:.4f}")

    df = pd.DataFrame(all_rows)
    df_ok = df[df["status"].astype(str).eq("ok")].copy()
    df_ok = df_ok.sort_values("score", ascending=False).reset_index(drop=True)

    best_cfg = None
    if len(df_ok) > 0:
        r0 = df_ok.iloc[0]
        best_cfg = {
            "k_patch": int(r0["k_patch"]),
            "n_layers": int(r0["n_layers"]),
            "d_hidden": int(r0["d_hidden"]),
            "dropout": float(r0["dropout"]),
            "lr": float(r0["lr"]),
            "pos_weight": float(r0["pos_weight"]),
        }
    return best_cfg, df_ok
