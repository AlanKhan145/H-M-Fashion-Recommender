from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

from wildfire.utils.io import load_yaml, ensure_dir, write_text, read_text, dump_json
from wildfire.utils.seed import seed_all, best_effort_gpu_memory_growth
from wildfire.data.download import download_dataset, split_patterns
from wildfire.data.stats import compute_train_stats
from wildfire.data.tfrecord import load_stats
from wildfire.train.hpo import successive_halving_hpo
from wildfire.train.gnn_train import run_train_trial, cleanup_tf
from wildfire.train.sklearn_train import train_lr, train_rf
from wildfire.eval.streaming import full_eval_metrics_stream_keras, full_eval_metrics_stream_sklearn

def main():
    paths = load_yaml("configs/paths.yaml")
    run_cfg = load_yaml("configs/run_all.yaml")["run"]

    artifacts_dir = ensure_dir(paths["artifacts_dir"])
    data_dir = ensure_dir(paths["data_dir"])
    reports_dir = ensure_dir(paths["reports_dir"])
    ensure_dir(Path(reports_dir) / "plots")

    seed_all(42)
    best_effort_gpu_memory_growth()

    # 0) download
    if run_cfg.get("download_data", True):
        ds_root = download_dataset()
        write_text(paths["dataset_path_file"], str(ds_root))
        print("Dataset path:", ds_root)
    else:
        ds_root = Path(read_text(paths["dataset_path_file"]))

    pats = split_patterns(ds_root)
    train_pat, val_pat, test_pat = pats["train"], pats["val"], pats["test"]

    # 1) stats
    if run_cfg.get("compute_stats", True):
        compute_train_stats(train_pat, save_json=paths["stats_json"], force=False)
        print("Saved stats:", paths["stats_json"])
    stats = load_stats(paths["stats_json"])

    # 2) GNN HPO
    best_cfg = None
    if run_cfg.get("gnn_hpo", True):
        hpo_cfg = load_yaml("configs/gnn_hpo.yaml")
        best_cfg, df = successive_halving_hpo(
            train_pat=train_pat, val_pat=val_pat, stats=stats,
            budgets=hpo_cfg["budgets"],
            n_initial=hpo_cfg["n_initial"], eta=hpo_cfg["eta"],
            seed=hpo_cfg["seed"], verbose_fit=hpo_cfg["verbose_fit"],
            thr_prob=0.5
        )
        ensure_dir("artifacts/gnn")
        df.to_csv("artifacts/gnn/hpo_trials.csv", index=False)
        dump_json("artifacts/gnn/best_cfg.json", best_cfg or {})
        print("BEST_CFG(HPO):", best_cfg)
    else:
        # fallback đọc từ gnn_final.yaml
        best_cfg = load_yaml("configs/gnn_final.yaml")["best_cfg"]

    # 3) GNN final
    model_final = None
    final_row = None
    if run_cfg.get("gnn_final", True):
        gfinal = load_yaml("configs/gnn_final.yaml")
        best_cfg = best_cfg or gfinal["best_cfg"]
        budget = gfinal["final_budget"]
        thr_prob = float(gfinal.get("thr_prob_default", 0.7))

        cleanup_tf()
        summ, model_final = run_train_trial(best_cfg, budget, stats, train_pat, val_pat, thr_prob=thr_prob, verbose_fit=1)
        final_row = {"status": "ok", **best_cfg, **budget, **summ}
        ensure_dir("artifacts/gnn")
        model_final.save("artifacts/gnn/model_final.keras")
        dump_json("artifacts/gnn/final_metrics.json", final_row)
        print("GNN FINAL SUMMARY:", {k: final_row.get(k) for k in ["best_val_iou","best_val_f1","best_val_precision","best_val_recall","best_val_acc_stream"]})

    # 4) LR
    lr_model = None
    if run_cfg.get("lr", True):
        lr_cfg = load_yaml("configs/lr.yaml")
        k = int(lr_cfg["k_patch"])
        lr_model = train_lr(train_pat, stats, k, lr_cfg)
        ensure_dir("artifacts/lr")
        joblib.dump(lr_model, "artifacts/lr/model.joblib")

        m_val = full_eval_metrics_stream_sklearn(val_pat, stats, lr_model, k=k, thr_prob=0.7, max_tiles=64, batch_patches=8192)
        dump_json("artifacts/lr/metrics.json", m_val)
        print("LR VAL:", m_val)

    # 5) RF
    rf_model = None
    if run_cfg.get("rf", True):
        rf_cfg = load_yaml("configs/rf.yaml")
        k = int(rf_cfg["k_patch"])
        rf_model = train_rf(train_pat, stats, k, rf_cfg)
        ensure_dir("artifacts/rf")
        joblib.dump(rf_model, "artifacts/rf/model.joblib")

        m_val = full_eval_metrics_stream_sklearn(val_pat, stats, rf_model, k=k, thr_prob=0.7, max_tiles=64, batch_patches=8192)
        dump_json("artifacts/rf/metrics.json", m_val)
        print("RF VAL:", m_val)

    # 6) quick compare report
    rows = []
    if model_final is not None:
        rows.append({"method":"PatchGNN", **full_eval_metrics_stream_keras(val_pat, stats, model_final, k=int(best_cfg["k_patch"]), thr_prob=0.7, max_tiles=64)})
    if lr_model is not None:
        rows.append({"method":"LogReg", **full_eval_metrics_stream_sklearn(val_pat, stats, lr_model, k=int(load_yaml("configs/lr.yaml")["k_patch"]), thr_prob=0.7, max_tiles=64)})
    if rf_model is not None:
        rows.append({"method":"RandomForest", **full_eval_metrics_stream_sklearn(val_pat, stats, rf_model, k=int(load_yaml("configs/rf.yaml")["k_patch"]), thr_prob=0.7, max_tiles=64)})

    if rows:
        df_cmp = pd.DataFrame(rows).sort_values("IoU", ascending=False)
        df_cmp.to_csv("artifacts/reports/compare_val.csv", index=False)
        print("\nSaved compare report -> artifacts/reports/compare_val.csv")

if __name__ == "__main__":
    main()
