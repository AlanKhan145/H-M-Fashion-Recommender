from __future__ import annotations
from pathlib import Path
import kagglehub

def download_dataset() -> Path:
    p = kagglehub.dataset_download("fantineh/next-day-wildfire-spread")
    return Path(p)

def split_patterns(dataset_root: Path) -> dict:
    return {
        "train": str(dataset_root / "next_day_wildfire_spread_train*"),
        "val":   str(dataset_root / "next_day_wildfire_spread_eval*"),
        "test":  str(dataset_root / "next_day_wildfire_spread_test*"),
    }
