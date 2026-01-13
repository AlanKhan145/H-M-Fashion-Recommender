from __future__ import annotations
import json
from pathlib import Path
import yaml

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_text(p: str | Path) -> str:
    return Path(p).read_text(encoding="utf-8").strip()

def write_text(p: str | Path, s: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(s, encoding="utf-8")

def load_yaml(p: str | Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_json(p: str | Path, obj) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(p: str | Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
