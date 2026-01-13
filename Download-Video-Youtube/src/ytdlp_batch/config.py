from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AppConfig:
    out_dir: Path
    cookies: Path | None
    urls_file: Path
    po_token: str | None

def _strip_quotes(v: str) -> str:
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1].strip()
    return v

def read_kv_txt(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Bad line (need key=value): {raw}")
        k, v = line.split("=", 1)
        cfg[k.strip()] = _strip_quotes(v)
    return cfg

def load_config(config_path: Path) -> AppConfig:
    cfg = read_kv_txt(config_path)

    out_dir = Path(cfg.get("out_dir", "./downloads")).expanduser().resolve()
    urls_file = Path(cfg.get("urls_file", "./urls.txt")).expanduser().resolve()

    cookies_raw = (cfg.get("cookies") or "").strip()
    cookies = Path(cookies_raw).expanduser().resolve() if cookies_raw else None

    po_token_raw = (cfg.get("po_token") or "").strip()
    po_token = po_token_raw if po_token_raw else None

    return AppConfig(out_dir=out_dir, cookies=cookies, urls_file=urls_file, po_token=po_token)
