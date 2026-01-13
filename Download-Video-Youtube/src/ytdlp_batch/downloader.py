from __future__ import annotations
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from yt_dlp import YoutubeDL

from .config import AppConfig

@dataclass
class DownloadResult:
    downloaded_files: list[str]
    failed_urls: list[str]
    failed_reasons: dict[str, str]

def read_urls_file(path: Path) -> list[str]:
    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        u = raw.strip()
        if not u or u.startswith("#"):
            continue
        # remove si=... param (optional)
        u = re.sub(r"([?&])si=[^&]+", r"\1", u).rstrip("?&")
        urls.append(u)
    return urls

def locate_deno() -> str | None:
    # prefer PATH first
    if shutil.which("deno"):
        return "deno"
    # fallback common install path
    p = Path.home() / ".deno" / "bin" / "deno"
    return str(p) if p.exists() else None

def build_ydl_opts(cfg: AppConfig, deno_path: str | None, hook):
    # Player clients:
    # - If you provide PO token -> mweb is commonly used
    # - Otherwise -> web/web_embedded/tv
    player_clients = ["mweb"] if cfg.po_token else ["web", "web_embedded", "tv"]

    extractor_args = {"youtube": {"player_client": player_clients}}
    if cfg.po_token:
        extractor_args["youtube"]["po_token"] = [cfg.po_token]

    ydl_opts = {
        "format": "bv*[protocol^=https][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "outtmpl": str(cfg.out_dir / "%(title).200s [%(id)s].%(ext)s"),
        "merge_output_format": "mp4",
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 1,
        "continuedl": True,
        "ignoreerrors": False,

        # JS challenge solver (deno + ejs)
        "remote_components": {"ejs:github"},
        # NOTE: your yt-dlp expects dict format
        **({"js_runtimes": {"deno": {"path": deno_path}}} if deno_path else {}),

        # avoid android PO-token warning by default
        "extractor_args": extractor_args,

        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
        },

        "sleep_interval": 1,
        "max_sleep_interval": 3,

        "postprocessors": [{"key": "FFmpegMetadata"}],
        "progress_hooks": [hook],
        "quiet": False,
        "noprogress": False,
    }

    if cfg.cookies and cfg.cookies.exists():
        ydl_opts["cookiefile"] = str(cfg.cookies)

    return ydl_opts

def download_all(cfg: AppConfig) -> DownloadResult:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.urls_file.exists():
        raise FileNotFoundError(f"urls_file not found: {cfg.urls_file}")

    urls = read_urls_file(cfg.urls_file)
    if not urls:
        raise ValueError("urls_file is empty (no URLs).")

    deno_path = locate_deno()

    downloaded_files: list[str] = []
    failed_urls: list[str] = []
    failed_reasons: dict[str, str] = {}

    def hook(d):
        if d.get("status") == "finished":
            fn = d.get("filename")
            if fn and os.path.exists(fn) and fn not in downloaded_files:
                downloaded_files.append(fn)

    ydl_opts = build_ydl_opts(cfg, deno_path, hook)

    print("=== START ===")
    print("Save to:", cfg.out_dir)
    print("Cookies:", str(cfg.cookies) if cfg.cookies else "(none)")
    print("PO token:", "(set)" if cfg.po_token else "(none)")
    print("Deno:", deno_path if deno_path else "(not found; JS challenge may fail)")
    print("URLs:", len(urls))

    with YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            print("\n------------------------------")
            print("Downloading:", url)
            try:
                ydl.download([url])
            except Exception as e:
                failed_urls.append(url)
                failed_reasons[url] = repr(e)
                print(">> FAILED:", repr(e))

    return DownloadResult(downloaded_files, failed_urls, failed_reasons)
