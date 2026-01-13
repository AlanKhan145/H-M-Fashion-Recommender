from __future__ import annotations
import argparse
from pathlib import Path

import yt_dlp

from .config import load_config
from .downloader import download_all

def main():
    p = argparse.ArgumentParser(prog="ytdlp_batch")
    p.add_argument("--config", default="config.txt", help="Path to config.txt")
    p.add_argument("--version", action="store_true", help="Print version info and exit")
    args = p.parse_args()

    if args.version:
        print("yt-dlp:", yt_dlp.version.__version__)
        return

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    res = download_all(cfg)

    print("\n=== SUMMARY ===")
    print("Success files:", len(res.downloaded_files))
    print("Failed URLs:", len(res.failed_urls))

    if res.downloaded_files:
        print("\nDOWNLOADED_FILES:")
        for f in res.downloaded_files:
            print(" -", f)

    if res.failed_urls:
        print("\nFAILED_URLS:")
        for u in res.failed_urls:
            print(" -", u)
            print("   reason:", res.failed_reasons.get(u, "unknown"))

if __name__ == "__main__":
    main()
