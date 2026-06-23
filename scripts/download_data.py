"""Download a starter dataset into data/raw/ (cross-platform, stdlib only).

Default is Karpathy's tiny Shakespeare (~1.1 MB). Works on Windows / macOS /
Linux without curl.

Usage:
    python scripts/download_data.py                 # tiny shakespeare
    python scripts/download_data.py --url <URL> --out data/raw/custom.txt
"""
import argparse
import sys
import urllib.request
from pathlib import Path

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download(url: str, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"   -> {out}")
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    size = out.stat().st_size
    print(f"Done. {size:,} bytes written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=SHAKESPEARE_URL)
    parser.add_argument("--out", default="data/raw/shakespeare.txt")
    args = parser.parse_args()
    download(args.url, args.out)
