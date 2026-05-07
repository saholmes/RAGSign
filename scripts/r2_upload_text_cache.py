"""One-off: upload a local IACR text cache to R2.

Used when bootstrapping a new R2 bucket from a workstation that
already has the extracted text cache.  Run this once locally; from
then on cloud GPU boxes can pull the cache via the bootstrap
script with no PDF parsing on the cloud side.

Usage:

    R2_ACCESS_KEY_ID=… R2_SECRET_ACCESS_KEY=… \\
    R2_ENDPOINT_URL=https://abcd.r2.cloudflarestorage.com \\
    .venv/bin/python -m scripts.r2_upload_text_cache \\
        --local ~/.cache/rag-sign/iacr_text \\
        --uri r2://iacr-text-cache/

Idempotent: re-running skips files already present in R2 with
matching size (the equivalent ``sync_to_local`` behaviour).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_sign.r2_sync import upload_directory  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--local", required=True,
        help="local directory to upload (e.g. ~/.cache/rag-sign/iacr_text)",
    )
    p.add_argument(
        "--uri", required=True,
        help="r2://bucket[/prefix] destination",
    )
    p.add_argument(
        "--glob", default="**/*",
        help="file glob relative to --local (default: every file)",
    )
    args = p.parse_args()

    local = Path(args.local).expanduser()
    if not local.is_dir():
        print(f"FATAL: --local is not a directory: {local}", file=sys.stderr)
        return 2

    upload_directory(local, args.uri, file_glob=args.glob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
