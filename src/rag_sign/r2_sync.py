"""Cloudflare R2 (or any S3-compatible) sync helpers.

Pairs the experiment harness with cloud object storage so that
rented GPU hosts can pull the IACR text cache (or raw PDFs) without
the local filesystem having to host the data first.

Design notes:

* R2 is **S3-compatible**, so we use ``boto3`` and just point the
  ``endpoint_url`` at Cloudflare's R2 endpoint.  No CF-specific SDK
  needed.
* All operations are **idempotent and resumable** — files already
  present locally with the right size are skipped, and a partial
  sync can be re-run safely.
* The module exposes :func:`sync_to_local` (download) and
  :func:`upload_directory` (upload).  Both take an ``r2://``
  URI and a local path.
* Credentials and endpoint live in environment variables; the
  module never persists them.

Required environment variables for any operation:

  * ``R2_ACCESS_KEY_ID``       — R2 access key
  * ``R2_SECRET_ACCESS_KEY``   — R2 secret key

Plus exactly one of:

  * ``R2_ENDPOINT_URL``        — full URL, e.g.
    ``https://<account>.r2.cloudflarestorage.com``
  * ``R2_ACCOUNT_ID``          — account ID; endpoint is constructed
    automatically

URI format
----------

``r2://<bucket>[/<prefix>]``

The prefix is optional.  Prefixes are recursive — a sync from
``r2://corpus/iacr_text/`` walks every key under that prefix.

Optional dependency
-------------------

This module imports ``boto3`` lazily at first use.  ``boto3`` is
listed in the ``cloud`` extra::

    uv pip install 'rag-sign[cloud]'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# All boto3 imports are deferred to call sites so that the rest of
# the package works in environments where boto3 is not installed.

R2_URI_SCHEME = "r2"


def parse_r2_uri(uri: str) -> tuple[str, str]:
    """Parse ``r2://bucket[/prefix]`` into ``(bucket, prefix)``.

    The prefix is returned without a leading slash and *with* a
    trailing slash if the URI ended with one (so callers can use it
    directly as an S3 ``Prefix`` argument).
    """
    parsed = urlparse(uri)
    if parsed.scheme != R2_URI_SCHEME:
        raise ValueError(
            f"expected r2:// URI, got {parsed.scheme!r} in {uri!r}"
        )
    bucket = parsed.netloc
    if not bucket:
        raise ValueError(f"r2:// URI must have a bucket: {uri!r}")
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def _r2_endpoint() -> str:
    """Resolve the R2 endpoint URL from environment."""
    endpoint = os.environ.get("R2_ENDPOINT_URL")
    if endpoint:
        return endpoint
    account = os.environ.get("R2_ACCOUNT_ID")
    if not account:
        raise RuntimeError(
            "R2 access requires either R2_ENDPOINT_URL or "
            "R2_ACCOUNT_ID to be set in the environment"
        )
    return f"https://{account}.r2.cloudflarestorage.com"


def _r2_client():  # type: ignore[no-untyped-def]
    """Construct a boto3 S3 client pointed at R2."""
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "rag_sign.r2_sync requires the 'cloud' extra: "
            "uv pip install 'rag-sign[cloud]'"
        ) from exc

    access = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise RuntimeError(
            "R2 access requires R2_ACCESS_KEY_ID and "
            "R2_SECRET_ACCESS_KEY to be set in the environment"
        )

    return boto3.client(
        "s3",
        endpoint_url=_r2_endpoint(),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="auto",  # R2 ignores region but boto3 wants one set
    )


def sync_to_local(
    uri: str,
    local_root: str | Path,
    *,
    progress: bool = True,
) -> dict:
    """Sync every object under ``uri`` to ``local_root``.

    Skips any file already present locally with the matching size.
    Returns a small status dict with counts of (downloaded, skipped,
    bytes_downloaded).
    """
    bucket, prefix = parse_r2_uri(uri)
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    client = _r2_client()
    paginator = client.get_paginator("list_objects_v2")

    n_downloaded = 0
    n_skipped = 0
    bytes_downloaded = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            size = obj["Size"]
            # Strip the prefix to compute the relative path under local_root.
            rel = key[len(prefix):].lstrip("/") if prefix else key
            if not rel:
                continue   # zero-length key (folder marker)
            local = local_root / rel
            if local.exists() and local.stat().st_size == size:
                n_skipped += 1
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local))
            n_downloaded += 1
            bytes_downloaded += size
            if progress and n_downloaded % 200 == 0:
                print(
                    f"  …{n_downloaded} files  "
                    f"({bytes_downloaded / 1e6:.0f} MB)",
                    file=sys.stderr,
                )

    if progress:
        print(
            f"R2 sync {uri} → {local_root}: "
            f"downloaded {n_downloaded}, skipped {n_skipped}, "
            f"{bytes_downloaded / 1e6:.0f} MB transferred",
            file=sys.stderr,
        )
    return {
        "uri": uri,
        "local_root": str(local_root),
        "downloaded": n_downloaded,
        "skipped": n_skipped,
        "bytes_downloaded": bytes_downloaded,
    }


def upload_directory(
    local_root: str | Path,
    uri: str,
    *,
    file_glob: str = "**/*",
    progress: bool = True,
) -> dict:
    """Upload every file under ``local_root`` to ``uri``.

    Mirrors the relative structure under ``uri``'s prefix.  Useful
    for shipping ``bench_results/*.json`` back to R2 after a remote
    run.
    """
    bucket, prefix = parse_r2_uri(uri)
    local_root = Path(local_root)
    if not local_root.is_dir():
        raise NotADirectoryError(f"local_root not a directory: {local_root}")

    client = _r2_client()
    n_uploaded = 0
    bytes_uploaded = 0

    for path in sorted(local_root.glob(file_glob)):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root).as_posix()
        key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
        client.upload_file(str(path), bucket, key)
        size = path.stat().st_size
        n_uploaded += 1
        bytes_uploaded += size
        if progress and n_uploaded % 50 == 0:
            print(
                f"  …{n_uploaded} files  "
                f"({bytes_uploaded / 1e6:.0f} MB)",
                file=sys.stderr,
            )

    if progress:
        print(
            f"R2 upload {local_root} → {uri}: "
            f"{n_uploaded} files, {bytes_uploaded / 1e6:.0f} MB",
            file=sys.stderr,
        )
    return {
        "uri": uri,
        "local_root": str(local_root),
        "uploaded": n_uploaded,
        "bytes_uploaded": bytes_uploaded,
    }


def is_r2_uri(value: str | os.PathLike[str]) -> bool:
    """Cheap predicate for env-var values that look like R2 URIs."""
    return str(value).startswith(f"{R2_URI_SCHEME}://")


def maybe_resolve(
    env_value: str | os.PathLike[str] | None,
    local_default: str | Path,
    *,
    label: str = "R2 source",
) -> Path:
    """Resolve an env-var value that may be either a local path or
    an ``r2://`` URI.

    If the value is an R2 URI, the contents are synced to the
    ``local_default`` path (creating it if needed) and that path is
    returned.  If the value is a local path, it is returned
    unchanged.  If the value is ``None``, the default is returned
    unchanged with no sync.

    The two-step pattern (sync once, operate on the local mirror)
    keeps the rest of the loader simple — every existing file-system
    walk continues to work without modification.
    """
    if env_value is None:
        return Path(local_default)
    s = str(env_value)
    if is_r2_uri(s):
        target = Path(local_default)
        print(f"{label}: syncing {s} → {target}", file=sys.stderr)
        sync_to_local(s, target)
        return target
    return Path(s)
