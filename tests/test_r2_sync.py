"""Tests for the pure-logic helpers in :mod:`rag_sign.r2_sync`.

Network-touching tests (real R2 sync) are out of scope here — the
``cloud`` extra is optional and the boto3 paths are exercised in
the integration tests under ``scripts/cloud_bootstrap.sh``.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from rag_sign.r2_sync import (
    is_r2_uri,
    maybe_resolve,
    parse_r2_uri,
)


class ParseUriTests(unittest.TestCase):
    def test_bucket_only(self) -> None:
        b, p = parse_r2_uri("r2://iacr-text/")
        self.assertEqual(b, "iacr-text")
        self.assertEqual(p, "")

    def test_bucket_and_prefix(self) -> None:
        b, p = parse_r2_uri("r2://iacr-text/2018/")
        self.assertEqual(b, "iacr-text")
        self.assertEqual(p, "2018/")

    def test_deep_prefix(self) -> None:
        b, p = parse_r2_uri("r2://corpus/iacr_text/2020/")
        self.assertEqual(b, "corpus")
        self.assertEqual(p, "iacr_text/2020/")

    def test_no_trailing_slash(self) -> None:
        b, p = parse_r2_uri("r2://corpus/iacr_text")
        self.assertEqual(b, "corpus")
        self.assertEqual(p, "iacr_text")

    def test_rejects_wrong_scheme(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected r2://"):
            parse_r2_uri("s3://bucket/key")

    def test_rejects_missing_bucket(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have a bucket"):
            parse_r2_uri("r2://")


class IsUriTests(unittest.TestCase):
    def test_recognises_uri(self) -> None:
        self.assertTrue(is_r2_uri("r2://bucket/key"))

    def test_rejects_local_path(self) -> None:
        self.assertFalse(is_r2_uri("/Volumes/data/iacr"))
        self.assertFalse(is_r2_uri("./relative/path"))

    def test_rejects_other_schemes(self) -> None:
        self.assertFalse(is_r2_uri("s3://bucket"))
        self.assertFalse(is_r2_uri("http://example.com"))


class MaybeResolveTests(unittest.TestCase):
    """``maybe_resolve`` is the hook the loader uses to choose between
    a local path and a remote sync.  We test only the local branch
    here — the R2 branch is exercised end-to-end in the bootstrap
    integration tests.
    """

    def test_none_returns_default(self) -> None:
        out = maybe_resolve(None, "/tmp/whatever")
        self.assertEqual(out, Path("/tmp/whatever"))

    def test_local_path_passes_through(self) -> None:
        out = maybe_resolve("/Users/x/data", "/tmp/default")
        self.assertEqual(out, Path("/Users/x/data"))

    def test_r2_uri_triggers_sync(self) -> None:
        # Patch sync_to_local so we don't touch the network.
        with mock.patch("rag_sign.r2_sync.sync_to_local") as fake:
            out = maybe_resolve(
                "r2://corpus/iacr_text/", "/tmp/iacr_local"
            )
            fake.assert_called_once_with(
                "r2://corpus/iacr_text/", Path("/tmp/iacr_local")
            )
        self.assertEqual(out, Path("/tmp/iacr_local"))


class EndpointResolutionTests(unittest.TestCase):
    """``_r2_endpoint`` reads either R2_ENDPOINT_URL or R2_ACCOUNT_ID."""

    def test_explicit_endpoint_wins(self) -> None:
        from rag_sign.r2_sync import _r2_endpoint
        with mock.patch.dict(
            os.environ,
            {
                "R2_ENDPOINT_URL": "https://custom.example.com",
                "R2_ACCOUNT_ID": "ignored-account",
            },
            clear=True,
        ):
            self.assertEqual(_r2_endpoint(), "https://custom.example.com")

    def test_account_id_derives_endpoint(self) -> None:
        from rag_sign.r2_sync import _r2_endpoint
        with mock.patch.dict(
            os.environ,
            {"R2_ACCOUNT_ID": "abc123"},
            clear=True,
        ):
            self.assertEqual(
                _r2_endpoint(),
                "https://abc123.r2.cloudflarestorage.com",
            )

    def test_neither_raises(self) -> None:
        from rag_sign.r2_sync import _r2_endpoint
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "R2_ENDPOINT_URL"),
        ):
            _r2_endpoint()


if __name__ == "__main__":
    unittest.main()
