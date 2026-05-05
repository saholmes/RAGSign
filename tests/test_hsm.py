"""Tests for :mod:`rag_sign.hsm`.

We only exercise the in-memory backend here.  The PKCS#11 backend is
integration-tested under ``scripts/run_softhsm_tests.sh`` which spins
up a SoftHSM token via Docker.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from rag_sign.hsm import (
    SECRET_LEN,
    HSMBackend,
    InMemoryHSM,
    default_backend,
)


class InMemoryHSMTests(unittest.TestCase):
    def test_satisfies_protocol(self) -> None:
        self.assertIsInstance(InMemoryHSM(), HSMBackend)

    def test_enrol_returns_unique_handles(self) -> None:
        h = InMemoryHSM()
        handles = [h.enrol() for _ in range(16)]
        self.assertEqual(len(set(handles)), len(handles))

    def test_fetch_returns_secret_of_expected_length(self) -> None:
        h = InMemoryHSM()
        handle = h.enrol()
        secret = h.fetch(handle)
        self.assertEqual(len(secret), SECRET_LEN)

    def test_fetch_is_idempotent(self) -> None:
        h = InMemoryHSM()
        handle = h.enrol()
        self.assertEqual(h.fetch(handle), h.fetch(handle))

    def test_distinct_enrolments_have_distinct_secrets(self) -> None:
        h = InMemoryHSM()
        s1 = h.fetch(h.enrol())
        s2 = h.fetch(h.enrol())
        self.assertNotEqual(s1, s2)

    def test_unknown_handle_raises(self) -> None:
        h = InMemoryHSM()
        with self.assertRaises(KeyError):
            h.fetch(b"\x00" * 16)

    def test_attestation_is_stable_string(self) -> None:
        h = InMemoryHSM()
        att = h.attestation()
        self.assertIsInstance(att, bytes)
        self.assertGreater(len(att), 0)
        # Identifies the backend uniquely.
        self.assertIn(b"in-memory", att)


class DefaultBackendTests(unittest.TestCase):
    def test_returns_inmemory_when_env_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_SIGN_HSM_LIBRARY", None)
            self.assertIsInstance(default_backend(), InMemoryHSM)


if __name__ == "__main__":
    unittest.main()
