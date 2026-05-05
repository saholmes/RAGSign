"""Tests for the ECDSA-P256 signer / verifier built on a derived seed."""

from __future__ import annotations

import unittest

from rag_sign.key_derivation import KeyMaterial, derive_signing_seed
from rag_sign.signer import RagSigner, verify
from rag_sign.verifier import verify_message


def _km(tag: int = 0) -> KeyMaterial:
    return KeyMaterial(
        bytes(((i + tag) & 0xFF) for i in range(32)),
        bytes(((i + tag + 1) & 0xFF) for i in range(32)),
        bytes(((i + tag + 2) & 0xFF) for i in range(32)),
    )


class SignerRoundTripTests(unittest.TestCase):
    def test_sign_then_verify_passes(self) -> None:
        seed = derive_signing_seed(_km())
        signer = RagSigner(seed)
        msg = signer.sign(b"the answer is 42")
        self.assertTrue(verify_message(msg))
        self.assertTrue(verify(msg.payload, msg.signature, msg.public_key_pem))

    def test_tampered_payload_fails(self) -> None:
        seed = derive_signing_seed(_km())
        signer = RagSigner(seed)
        msg = signer.sign(b"original")
        self.assertFalse(verify(b"tampered", msg.signature, msg.public_key_pem))

    def test_tampered_signature_fails(self) -> None:
        seed = derive_signing_seed(_km())
        signer = RagSigner(seed)
        msg = signer.sign(b"original")
        bad_sig = bytearray(msg.signature)
        bad_sig[-1] ^= 0x01
        self.assertFalse(verify(msg.payload, bytes(bad_sig), msg.public_key_pem))

    def test_wrong_public_key_fails(self) -> None:
        sig_a = RagSigner(derive_signing_seed(_km(0))).sign(b"hello")
        wrong_pub = RagSigner(derive_signing_seed(_km(99))).public_key_pem
        self.assertFalse(verify(sig_a.payload, sig_a.signature, wrong_pub))


class DeterminismTests(unittest.TestCase):
    def test_same_seed_yields_same_public_key(self) -> None:
        seed = derive_signing_seed(_km())
        a = RagSigner(seed).public_key_pem
        b = RagSigner(seed).public_key_pem
        self.assertEqual(a, b)

    def test_different_seed_yields_different_public_key(self) -> None:
        a = RagSigner(derive_signing_seed(_km(0))).public_key_pem
        b = RagSigner(derive_signing_seed(_km(1))).public_key_pem
        self.assertNotEqual(a, b)


class SeedShapeTests(unittest.TestCase):
    def test_rejects_wrong_length_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "32 bytes"):
            RagSigner(b"too-short")


if __name__ == "__main__":
    unittest.main()
