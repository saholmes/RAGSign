"""Apple-Silicon-native LLM backend (MLX).

Companion to :mod:`rag_sign.llm`'s ``LlamaCppLLM``.  Where llama.cpp is
the cross-platform GGUF baseline, this backend uses Apple's MLX
framework directly so that:

* The model lives in unified memory (no CPU↔GPU copies on Apple
  Silicon).
* Q4 quantisation halves memory pressure on the Mac Mini, putting
  7B-class models on a 16 GB box and 13B-class on a 24 GB box.
* LoRA fine-tuning is available via the ``mlx_lm`` tuner — unlike
  ``AirLLM``-style layer streaming, which is inference-only.
* Final-layer logits and the parameter tree are accessible directly,
  which the model-fingerprint sweep needs.

Lazy-imports ``mlx`` and ``mlx_lm`` so the module is importable even
on a vanilla install; the heavy native deps are only required when
:class:`MlxLLM` is actually instantiated.

Install via the ``mlx`` extra::

    uv pip install 'rag-sign[mlx]'

Examples
--------
End-to-end behavioural fingerprint of a Q4-quantised 3B model::

    from rag_sign.llm_mlx import MlxLLM

    llm = MlxLLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
    logits = llm.last_token_logits("The capital of France is")
    print(logits.shape)   # (vocab_size,) numpy float32

    # Then feed the iterator into rag_sign.model_fingerprint.behavioral_fingerprint.

Static weight-space SimHash::

    from rag_sign.model_fingerprint import simhash_arrays
    fp = simhash_arrays(llm.iter_param_arrays(), seed=b"enrol", dim=512)
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from rag_sign.llm import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

if TYPE_CHECKING:  # pragma: no cover — imported for typing only
    pass


_INSTALL_HINT: Final[str] = (
    "MlxLLM requires the 'mlx' extra (Apple Silicon only): "
    "uv pip install 'rag-sign[mlx]'"
)


class MlxLLM:
    """LLM backend backed by Apple's MLX framework.

    Parameters
    ----------
    model_id:
        Either an MLX-formatted HuggingFace repo id (e.g.
        ``"mlx-community/Llama-3.2-3B-Instruct-4bit"``) or a local
        path containing the converted weights + tokenizer.  ``mlx_lm``
        caches converted models under ``~/.cache/huggingface/hub``.
    eos_token:
        Optional override for the EOS token used as a generation
        stop-string.  Defaults to whatever the tokenizer reports.

    The constructor performs the (potentially slow) model load.  For
    repeated fingerprint sweeps, instantiate once and reuse — both
    :meth:`last_token_logits` and :meth:`iter_param_arrays` operate
    on the in-memory model.
    """

    def __init__(
        self,
        model_id: str,
        *,
        eos_token: str | None = None,
    ) -> None:
        try:
            from mlx_lm import load  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — exercised only on non-MLX hosts
            raise RuntimeError(_INSTALL_HINT) from exc

        self._model_id = model_id
        self._model, self._tokenizer = load(model_id)
        self._eos = eos_token if eos_token is not None else self._tokenizer.eos_token

    # ------------------------------------------------------------------
    # LLMBackend Protocol
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Produce a single completion for ``prompt`` (LLMBackend Protocol)."""
        from mlx_lm import generate  # type: ignore[import-not-found]

        return str(
            generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )
        ).strip()

    # ------------------------------------------------------------------
    # Fingerprint primitives
    # ------------------------------------------------------------------

    def last_token_logits(self, prompt: str) -> np.ndarray:
        """Final-position logits for ``prompt`` as a numpy float32 vector.

        Pairs with :func:`rag_sign.model_fingerprint.behavioral_fingerprint`
        — feed each probe through this method, collect the resulting
        arrays, then SimHash the concatenation.

        Returns
        -------
        ``np.ndarray`` of shape ``(vocab_size,)``, dtype ``float32``.
        """
        import mlx.core as mx  # type: ignore[import-not-found]

        ids = mx.array(self._tokenizer.encode(prompt))[None, :]  # (1, T)
        logits = self._model(ids)                                # (1, T, V)
        # Last position only.  ``np.array(..., copy=False)`` works
        # because MLX exposes the buffer protocol via ``__array__``.
        return np.array(logits[0, -1, :], copy=False).astype(np.float32, copy=False)

    def iter_param_arrays(self) -> Iterator[np.ndarray]:
        """Yield each parameter tensor as a numpy array (dequantised view).

        For Q4-quantised models the underlying MLX tensors are stored
        as packed nibbles plus per-block scales; the ``np.array(...)``
        cast materialises the dequantised float view, which is what
        the SimHash projection should see (we want the *behaviour*
        of the deployed weights, not the storage layout).
        """
        from mlx.utils import tree_flatten  # type: ignore[import-not-found]

        for _, p in tree_flatten(self._model.parameters()):
            yield np.array(p, copy=False)

    def iter_named_param_arrays(self) -> Iterator[tuple[str, np.ndarray]]:
        """Like :meth:`iter_param_arrays` but with the parameter name.

        Pairs with :func:`rag_sign.model_fingerprint.simhash_per_tensor`
        when an audit needs to attribute drift to a specific layer.
        """
        from mlx.utils import tree_flatten  # type: ignore[import-not-found]

        for name, p in tree_flatten(self._model.parameters()):
            yield name, np.array(p, copy=False)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """The HuggingFace id (or local path) the model was loaded from."""
        return self._model_id

    @property
    def tokenizer(self) -> Any:
        """The MLX-LM tokenizer.  Useful for the LoRA fine-tune driver
        which needs to tokenise the training corpus into the same
        vocabulary the model expects."""
        return self._tokenizer

    @property
    def model(self) -> Any:
        """The underlying ``mlx.nn.Module``.  Power-users only — the
        public surface is :meth:`last_token_logits` /
        :meth:`iter_param_arrays`; reach for ``model`` directly only
        when you need something neither helper exposes."""
        return self._model

    def parameter_count(self) -> int:
        """Total number of scalar parameters across the model."""
        return sum(int(np.prod(p.shape)) for p in self.iter_param_arrays())


# ---------------------------------------------------------------------------
# LoRA fine-tune driver
# ---------------------------------------------------------------------------


def lora_finetune(
    base_model_id: str,
    corpus: list[str],
    *,
    iters: int,
    out_dir: Path,
    rank: int = 8,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    seed: int = 0,
) -> Path:
    """Fine-tune ``base_model_id`` with LoRA on ``corpus``; return the
    fused merged-model path.

    Drives ``python -m mlx_lm.lora --train`` and then
    ``python -m mlx_lm.fuse`` via subprocess, which is the documented
    public API for MLX LoRA training.  In-process Python APIs exist
    in the ``mlx_lm.tuner`` module but their surface is less stable
    across mlx-lm releases — subprocess insulates us from API churn.

    The returned path can be passed straight back into
    :class:`MlxLLM` to fingerprint the fine-tuned model.

    Parameters
    ----------
    base_model_id:
        HuggingFace id (or local path) of the pre-quantised base.
        Example: ``"mlx-community/Llama-3.2-3B-Instruct-4bit"``.
    corpus:
        Training texts.  Written verbatim as one ``{"text": "..."}``
        record per JSONL line; tokenisation happens inside mlx-lm.
    iters:
        Number of LoRA training iterations.  Pass 0 to skip
        training entirely (returns the base model id, useful for
        the n_steps=0 row in a sweep).
    out_dir:
        Directory under which the adapter and the merged checkpoint
        are written.  Subdirectories ``data/``, ``adapter/`` and
        ``merged/`` are created.
    rank, learning_rate, batch_size, seed:
        Forwarded as ``--lora-rank`` / ``--learning-rate`` /
        ``--batch-size`` / ``--seed`` to the mlx-lm CLI.

    Returns
    -------
    ``Path`` to the merged-model directory (when ``iters > 0``) or
    a string equal to ``base_model_id`` (when ``iters == 0``).
    """
    import subprocess

    if iters <= 0:
        # Caller is asking for the n_steps=0 baseline — no LoRA
        # adapter to fuse.  Hand back the base id so the caller can
        # construct ``MlxLLM(base_model_id)`` uniformly.
        return Path(base_model_id) if Path(base_model_id).exists() else base_model_id  # type: ignore[return-value]

    out_dir = Path(out_dir)
    data_dir   = out_dir / "data"
    adapter_dir = out_dir / "adapter"
    merged_dir  = out_dir / "merged"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # mlx-lm expects train.jsonl + valid.jsonl in the data directory.
    # The 100-sentence demo corpus has no meaningful train/val split,
    # so valid is identical to train; this only matters for the
    # validation-loss readout, not for the trained weights themselves.
    train_jsonl = data_dir / "train.jsonl"
    valid_jsonl = data_dir / "valid.jsonl"
    payload = "\n".join(json.dumps({"text": s}) for s in corpus) + "\n"
    train_jsonl.write_text(payload, encoding="utf-8")
    valid_jsonl.write_text(payload, encoding="utf-8")

    # mlx-lm 0.20+ exposes LoRA hyperparameters (rank, dropout, scale)
    # *only* through a YAML config file passed via `-c`, not via
    # individual CLI flags.  Write a minimal config in the cell's
    # workdir and reference it.  CLI args (--iters, --batch-size, ...)
    # override matching YAML keys, so we keep the rest as flags.
    #
    # The default `scale` is 20.0 in mlx-lm's `CONFIG_DEFAULTS`; we
    # mirror it so the rank=8 default-equivalent path produces
    # numerically-identical results to a no-`-c` invocation.
    config_path = out_dir / "lora_config.yaml"
    config_path.write_text(
        "fine_tune_type: lora\n"
        "lora_parameters:\n"
        f"  rank: {rank}\n"
        "  dropout: 0.0\n"
        "  scale: 20.0\n",
        encoding="utf-8",
    )

    train_cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--train",
        "-c",              str(config_path),
        "--model",         base_model_id,
        "--data",          str(data_dir),
        "--iters",         str(iters),
        "--batch-size",    str(batch_size),
        "--learning-rate", str(learning_rate),
        "--adapter-path",  str(adapter_dir),
        "--seed",          str(seed),
        "--num-layers",    "-1",      # apply LoRA to all eligible layers
        "--save-every",    str(max(iters, 1)),
        # Disable validation; our corpus is tiny and val=train anyway.
        "--steps-per-eval", str(iters * 2),
        "--val-batches",   "0",
    ]
    subprocess.run(train_cmd, check=True)

    fuse_cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model",        base_model_id,
        "--adapter-path", str(adapter_dir),
        "--save-path",    str(merged_dir),
    ]
    subprocess.run(fuse_cmd, check=True)

    return merged_dir
