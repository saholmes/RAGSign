"""Pluggable LLM backend.

Two implementations:

* :class:`EchoLLM` — deterministic, no-dependency stub used in tests.
  Returns a string that *contains* the retrieved-context strings, so
  the RAG end-to-end test can assert that the right chunks made it
  into the generation prompt.

* :class:`LlamaCppLLM` — wraps ``llama-cpp-python`` for local
  inference.  Llama 3.2 2B is the default per the paper (§6.1); other
  GGUF models work as long as you point ``LlamaCppLLM`` at the file.
  Lazy-imports the heavy native dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

DEFAULT_TEMPERATURE: Final[float] = 0.2  # paper-aligned: low for factuality
DEFAULT_MAX_TOKENS: Final[int] = 512


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface every LLM backend must satisfy."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str: ...
    """Produce a single completion for ``prompt``."""


# ---------------------------------------------------------------------------
# Echo backend (deterministic — for tests / dev)
# ---------------------------------------------------------------------------


class EchoLLM:
    """No-op stand-in that returns its prompt verbatim.

    Useful for asserting that the right context made it into the
    prompt assembly step (the prompt itself becomes the "answer", so
    the test can grep for chunk substrings inside the signed payload).
    """

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        return prompt[: max_tokens * 4]  # approximate ~4 chars/token


# ---------------------------------------------------------------------------
# Llama.cpp backend (Llama 3.2 / 3.1 via GGUF)
# ---------------------------------------------------------------------------


class LlamaCppLLM:  # pragma: no cover — requires native llama-cpp install
    """Llama 3.2 (or compatible GGUF) backend via ``llama-cpp-python``.

    Pass the GGUF model path on construction.  Defaults are tuned for
    the paper's ``Llama 3.2 2B`` (Q4_K_M quantisation) running on a
    laptop CPU; tune ``n_ctx`` / ``n_threads`` for your hardware.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,  # offload all if a GPU is available
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "LlamaCppLLM requires the 'llm' extra: "
                "uv pip install 'rag-sign[llm]'"
            ) from exc

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        result = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>"],
        )
        # llama_cpp returns OpenAI-shaped completion dicts.
        return str(result["choices"][0]["text"]).strip()


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT: Final[str] = (
    "You are an assistant that answers questions strictly from the supplied "
    "context.  Cite the source of each fact in square brackets after the "
    "sentence that uses it.  If the context does not answer the question, "
    "say so plainly."
)


def assemble_prompt(question: str, contexts: Sequence[str]) -> str:
    """Build a deterministic chat-style prompt from contexts and a question.

    The prompt format matters for two reasons:

    * The same ``(question, contexts)`` pair must always produce the
      same prompt — so a re-issue of the same query under the same
      corpus produces a verifiable signature over a *known* string.
    * Context order is the retrieval-score order; we don't shuffle.
    """
    ctx_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    return (
        f"<|system|>\n{_SYSTEM_PROMPT}\n"
        f"<|context|>\n{ctx_block}\n"
        f"<|user|>\n{question}\n"
        f"<|assistant|>\n"
    )
