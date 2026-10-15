"""Runtime-agnostic seq2seq interface. PyTorch / ONNX Runtime / OpenVINO backends implement it,
so the pipeline and MVP switch runtime through config only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class GenerationRequest:
    text: str
    min_new_tokens: int
    max_new_tokens: int


class Seq2SeqBackend(Protocol):
    name: str
    max_input_tokens: int

    def count_tokens(self, text: str) -> int: ...

    def generate(self, requests: Sequence[GenerationRequest], decoding: Mapping) -> list[str]: ...


class EchoBackend:
    """Test double: returns the first N words of the input, honoring the token bounds
    with 1 token == 1 word. Lets pipeline logic be tested without a model."""

    name = "echo"

    def __init__(self, max_input_tokens: int = 1024):
        self.max_input_tokens = max_input_tokens
        self.calls: list[GenerationRequest] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, requests, decoding):
        self.calls.extend(requests)
        out = []
        for r in requests:
            words = r.text.split()
            n = min(len(words), max(r.min_new_tokens, (r.min_new_tokens + r.max_new_tokens) // 2))
            out.append(" ".join(words[:n]))
        return out
