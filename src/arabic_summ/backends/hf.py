"""PyTorch / Hugging Face Transformers backend (training-time default and FP32 reference)."""

from __future__ import annotations

from typing import Mapping, Sequence

from .base import GenerationRequest

GEN_KEYS = ("num_beams", "length_penalty", "no_repeat_ngram_size", "early_stopping",
            "repetition_penalty", "do_sample")


class HFBackend:
    def __init__(self, model_path: str, revision: str | None = None, device: str | None = None,
                 dtype: str = "float32", max_input_tokens: int = 1024, batch_size: int = 4,
                 local_files_only: bool = False, num_threads: int | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM

        from ..models import load_tokenizer

        if num_threads:
            torch.set_num_threads(num_threads)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(model_path, revision=revision, local_files_only=local_files_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, revision=revision, local_files_only=local_files_only,
            dtype=getattr(torch, dtype),
        ).to(self.device).eval()
        self.name = f"hf:{model_path}"
        # learned absolute positions (BART) cap both input and output length; T5 has none
        self.max_positions = getattr(self.model.config, "max_position_embeddings", None)
        self.max_input_tokens = min(max_input_tokens, self.max_positions or max_input_tokens)
        self.batch_size = batch_size

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, requests: Sequence[GenerationRequest], decoding: Mapping) -> list[str]:
        gen = {k: decoding[k] for k in GEN_KEYS if k in decoding}
        out: list[str | None] = [None] * len(requests)
        # only requests with identical length bounds share a batch, so each keeps its own bounds
        groups: dict[tuple[int, int], list[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault((r.min_new_tokens, r.max_new_tokens), []).append(i)
        for (lo, hi), idx in groups.items():
            for j in range(0, len(idx), self.batch_size):
                part = idx[j : j + self.batch_size]
                enc = self.tokenizer([requests[k].text for k in part], return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.max_input_tokens).to(self.device)
                with self.torch.inference_mode():
                    cap = (self.max_positions - 2) if self.max_positions else hi
                    ids = self.model.generate(**enc, **gen, min_new_tokens=min(lo, cap), max_new_tokens=min(hi, cap))
                for k, text in zip(part, self.tokenizer.batch_decode(ids, skip_special_tokens=True)):
                    out[k] = text.strip()
        return out  # type: ignore[return-value]
