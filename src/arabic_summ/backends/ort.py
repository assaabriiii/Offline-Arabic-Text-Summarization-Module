"""ONNX Runtime backend (CPU) through optimum's ORTModelForSeq2SeqLM.

optimum-onnx 0.1.0 requires transformers 4.x, while training uses transformers 5, so ONNX
export and ONNX inference run in a separate environment (requirements/colab-onnx.txt) and are
driven from notebooks through subprocesses. The generation logic is shared with HFBackend.
"""

from __future__ import annotations

from .hf import HFBackend


class ORTBackend(HFBackend):
    def __init__(self, model_dir: str, max_input_tokens: int = 1024, batch_size: int = 1,
                 num_threads: int | None = None):
        import onnxruntime as ort
        import torch
        from optimum.onnxruntime import ORTModelForSeq2SeqLM

        from ..models import load_tokenizer

        so = ort.SessionOptions()
        if num_threads:
            so.intra_op_num_threads = num_threads
            torch.set_num_threads(num_threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.torch = torch
        self.device = "cpu"
        self.tokenizer = load_tokenizer(model_dir, local_files_only=True)
        self.model = ORTModelForSeq2SeqLM.from_pretrained(model_dir, provider="CPUExecutionProvider",
                                                          session_options=so, local_files_only=True)
        self.name = f"onnx:{model_dir}"
        self._set_limits(self.model.config, max_input_tokens, batch_size)
