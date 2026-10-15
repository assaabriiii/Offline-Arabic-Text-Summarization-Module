"""Candidate CPU variants of one fine-tuned model.

* torch_fp32 - PyTorch FP32 (reference)
* torch_int8 - PyTorch dynamic INT8 on nn.Linear (no export step)
* onnx_fp32  - ONNX Runtime FP32 (optimum export, graph optimizations)
* onnx_int8  - ONNX Runtime dynamic INT8 (MatMul weights int8, AVX2 kernels)
"""

from __future__ import annotations

from pathlib import Path

VARIANTS = {
    "torch_fp32": {"runtime": "torch", "quantize": None},
    "torch_int8": {"runtime": "torch", "quantize": "dynamic_int8"},
    "onnx_fp32": {"runtime": "onnx", "quantize": None},
    "onnx_int8": {"runtime": "onnx", "quantize": "dynamic_int8"},
}


def model_dir_for(variant: str, torch_dir: Path, onnx_root: Path) -> Path:
    v = VARIANTS[variant]
    if v["runtime"] == "torch":
        return Path(torch_dir)
    return Path(onnx_root) / ("int8" if v["quantize"] else "fp32")


def load_backend(variant: str, model_dir: str | Path, num_threads: int | None, max_input_tokens: int = 1024):
    v = VARIANTS[variant]
    if v["runtime"] == "torch":
        from ..backends.hf import HFBackend

        return HFBackend(str(model_dir), device="cpu", dtype="float32", batch_size=1, local_files_only=True,
                         num_threads=num_threads, quantize=v["quantize"], max_input_tokens=max_input_tokens)
    from ..backends.ort import ORTBackend

    return ORTBackend(str(model_dir), max_input_tokens=max_input_tokens, batch_size=1, num_threads=num_threads)


def dir_size_mb(d: str | Path, pattern: str = "*") -> float:
    return round(sum(p.stat().st_size for p in Path(d).glob(pattern) if p.is_file()) / 2**20, 1)


def weights_size_mb(d: str | Path) -> float:
    """Model weights only (safetensors / bin / onnx + external data), the proposal's 'model size'."""
    d = Path(d)
    exts = {".safetensors", ".bin", ".onnx", ".onnx_data", ".data"}
    return round(sum(p.stat().st_size for p in d.iterdir() if p.is_file() and p.suffix in exts) / 2**20, 1)
