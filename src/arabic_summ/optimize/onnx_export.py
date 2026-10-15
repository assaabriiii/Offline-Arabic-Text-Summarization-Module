"""ONNX export + dynamic INT8 quantization with optimum (run inside the ONNX environment)."""

from __future__ import annotations

import shutil
from pathlib import Path

COPY = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "sentencepiece.bpe.model", "spiece.model", "vocab.json", "merges.txt",
        "decoding.json", "provenance.json")


def export_fp32(model_dir: str | Path, out_dir: str | Path) -> list[str]:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ORTModelForSeq2SeqLM.from_pretrained(str(model_dir), export=True).save_pretrained(out)
    # tokenizer files are copied unchanged (re-saving with transformers 4.x could alter them)
    for name in COPY:
        if (Path(model_dir) / name).exists():
            shutil.copy2(Path(model_dir) / name, out / name)
    _write_runtime(out, "onnx_fp32")
    return sorted(p.name for p in out.glob("*.onnx"))


def _write_runtime(d: Path, variant: str) -> None:
    import json

    rt = d / "runtime.json"
    data = json.loads(rt.read_text(encoding="utf-8")) if rt.exists() else {}
    data["variant"] = variant
    rt.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def quantize_int8(fp32_dir: str | Path, out_dir: str | Path, arch: str = "avx2") -> list[str]:
    """Dynamic INT8 per ONNX file; quantized files keep the FP32 file names so the folder loads
    with the same ORTModelForSeq2SeqLM.from_pretrained call."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    src, out = Path(fp32_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qcfg = getattr(AutoQuantizationConfig, arch)(is_static=False, per_channel=False)
    tmp = out / "_tmp"
    names = sorted(p.name for p in src.glob("*.onnx"))
    for name in names:
        ORTQuantizer.from_pretrained(src, file_name=name).quantize(save_dir=tmp, quantization_config=qcfg)
        q = next(tmp.glob(name.replace(".onnx", "*quantized*.onnx")))
        q.replace(out / name)
    shutil.rmtree(tmp, ignore_errors=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix != ".onnx" and not p.name.endswith(".onnx_data") and not (out / p.name).exists():
            shutil.copy2(p, out / p.name)
    _write_runtime(out, "onnx_int8")
    return names
