"""Environment / hardware report recorded with every run (reproducibility requirement)."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from . import __version__
from . import config as _config
from .io_utils import utc_now, write_json

TRACKED_PACKAGES = (
    "torch", "transformers", "datasets", "tokenizers", "accelerate", "evaluate",
    "optimum", "onnx", "onnxruntime", "openvino", "huggingface_hub", "sentencepiece",
    "numpy", "pandas", "pyarrow", "pyarabic", "datasketch", "PyYAML", "rouge_score",
    "bert_score", "gradio", "psutil",
)


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except Exception:
        return None


def _cpu_model() -> str:
    if sys.platform == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2)
    except (ValueError, OSError, AttributeError):
        return None


def _git_commit() -> str | None:
    pkg_dir = Path(__file__).resolve().parent
    return _run(["git", "-C", str(pkg_dir), "rev-parse", "HEAD"])


def _gpu() -> dict:
    info: dict = {"available": False}
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if smi:
        name, mem, driver = [x.strip() for x in smi.splitlines()[0].split(",")]
        info.update(available=True, name=name, memory=mem, driver=driver, count=len(smi.splitlines()))
    try:
        import torch  # optional

        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:
        pass
    return info


def environment_report() -> dict:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return {
        "timestamp_utc": utc_now(),
        "arabic_summ_version": __version__,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu(),
        "in_colab": "COLAB_RELEASE_TAG" in os.environ,
        "packages": versions,
    }


def format_report(rep: Mapping) -> str:
    g = rep["gpu"]
    gpu = f"{g.get('name')} ({g.get('memory')}), CUDA {g.get('torch_cuda')}, bf16={g.get('bf16_supported')}" if g.get("available") else "none"
    lines = [
        f"arabic_summ {rep['arabic_summ_version']}  commit={rep['git_commit']}",
        f"Python {rep['python']}  |  {rep['os']}",
        f"CPU {rep['cpu_model']} x{rep['cpu_count_logical']}  |  RAM {rep['ram_gb']} GB",
        f"GPU {gpu}",
        "Packages: " + ", ".join(f"{k}={v}" for k, v in rep["packages"].items()),
    ]
    return "\n".join(lines)


def save_environment(cfg: Mapping, tag: str) -> dict:
    """Write env_<tag>_<ts>.json and pip_freeze_<tag>_<ts>.txt under logs/."""
    rep = environment_report()
    ts = rep["timestamp_utc"].replace(":", "").replace("-", "")[:15]
    write_json(_config.path(cfg, "logs", f"env_{tag}_{ts}.json"), rep)
    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    if freeze:
        _config.path(cfg, "logs", f"pip_freeze_{tag}_{ts}.txt").write_text(freeze + "\n", encoding="utf-8")
    return rep
