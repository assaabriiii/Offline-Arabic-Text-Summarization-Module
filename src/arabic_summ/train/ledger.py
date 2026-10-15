"""GPU-hour ledger (proposal: "actual usage, GPU type and experiment are recorded").

Each training/tuning session writes one JSON file under logs/gpu_ledger/ and refreshes it on a
heartbeat, so a Colab disconnect still leaves a record whose `end_utc` is the last heartbeat.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from pathlib import Path
from typing import Mapping

from .. import config as C
from ..io_utils import read_json, utc_now, write_json


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


class GPUSession:
    def __init__(self, cfg: Mapping, experiment: str, notebook: str, min_interval_s: float = 60.0, **meta):
        self.path = C.path(cfg, "logs", "gpu_ledger", f"{utc_now()[:19].replace(':', '')}_{uuid.uuid4().hex[:6]}.json")
        self.t0 = time.time()
        self.min_interval = min_interval_s
        self._last = 0.0
        self.record = {"experiment": experiment, "notebook": notebook, "device": _gpu_name(),
                       "start_utc": utc_now(), "end_utc": None, "hours": 0.0, "status": "running", **meta}
        self.beat(force=True)

    def beat(self, force: bool = False, **update) -> None:
        now = time.time()
        if not force and now - self._last < self.min_interval and not update:
            return
        self._last = now
        self.record.update(update)
        self.record["end_utc"] = utc_now()
        self.record["hours"] = round((now - self.t0) / 3600, 4)
        write_json(self.path, self.record)

    def close(self, status: str = "finished", **update) -> None:
        self.beat(force=True, status=status, **update)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close("finished" if exc_type is None else f"failed: {exc_type.__name__}")
        return False


def summary(cfg: Mapping) -> dict:
    d = C.path(cfg, "logs", "gpu_ledger", "x").parent
    recs = [read_json(p) for p in sorted(Path(d).glob("*.json"))]
    by_dev: dict[str, float] = {}
    by_exp: dict[str, float] = {}
    for r in recs:
        if r.get("device", "cpu") == "cpu":
            continue
        by_dev[r["device"]] = round(by_dev.get(r["device"], 0) + r["hours"], 3)
        by_exp[r["experiment"]] = round(by_exp.get(r["experiment"], 0) + r["hours"], 3)
    return {"sessions": len(recs), "gpu_hours_total": round(sum(by_dev.values()), 3),
            "by_device": by_dev, "by_experiment": by_exp}
